## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.38370851399999995


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.9999943, -6.0890894, -7.9999943, -6.0890894, -1.3888612, 1.3888612)
1: (-8.6659203, -7.4363751, -8.6659203, -7.4363751, -0.7070894, 0.7070894)
2: (10.9171724, 11.8272743, 10.9171724, 11.8272743, -0.6135144, 0.6135144)
3: (-7.1265326, -5.6245265, -7.1265326, -5.6245265, -1.2651591, 1.2651591)
4: (-7.9751201, -6.4292212, -7.9751201, -6.4292212, -1.1643662, 1.1643662)
5: (-13.4319868, -11.7855387, -13.4319868, -11.7855387, -1.3121305, 1.3121300)
6: (-12.6905022, -10.7197313, -12.6905022, -10.7197313, -1.5428610, 1.5428610)
7: (-5.1237650, -3.5955453, -5.1237650, -3.5955453, -1.2241964, 1.2241964)
8: (-3.2843947, -2.1668067, -3.2843947, -2.1668067, -0.9989872, 0.9989872)
9: (-5.1203327, -3.6108992, -5.1203327, -3.6108992, -1.2323804, 1.2323804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 23.46 + 35.78 = 59.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3915383, upper bound: 0.3915381

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4629
type: A, layer: 1, pos: 6110

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 4629

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915376
time: 6.32 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915373, upper bound: 0.3915385
time: 6.70 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 13.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 13.27
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915376
NS_A2, status: Status.UNKNOWN, split count: 1, time: 13.27
Output dim: 2, lower bound: -0.3915373, upper bound: 0.3915385

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -7.9841719, -6.1114225, -7.9975190, -6.0949278, -1.3671460, 1.3644700
1: -8.6630688, -7.4411459, -8.6655350, -7.4376101, -0.7024815, 0.7019224
2: 10.9245663, 11.8253231, 10.9188461, 11.8272009, -0.6066878, 0.6087310
3: -7.1201982, -5.6306152, -7.1249795, -5.6258416, -1.2592249, 1.2576046
4: -7.9626741, -6.4872665, -7.9744415, -6.4443436, -1.1310387, 1.1062264
5: -13.4101553, -11.7965965, -13.4263916, -11.7870398, -1.2894330, 1.2948713
6: -12.6830950, -10.7319241, -12.6894464, -10.7228622, -1.5305238, 1.5291443
7: -5.1188321, -3.6201439, -5.1234989, -3.6019237, -1.2104588, 1.2003121
8: -3.2815256, -2.1717944, -3.2838736, -2.1679921, -0.9944539, 0.9938879
9: -5.1076508, -3.6356163, -5.1186171, -3.6173499, -1.2118673, 1.2055025

Time for backsubstitution: 21.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3909416
time: 5.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915373
time: 6.16 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -7.9999909, -6.0890999, -7.9999943, -6.0890894, -1.3888302, 1.3731675
1: -8.6659222, -7.4363766, -8.6659203, -7.4363751, -0.7070887, 0.7041106
2: 10.9171734, 11.8272743, 10.9171724, 11.8272743, -0.6138716, 0.6132891
3: -7.1265302, -5.6245279, -7.1265326, -5.6245265, -1.2661219, 1.2647018
4: -7.9751205, -6.4292331, -7.9751201, -6.4292212, -1.1639767, 1.1198440
5: -13.4319754, -11.7855415, -13.4319868, -11.7855387, -1.2963243, 1.3121281
6: -12.6905041, -10.7197304, -12.6905022, -10.7197313, -1.5476947, 1.5428553
7: -5.1237631, -3.5955470, -5.1237650, -3.5955453, -1.2232952, 1.2130084
8: -3.2843943, -2.1668072, -3.2843947, -2.1668067, -1.0035300, 0.9989839
9: -5.1203327, -3.6109035, -5.1203327, -3.6108992, -1.2323785, 1.2122574

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4629
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4629

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3909412
time: 7.01 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3915374
time: 6.82 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 36.16 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 36.16
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3909416
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 36.16
Output dim: 2, lower bound: -0.3909413, upper bound: 0.3915373
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 36.16
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3909412
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 36.16
Output dim: 2, lower bound: -0.3915374, upper bound: 0.3915374

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -7.9841719, -6.1114225, -7.9841719, -6.1114225, -1.3518219, 1.3518219
1: -8.6630688, -7.4411459, -8.6630688, -7.4411459, -0.6993170, 0.6993170
2: 10.9245663, 11.8253231, 10.9245663, 11.8253231, -0.6045914, 0.6045911
3: -7.1201982, -5.6306152, -7.1201982, -5.6306152, -1.2542639, 1.2542639
4: -7.9626741, -6.4872665, -7.9626741, -6.4872665, -1.0925612, 1.0925612
5: -13.4101553, -11.7965965, -13.4101553, -11.7965965, -1.2796631, 1.2796631
6: -12.6830950, -10.7319241, -12.6830950, -10.7319241, -1.5220690, 1.5220699
7: -5.1188321, -3.6201439, -5.1188321, -3.6201439, -1.1944981, 1.1944981
8: -3.2815256, -2.1717944, -3.2815256, -2.1717944, -0.9916377, 0.9916377
9: -5.1076508, -3.6356163, -5.1076508, -3.6356163, -1.1940746, 1.1940746

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6110

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 6110

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905981, upper bound: 0.3909413
time: 5.83 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909403, upper bound: 0.3909421
time: 7.31 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -7.9841719, -6.1114225, -7.9999909, -6.0890999, -1.3735957, 1.3670521
1: -8.6630688, -7.4411459, -8.6659222, -7.4363766, -0.7040529, 0.7023504
2: 10.9245663, 11.8253231, 10.9171734, 11.8272743, -0.6065395, 0.6113386
3: -7.1201982, -5.6306152, -7.1265302, -5.6245279, -1.2600403, 1.2589245
4: -7.9626741, -6.4872665, -7.9751205, -6.4292331, -1.1316476, 1.1070700
5: -13.4101553, -11.7965965, -13.4319754, -11.7855415, -1.2909822, 1.3008022
6: -12.6830950, -10.7319241, -12.6905041, -10.7197304, -1.5347714, 1.5295019
7: -5.1188321, -3.6201439, -5.1237631, -3.5955470, -1.2180605, 1.1997285
8: -3.2815256, -2.1717944, -3.2843943, -2.1668072, -0.9964814, 0.9935274
9: -5.1076508, -3.6356163, -5.1203327, -3.6109035, -1.2155647, 1.2074289

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6110

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905981, upper bound: 0.3915365
time: 7.67 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909403, upper bound: 0.3915366
time: 7.09 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -7.9999909, -6.0890999, -7.9841719, -6.1114225, -1.3670521, 1.3735957
1: -8.6659222, -7.4363766, -8.6630688, -7.4411459, -0.7023504, 0.7040529
2: 10.9171734, 11.8272743, 10.9245663, 11.8253231, -0.6113389, 0.6065397
3: -7.1265302, -5.6245279, -7.1201982, -5.6306152, -1.2589245, 1.2600403
4: -7.9751205, -6.4292331, -7.9626741, -6.4872665, -1.1070700, 1.1316476
5: -13.4319754, -11.7855415, -13.4101553, -11.7965965, -1.3008022, 1.2909827
6: -12.6905041, -10.7197304, -12.6830950, -10.7319241, -1.5295019, 1.5347719
7: -5.1237631, -3.5955470, -5.1188321, -3.6201439, -1.1997285, 1.2180605
8: -3.2843943, -2.1668072, -3.2815256, -2.1717944, -0.9935274, 0.9964814
9: -5.1203327, -3.6109035, -5.1076508, -3.6356163, -1.2074289, 1.2155647

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6110

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911993, upper bound: 0.3909416
time: 4.63 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915363, upper bound: 0.3909406
time: 5.74 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -7.9999909, -6.0890999, -7.9999909, -6.0890999, -1.3731647, 1.3731647
1: -8.6659222, -7.4363766, -8.6659222, -7.4363766, -0.7041104, 0.7041104
2: 10.9171734, 11.8272743, 10.9171734, 11.8272743, -0.6138711, 0.6138711
3: -7.1265302, -5.6245279, -7.1265302, -5.6245279, -1.2661200, 1.2661200
4: -7.9751205, -6.4292331, -7.9751205, -6.4292331, -1.1198425, 1.1198425
5: -13.4319754, -11.7855415, -13.4319754, -11.7855415, -1.2963219, 1.2963219
6: -12.6905041, -10.7197304, -12.6905041, -10.7197304, -1.5476894, 1.5476885
7: -5.1237631, -3.5955470, -5.1237631, -3.5955470, -1.2130051, 1.2130051
8: -3.2843943, -2.1668072, -3.2843943, -2.1668072, -1.0035267, 1.0035267
9: -5.1203327, -3.6109035, -5.1203327, -3.6109035, -1.2122564, 1.2122564

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6110

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6110

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911994, upper bound: 0.3909416
time: 4.77 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909406
time: 6.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.04 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3905981, upper bound: 0.3909413
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3909403, upper bound: 0.3909421
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3905981, upper bound: 0.3915365
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3909403, upper bound: 0.3915366
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3911993, upper bound: 0.3909416
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3915363, upper bound: 0.3909406
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3911994, upper bound: 0.3909416
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 33.04
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909406

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -7.9808340, -6.1114964, -7.9836020, -6.1114388, -1.3484678, 1.3511496
1: -8.6618271, -7.4411869, -8.6628571, -7.4411511, -0.6977603, 0.6987813
2: 10.9251566, 11.8253021, 10.9246674, 11.8253193, -0.6040111, 0.6044743
3: -7.1185160, -5.6306791, -7.1199112, -5.6306267, -1.2517581, 1.2533779
4: -7.9626474, -6.4913063, -7.9626684, -6.4879580, -1.0918436, 1.0884972
5: -13.4097099, -11.7967863, -13.4100742, -11.7966299, -1.2781143, 1.2783470
6: -12.6801157, -10.7320118, -12.6825848, -10.7319441, -1.5191207, 1.5215216
7: -5.1151991, -3.6204593, -5.1182117, -3.6201982, -1.1903253, 1.1931763
8: -3.2812071, -2.1729259, -3.2814713, -2.1719894, -0.9903708, 0.9893651
9: -5.1073589, -3.6373122, -5.1076016, -3.6359036, -1.1935716, 1.1923470

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905983, upper bound: 0.3905983
time: 6.04 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905983, upper bound: 0.3909413
time: 6.36 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -7.9859414, -6.1038542, -7.9841700, -6.1114264, -1.3524847, 1.3592458
1: -8.6631432, -7.4382448, -8.6630697, -7.4411454, -0.6995995, 0.7019255
2: 10.9236965, 11.8269310, 10.9245682, 11.8253241, -0.6053257, 0.6062107
3: -7.1211743, -5.6263132, -7.1201982, -5.6306148, -1.2558227, 1.2581711
4: -7.9710293, -6.4858484, -7.9626741, -6.4872694, -1.1009750, 1.0925822
5: -13.4115419, -11.7938528, -13.4101553, -11.7965984, -1.2793217, 1.2838821
6: -12.6842232, -10.7260094, -12.6830921, -10.7319260, -1.5227518, 1.5280457
7: -5.1189280, -3.6114116, -5.1188316, -3.6201446, -1.1945181, 1.2025752
8: -3.2852325, -2.1713867, -3.2815261, -2.1717978, -0.9944839, 0.9937005
9: -5.1120291, -3.6356018, -5.1076517, -3.6356165, -1.1983938, 1.1938162

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909410, upper bound: 0.3905985
time: 8.78 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909410, upper bound: 0.3909414
time: 6.20 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -7.9808340, -6.1114964, -7.9994226, -6.0891109, -1.3702402, 1.3663797
1: -8.6618271, -7.4411869, -8.6657085, -7.4363837, -0.7024963, 0.7018154
2: 10.9251566, 11.8253021, 10.9172745, 11.8272705, -0.6059599, 0.6112204
3: -7.1185160, -5.6306791, -7.1262431, -5.6245403, -1.2575336, 1.2580371
4: -7.9626474, -6.4913063, -7.9751148, -6.4299240, -1.1306877, 1.1030064
5: -13.4097099, -11.7967863, -13.4319019, -11.7855778, -1.2894325, 1.2994871
6: -12.6801157, -10.7320118, -12.6899910, -10.7197475, -1.5318232, 1.5289526
7: -5.1151991, -3.6204593, -5.1231413, -3.5956039, -1.2138896, 1.1984076
8: -3.2812071, -2.1729259, -3.2843375, -2.1670017, -0.9952145, 0.9912553
9: -5.1073589, -3.6373122, -5.1202812, -3.6111934, -1.2149515, 1.2057014

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905976, upper bound: 0.3911989
time: 9.51 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3905976, upper bound: 0.3915365
time: 10.24 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -7.9859414, -6.1038542, -7.9999866, -6.0890989, -1.3742576, 1.3744764
1: -8.6631432, -7.4382448, -8.6659203, -7.4363756, -0.7043355, 0.7049589
2: 10.9236965, 11.8269310, 10.9171734, 11.8272724, -0.6072741, 0.6129584
3: -7.1211743, -5.6263132, -7.1265287, -5.6245279, -1.2615981, 1.2628322
4: -7.9710293, -6.4858484, -7.9751205, -6.4292383, -1.1319485, 1.1070900
5: -13.4115419, -11.7938528, -13.4319763, -11.7855406, -1.2906413, 1.3050232
6: -12.6842232, -10.7260094, -12.6905012, -10.7197304, -1.5354538, 1.5354767
7: -5.1189280, -3.6114116, -5.1237631, -3.5955486, -1.2180815, 1.2078061
8: -3.2852325, -2.1713867, -3.2843933, -2.1668081, -0.9993277, 0.9955902
9: -5.1120291, -3.6356018, -5.1203308, -3.6109049, -1.2167010, 1.2071705

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3911988
time: 6.73 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3915365
time: 7.79 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -7.9966545, -6.0891705, -7.9836020, -6.1114388, -1.3636975, 1.3729234
1: -8.6646767, -7.4364185, -8.6628571, -7.4411511, -0.7007949, 0.7035184
2: 10.9177704, 11.8272514, 10.9246674, 11.8253193, -0.6107464, 0.6064231
3: -7.1248555, -5.6245956, -7.1199112, -5.6306267, -1.2564135, 1.2591567
4: -7.9750934, -6.4332705, -7.9626684, -6.4879580, -1.1063495, 1.1275744
5: -13.4315376, -11.7857361, -13.4100742, -11.7966299, -1.2992563, 1.2896609
6: -12.6875286, -10.7198181, -12.6825848, -10.7319441, -1.5265517, 1.5342245
7: -5.1201315, -3.5958669, -5.1182117, -3.6201982, -1.1955557, 1.2167397
8: -3.2840633, -2.1679406, -3.2814713, -2.1719894, -0.9922619, 0.9942079
9: -5.1200371, -3.6126008, -5.1076016, -3.6359036, -1.2069225, 1.2138343

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911988, upper bound: 0.3905978
time: 6.68 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911988, upper bound: 0.3909405
time: 6.35 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.0017624, -6.0815187, -7.9841700, -6.1114264, -1.3677540, 1.3810334
1: -8.6659975, -7.4334698, -8.6630697, -7.4411454, -0.7026346, 0.7066691
2: 10.9163208, 11.8288832, 10.9245682, 11.8253241, -0.6120572, 0.6081595
3: -7.1275101, -5.6202259, -7.1201982, -5.6306148, -1.2605290, 1.2639523
4: -7.9834752, -6.4278197, -7.9626741, -6.4872694, -1.1154814, 1.1316776
5: -13.4333687, -11.7827854, -13.4101553, -11.7965984, -1.3004799, 1.2952070
6: -12.6916399, -10.7138138, -12.6830921, -10.7319260, -1.5301886, 1.5407486
7: -5.1238594, -3.5868030, -5.1188316, -3.6201446, -1.1997471, 1.2242446
8: -3.2881126, -2.1664076, -3.2815261, -2.1717978, -0.9964085, 0.9985352
9: -5.1247120, -3.6108913, -5.1076517, -3.6356165, -1.2117510, 1.2153075

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915363, upper bound: 0.3905980
time: 5.77 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915363, upper bound: 0.3909405
time: 6.16 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -7.9966545, -6.0891705, -7.9994226, -6.0891109, -1.3698039, 1.3724885
1: -8.6646767, -7.4364185, -8.6657085, -7.4363837, -0.7025545, 0.7035754
2: 10.9177704, 11.8272514, 10.9172745, 11.8272705, -0.6132882, 0.6137543
3: -7.1248555, -5.6245956, -7.1262431, -5.6245403, -1.2636032, 1.2652373
4: -7.9750934, -6.4332705, -7.9751148, -6.4299240, -1.1191235, 1.1157784
5: -13.4315376, -11.7857361, -13.4319019, -11.7855778, -1.2947745, 1.2950015
6: -12.6875286, -10.7198181, -12.6899910, -10.7197475, -1.5447397, 1.5471401
7: -5.1201315, -3.5958669, -5.1231413, -3.5956039, -1.2088337, 1.2116852
8: -3.2840633, -2.1679406, -3.2843375, -2.1670017, -1.0022602, 1.0012531
9: -5.1200371, -3.6126008, -5.1202812, -3.6111934, -1.2117496, 1.2105279

Time for backsubstitution: 22.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3905988
time: 5.19 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909406
time: 7.34 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.0017624, -6.0815187, -7.9999866, -6.0890989, -1.3738623, 1.3805971
1: -8.6659975, -7.4334698, -8.6659203, -7.4363756, -0.7043941, 0.7067208
2: 10.9163208, 11.8288832, 10.9171734, 11.8272724, -0.6146212, 0.6154909
3: -7.1275101, -5.6202259, -7.1265287, -5.6245279, -1.2676969, 1.2700524
4: -7.9834752, -6.4278197, -7.9751205, -6.4292383, -1.1282640, 1.1198606
5: -13.4333687, -11.7827854, -13.4319763, -11.7855406, -1.2959795, 1.3005486
6: -12.6916399, -10.7138138, -12.6905012, -10.7197304, -1.5483770, 1.5536642
7: -5.1238594, -3.5868030, -5.1237631, -3.5955486, -1.2130260, 1.2210999
8: -3.2881126, -2.1664076, -3.2843933, -2.1668081, -1.0064087, 1.0055814
9: -5.1247120, -3.6108913, -5.1203308, -3.6109049, -1.2165771, 1.2119970

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6110

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6110

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3905988
time: 4.78 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909405
time: 5.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 32.92 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3905983, upper bound: 0.3905983
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3905983, upper bound: 0.3909413
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3909410, upper bound: 0.3905985
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3909410, upper bound: 0.3909414
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3905976, upper bound: 0.3911989
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3905976, upper bound: 0.3915365
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3911988
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3909404, upper bound: 0.3915365
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3911988, upper bound: 0.3905978
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3911988, upper bound: 0.3909405
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3915363, upper bound: 0.3905980
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3915363, upper bound: 0.3909405
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3905988
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3911989, upper bound: 0.3909406
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3905988
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 32.92
Output dim: 2, lower bound: -0.3915364, upper bound: 0.3909405

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -7.9808340, -6.1114964, -7.9808340, -6.1114964, -1.3483815, 1.3483815
1: -8.6618271, -7.4411869, -8.6618271, -7.4411869, -0.6975300, 0.6975300
2: 10.9251566, 11.8253021, 10.9251566, 11.8253021, -0.6039960, 0.6039960
3: -7.1185160, -5.6306791, -7.1185160, -5.6306791, -1.2513661, 1.2513661
4: -7.9626474, -6.4913063, -7.9626474, -6.4913063, -1.0884776, 1.0884776
5: -13.4097099, -11.7967863, -13.4097099, -11.7967863, -1.2772179, 1.2772174
6: -12.6801157, -10.7320118, -12.6801157, -10.7320118, -1.5190821, 1.5190821
7: -5.1151991, -3.6204593, -5.1151991, -3.6204593, -1.1898065, 1.1898065
8: -3.2812071, -2.1729259, -3.2812071, -2.1729259, -0.9886150, 0.9886150
9: -5.1073589, -3.6373122, -5.1073589, -3.6373122, -1.1921692, 1.1921692

Time for backsubstitution: 22.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 227
type: A, layer: 3, pos: 2132
type: A, layer: 3, pos: 1255
type: A, layer: 3, pos: 711
type: A, layer: 3, pos: 1096
type: A, layer: 3, pos: 1488
type: A, layer: 3, pos: 2511
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 1090
type: A, layer: 3, pos: 663
type: A, layer: 3, pos: 1236
type: A, layer: 3, pos: 31
type: A, layer: 3, pos: 1256
type: A, layer: 3, pos: 661
type: A, layer: 3, pos: 226
type: A, layer: 3, pos: 2822
type: A, layer: 3, pos: 37
type: A, layer: 3, pos: 2462
type: A, layer: 3, pos: 2613
type: A, layer: 3, pos: 2633
type: A, layer: 3, pos: 326
type: A, layer: 3, pos: 1202
type: A, layer: 3, pos: 2140
type: A, layer: 3, pos: 212
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 2922
type: A, layer: 3, pos: 1248
type: A, layer: 3, pos: 1229
type: A, layer: 3, pos: 2847
type: A, layer: 3, pos: 2142
type: A, layer: 3, pos: 2568
type: A, layer: 3, pos: 2380
type: A, layer: 3, pos: 3122

Time for candidate selection: 0.55 seconds

### Candidate
type: A, layer: 3, pos: 227

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3752878, upper bound: 0.3805453
time: 5.96 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3815102, upper bound: 0.3815104
time: 7.44 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.24 + 544.40 = 603.64 seconds
