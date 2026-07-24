## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 12)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.325746828


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-3.0641866, -2.2406847, -3.0641866, -2.2406847, -0.6628447, 0.6628447)
1: (-7.4437399, -6.4144506, -7.4437399, -6.4144506, -0.5975900, 0.5975902)
2: (-7.1122885, -6.3522425, -7.1122885, -6.3522425, -0.4296961, 0.4296960)
3: (-12.7483730, -11.7579308, -12.7483730, -11.7579308, -0.5116768, 0.5116767)
4: (-0.4580393, 0.4665954, -0.4580393, 0.4665954, -0.7241158, 0.7241156)
5: (-7.6689425, -6.8097653, -7.6689425, -6.8097653, -0.5543370, 0.5543370)
6: (0.4248850, 1.1547067, 0.4248850, 1.1547067, -0.4935509, 0.4935508)
7: (-4.9145422, -3.7904553, -4.9145422, -3.7904553, -0.7933307, 0.7933311)
8: (-0.8799987, -0.1870537, -0.8799987, -0.1870537, -0.5398700, 0.5398700)
9: (-5.6880784, -4.6347733, -5.6880784, -4.6347733, -0.6888375, 0.6888375)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.59 + 33.86 = 56.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3290372, upper bound: 0.3290372

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5758
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290352, upper bound: 0.3261551
time: 3.58 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290352, upper bound: 0.3290350
time: 3.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.53 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 6, lower bound: -0.3290352, upper bound: 0.3261551
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 6, lower bound: -0.3290352, upper bound: 0.3290350

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -3.0597932, -2.2427649, -3.0628166, -2.2407885, -0.6587472, 0.6583116
1: -7.4424210, -6.4198570, -7.4437180, -6.4160538, -0.5940342, 0.5917563
2: -7.1121569, -6.3548131, -7.1122837, -6.3529053, -0.4288998, 0.4269917
3: -12.7480383, -11.7607479, -12.7483006, -11.7585335, -0.5106988, 0.5086143
4: -0.4519433, 0.4588590, -0.4572639, 0.4640163, -0.7156477, 0.7156639
5: -7.6673298, -6.8119040, -7.6687894, -6.8104429, -0.5518863, 0.5519552
6: 0.4309227, 1.1461120, 0.4255965, 1.1517839, -0.4845268, 0.4840227
7: -4.9118948, -3.7942643, -4.9143229, -3.7916636, -0.7881093, 0.7882538
8: -0.8763628, -0.1937261, -0.8795466, -0.1892772, -0.5334609, 0.5326631
9: -5.6802359, -4.6374364, -5.6856060, -4.6348090, -0.6810076, 0.6830261

Time for backsubstitution: 20.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5772
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 486
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5772

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269549, upper bound: 0.3261539
time: 3.60 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290333, upper bound: 0.3261539
time: 3.96 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -3.0641847, -2.2406852, -3.0641847, -2.2406852, -0.6634290, 0.6620121
1: -7.4437399, -6.4144535, -7.4437399, -6.4144492, -0.5975900, 0.5962729
2: -7.1122885, -6.3522439, -7.1122885, -6.3522429, -0.4296887, 0.4296854
3: -12.7483749, -11.7579327, -12.7483730, -11.7579317, -0.5115085, 0.5116763
4: -0.4580395, 0.4665931, -0.4580393, 0.4665953, -0.7241149, 0.7177172
5: -7.6689420, -6.8097658, -7.6689444, -6.8097653, -0.5543365, 0.5525832
6: 0.4248867, 1.1547046, 0.4248850, 1.1547050, -0.4935493, 0.4847085
7: -4.9145422, -3.7904549, -4.9145422, -3.7904549, -0.7930212, 0.7924922
8: -0.8799989, -0.1870546, -0.8799992, -0.1870532, -0.5397823, 0.5341341
9: -5.6880751, -4.6347718, -5.6880784, -4.6347718, -0.6836114, 0.6886091

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 5772

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290335, upper bound: 0.3269547
time: 3.64 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290334, upper bound: 0.3290331
time: 3.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.94 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.94
Output dim: 6, lower bound: -0.3269549, upper bound: 0.3261539
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.94
Output dim: 6, lower bound: -0.3290333, upper bound: 0.3261539
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 29.94
Output dim: 6, lower bound: -0.3290335, upper bound: 0.3269547
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 29.94
Output dim: 6, lower bound: -0.3290334, upper bound: 0.3290331

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -3.0590818, -2.2437296, -3.0587335, -2.2440248, -0.6552556, 0.6530368
1: -7.4394040, -6.4203043, -7.4349194, -6.4273548, -0.5786524, 0.5824339
2: -7.1094141, -6.3550348, -7.1042013, -6.3617415, -0.4180013, 0.4186473
3: -12.7442970, -11.7610483, -12.7372932, -11.7680178, -0.4974313, 0.4973776
4: -0.4511862, 0.4553369, -0.4482143, 0.4535381, -0.7045407, 0.7034295
5: -7.6670098, -6.8128748, -7.6665268, -6.8133631, -0.5483990, 0.5489318
6: 0.4313319, 1.1437082, 0.4339437, 1.1447680, -0.4766064, 0.4738638
7: -4.9112806, -3.7949514, -4.9122181, -3.7961187, -0.7814279, 0.7855904
8: -0.8758898, -0.1953993, -0.8734846, -0.1944103, -0.5278940, 0.5243587
9: -5.6765966, -4.6375313, -5.6741447, -4.6387386, -0.6725194, 0.6710093

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6170

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3250026, upper bound: 0.3261533
time: 3.84 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269542, upper bound: 0.3261533
time: 4.05 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -3.0597920, -2.2427657, -3.0628154, -2.2407901, -0.6587279, 0.6591749
1: -7.4424181, -6.4198575, -7.4437146, -6.4160538, -0.5940342, 0.5829811
2: -7.1121564, -6.3548121, -7.1122813, -6.3529053, -0.4288994, 0.4187413
3: -12.7480383, -11.7607498, -12.7482948, -11.7585335, -0.5106981, 0.4978672
4: -0.4519434, 0.4588579, -0.4572625, 0.4640130, -0.7073827, 0.7156630
5: -7.6673288, -6.8119035, -7.6687889, -6.8104444, -0.5505040, 0.5519547
6: 0.4309227, 1.1461124, 0.4255972, 1.1517823, -0.4775833, 0.4840219
7: -4.9118948, -3.7942634, -4.9143224, -3.7916651, -0.7881274, 0.7882528
8: -0.8763626, -0.1937265, -0.8795476, -0.1892800, -0.5307155, 0.5326622
9: -5.6802368, -4.6374364, -5.6856022, -4.6348090, -0.6805186, 0.6756835

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 858

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 6170

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3242009
time: 3.64 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3261534
time: 3.87 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -3.0600011, -2.2439284, -3.0634615, -2.2416499, -0.6579735, 0.6584966
1: -7.4349422, -6.4264908, -7.4407234, -6.4149117, -0.5882721, 0.5801122
2: -7.1042080, -6.3615994, -7.1095457, -6.3525186, -0.4213002, 0.4182454
3: -12.7373610, -11.7679167, -12.7446327, -11.7582960, -0.5002065, 0.4978817
4: -0.4487015, 0.4561107, -0.4572669, 0.4630710, -0.7115822, 0.7065842
5: -7.6666784, -6.8126965, -7.6686220, -6.8107214, -0.5513220, 0.5490439
6: 0.4337521, 1.1476860, 0.4253438, 1.1523039, -0.4828231, 0.4769357
7: -4.9124336, -3.7951841, -4.9139280, -3.7911739, -0.7903728, 0.7852197
8: -0.8736076, -0.1921945, -0.8795009, -0.1887302, -0.5311084, 0.5285590
9: -5.6765518, -4.6387024, -5.6844368, -4.6348681, -0.6715195, 0.6800919

Time for backsubstitution: 22.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: B, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6170

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3270802, upper bound: 0.3269542
time: 4.19 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3269543
time: 3.67 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -3.0641847, -2.2406869, -3.0641868, -2.2406852, -0.6642921, 0.6619930
1: -7.4437351, -6.4144516, -7.4437399, -6.4144502, -0.5888155, 0.5962725
2: -7.1122847, -6.3522444, -7.1122875, -6.3522434, -0.4214389, 0.4296851
3: -12.7483711, -11.7579317, -12.7483749, -11.7579308, -0.5007613, 0.5116764
4: -0.4580399, 0.4665909, -0.4580406, 0.4665947, -0.7241139, 0.7094500
5: -7.6689444, -6.8097672, -7.6689420, -6.8097658, -0.5543375, 0.5512002
6: 0.4248857, 1.1547019, 0.4248848, 1.1547053, -0.4935493, 0.4777648
7: -4.9145417, -3.7904553, -4.9145422, -3.7904558, -0.7930198, 0.7925107
8: -0.8799982, -0.1870570, -0.8799992, -0.1870532, -0.5397816, 0.5313885
9: -5.6880713, -4.6347713, -5.6880774, -4.6347723, -0.6763003, 0.6881216

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5772
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 858

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6170

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3270802, upper bound: 0.3290332
time: 3.52 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3290326
time: 3.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.59 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3250026, upper bound: 0.3261533
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3269542, upper bound: 0.3261533
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3242009
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3261534
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3270802, upper bound: 0.3269542
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3269543
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3270802, upper bound: 0.3290332
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.59
Output dim: 6, lower bound: -0.3290328, upper bound: 0.3290326

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -3.0580132, -2.2438855, -3.0539124, -2.2460687, -0.6511750, 0.6470430
1: -7.4375911, -6.4205360, -7.4293332, -6.4299183, -0.5739684, 0.5764880
2: -7.1075892, -6.3552628, -7.0982213, -6.3657045, -0.4123552, 0.4122825
3: -12.7426758, -11.7624607, -12.7319193, -11.7749567, -0.4881172, 0.4903971
4: -0.4465346, 0.4550278, -0.4345109, 0.4479226, -0.6919250, 0.6886399
5: -7.6639996, -6.8133593, -7.6574259, -6.8190403, -0.5396383, 0.5393808
6: 0.4318967, 1.1420059, 0.4403350, 1.1395445, -0.4707792, 0.4649279
7: -4.9060807, -3.7953696, -4.8966694, -3.8045602, -0.7673330, 0.7691746
8: -0.8755517, -0.1982851, -0.8679619, -0.2033801, -0.5186050, 0.5159547
9: -5.6757140, -4.6390247, -5.6607475, -4.6433020, -0.6672122, 0.6561491

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 486

## Relational analysis of NS_A1_B1_B1_B1

### Relational analysis result of NS_A1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3220734, upper bound: 0.3257059
time: 3.77 seconds

## Relational analysis of NS_A1_B1_B1_B2

### Relational analysis result of NS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3250010, upper bound: 0.3261516
time: 3.63 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -3.0590818, -2.2437282, -3.0587306, -2.2440245, -0.6581774, 0.6518941
1: -7.4394035, -6.4203043, -7.4349189, -6.4273572, -0.5786519, 0.5802844
2: -7.1094141, -6.3550339, -7.1041985, -6.3617415, -0.4180009, 0.4171845
3: -12.7442989, -11.7610483, -12.7372904, -11.7680197, -0.4978504, 0.4973668
4: -0.4511864, 0.4553374, -0.4482105, 0.4535377, -0.7027411, 0.7049587
5: -7.6670103, -6.8128757, -7.6665235, -6.8133650, -0.5483983, 0.5406971
6: 0.4313319, 1.1437078, 0.4339440, 1.1447659, -0.4742042, 0.4738629
7: -4.9112797, -3.7949505, -4.9122143, -3.7961168, -0.7814274, 0.7745919
8: -0.8758914, -0.1953979, -0.8734841, -0.1944127, -0.5240877, 0.5243583
9: -5.6765957, -4.6375299, -5.6741433, -4.6387405, -0.6702113, 0.6710088

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 486

## Relational analysis of NS_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3247618, upper bound: 0.3259450
time: 3.61 seconds

## Relational analysis of NS_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269521, upper bound: 0.3261515
time: 3.58 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.0549819, -2.2448058, -3.0617518, -2.2409468, -0.6527450, 0.6550748
1: -7.4368420, -6.4224238, -7.4419012, -6.4162865, -0.5881023, 0.5782948
2: -7.1061797, -6.3587751, -7.1104546, -6.3531351, -0.4225372, 0.4130949
3: -12.7426682, -11.7676783, -12.7466745, -11.7599459, -0.5037169, 0.4885775
4: -0.4382710, 0.4532441, -0.4526103, 0.4637041, -0.6926222, 0.7030447
5: -7.6582270, -6.8175907, -7.6657777, -6.8109264, -0.5409584, 0.5431926
6: 0.4373250, 1.1408873, 0.4261651, 1.1500804, -0.4686360, 0.4781786
7: -4.8963604, -3.8027086, -4.9091172, -3.7920828, -0.7717195, 0.7741585
8: -0.8708367, -0.2026973, -0.8792067, -0.1921663, -0.5223150, 0.5233495
9: -5.6668348, -4.6419997, -5.6847258, -4.6363044, -0.6656437, 0.6703830

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 486
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 858

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 486

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3285845, upper bound: 0.3212705
time: 3.74 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290310, upper bound: 0.3241995
time: 3.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.0597911, -2.2427640, -3.0628154, -2.2407901, -0.6575780, 0.6620963
1: -7.4424171, -6.4198561, -7.4437141, -6.4160538, -0.5918868, 0.5829811
2: -7.1121550, -6.3548121, -7.1122799, -6.3529038, -0.4274364, 0.4187409
3: -12.7480364, -11.7607498, -12.7482948, -11.7585335, -0.5106874, 0.4982861
4: -0.4519392, 0.4588572, -0.4572631, 0.4640135, -0.7089186, 0.7138638
5: -7.6673269, -6.8119035, -7.6687884, -6.8104434, -0.5422668, 0.5519543
6: 0.4309247, 1.1461091, 0.4255981, 1.1517823, -0.4775827, 0.4816197
7: -4.9118910, -3.7942648, -4.9143219, -3.7916646, -0.7771297, 0.7882528
8: -0.8763614, -0.1937304, -0.8795466, -0.1892796, -0.5307143, 0.5288575
9: -5.6802354, -4.6374378, -5.6856022, -4.6348095, -0.6805177, 0.6733775

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 486
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 858

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 486

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3288243, upper bound: 0.3239601
time: 3.75 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290310, upper bound: 0.3261512
time: 3.77 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -3.0589333, -2.2440848, -3.0586524, -2.2436929, -0.6538942, 0.6524897
1: -7.4331293, -6.4267216, -7.4351287, -6.4174805, -0.5835857, 0.5741599
2: -7.1023827, -6.3618279, -7.1035652, -6.3564830, -0.4156528, 0.4118798
3: -12.7357397, -11.7693310, -12.7392616, -11.7652340, -0.4909055, 0.4908906
4: -0.4440441, 0.4558038, -0.4435726, 0.4574544, -0.6989567, 0.6917994
5: -7.6636672, -6.8131742, -7.6595206, -6.8164091, -0.5425618, 0.5394914
6: 0.4343140, 1.1459842, 0.4317496, 1.1470802, -0.4769897, 0.4679879
7: -4.9072313, -3.7956038, -4.8983727, -3.7996206, -0.7762752, 0.7688167
8: -0.8732679, -0.1950788, -0.8739743, -0.1976976, -0.5218225, 0.5201561
9: -5.6756792, -4.6401978, -5.6710420, -4.6394296, -0.6662107, 0.6652222

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 486
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5772
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3242010, upper bound: 0.3269535
time: 4.09 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3242009, upper bound: 0.3269550
time: 3.58 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -3.0600011, -2.2439284, -3.0634599, -2.2416515, -0.6608984, 0.6573472
1: -7.4349418, -6.4264898, -7.4407215, -6.4149113, -0.5882711, 0.5779598
2: -7.1042070, -6.3615980, -7.1095419, -6.3525186, -0.4212995, 0.4167825
3: -12.7373648, -11.7679176, -12.7446289, -11.7582979, -0.5006251, 0.4978713
4: -0.4487004, 0.4561117, -0.4572628, 0.4630709, -0.7097840, 0.7081215
5: -7.6666784, -6.8126965, -7.6686196, -6.8107224, -0.5513208, 0.5408123
6: 0.4337521, 1.1476870, 0.4253445, 1.1523008, -0.4804211, 0.4769351
7: -4.9124327, -3.7951841, -4.9139256, -3.7911758, -0.7903728, 0.7742229
8: -0.8736079, -0.1921949, -0.8794999, -0.1887336, -0.5273035, 0.5285580
9: -5.6765518, -4.6387038, -5.6844330, -4.6348696, -0.6692142, 0.6800904

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 486
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 486

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3268387, upper bound: 0.3267475
time: 3.60 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290308, upper bound: 0.3269527
time: 3.88 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -3.0631244, -2.2408438, -3.0593791, -2.2427256, -0.6601944, 0.6559777
1: -7.4419203, -6.4146843, -7.4381437, -6.4170218, -0.5841269, 0.5903165
2: -7.1104598, -6.3524756, -7.1063070, -6.3562078, -0.4157915, 0.4233178
3: -12.7467480, -11.7593451, -12.7430048, -11.7648687, -0.4914670, 0.5046961
4: -0.4533856, 0.4662820, -0.4443476, 0.4609792, -0.7114968, 0.6946676
5: -7.6659317, -6.8102493, -7.6598411, -6.8154516, -0.5455699, 0.5416515
6: 0.4254553, 1.1529999, 0.4312963, 1.1494803, -0.4876978, 0.4688087
7: -4.9093351, -3.7908740, -4.8989840, -3.7988987, -0.7789202, 0.7761049
8: -0.8796577, -0.1899409, -0.8744702, -0.1960206, -0.5304902, 0.5229878
9: -5.6871967, -4.6362653, -5.6746845, -4.6393347, -0.6709907, 0.6732514

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5772
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 858

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3242009, upper bound: 0.3290320
time: 4.14 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3242009, upper bound: 0.3290334
time: 3.51 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -3.0641854, -2.2406857, -3.0641849, -2.2406855, -0.6672111, 0.6608424
1: -7.4437370, -6.4144521, -7.4437361, -6.4144511, -0.5888147, 0.5941174
2: -7.1122866, -6.3522439, -7.1122866, -6.3522439, -0.4214387, 0.4282223
3: -12.7483711, -11.7579317, -12.7483711, -11.7579298, -0.5011799, 0.5116650
4: -0.4580374, 0.4665899, -0.4580352, 0.4665935, -0.7223158, 0.7109864
5: -7.6689434, -6.8097663, -7.6689410, -6.8097653, -0.5543358, 0.5429688
6: 0.4248869, 1.1547022, 0.4248869, 1.1547041, -0.4911468, 0.4777644
7: -4.9145412, -3.7904549, -4.9145384, -3.7904549, -0.7930193, 0.7815118
8: -0.8799982, -0.1870575, -0.8799992, -0.1870575, -0.5359766, 0.5313880
9: -5.6880708, -4.6347723, -5.6880760, -4.6347733, -0.6739953, 0.6881204

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 486
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 858

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 486

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3268387, upper bound: 0.3288241
time: 3.76 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290308, upper bound: 0.3290310
time: 3.79 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.85 seconds
NS_A1_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3220734, upper bound: 0.3257059
NS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3250010, upper bound: 0.3261516
NS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3247618, upper bound: 0.3259450
NS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3269521, upper bound: 0.3261515
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3285845, upper bound: 0.3212705
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3290310, upper bound: 0.3241995
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3288243, upper bound: 0.3239601
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3290310, upper bound: 0.3261512
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3242010, upper bound: 0.3269535
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3242009, upper bound: 0.3269550
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3268387, upper bound: 0.3267475
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3290308, upper bound: 0.3269527
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3242009, upper bound: 0.3290320
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3242009, upper bound: 0.3290334
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3268387, upper bound: 0.3288241
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 29.85
Output dim: 6, lower bound: -0.3290308, upper bound: 0.3290310

## BFS NS instance: NS_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -3.0580132, -2.2438855, -3.0539126, -2.2460678, -0.6523032, 0.6470416
1: -7.4375911, -6.4205360, -7.4293289, -6.4299197, -0.5739677, 0.5792329
2: -7.1075892, -6.3552628, -7.0982199, -6.3657045, -0.4123551, 0.4097611
3: -12.7426758, -11.7624607, -12.7319202, -11.7749586, -0.4878430, 0.4903970
4: -0.4465346, 0.4550278, -0.4345101, 0.4479239, -0.6916950, 0.6895945
5: -7.6639996, -6.8133593, -7.6574249, -6.8190393, -0.5388041, 0.5440750
6: 0.4318967, 1.1420059, 0.4403355, 1.1395440, -0.4707779, 0.4663100
7: -4.9060807, -3.7953696, -4.8966680, -3.8045616, -0.7673326, 0.7680936
8: -0.8755517, -0.1982851, -0.8679614, -0.2033811, -0.5190539, 0.5158851
9: -5.6757140, -4.6390247, -5.6607475, -4.6433029, -0.6640372, 0.6561489

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 486

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A1_B1_B1_B2_B1

### Relational analysis result of NS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3221430, upper bound: 0.3261524
time: 3.44 seconds

## Relational analysis of NS_A1_B1_B1_B2_B2

### Relational analysis result of NS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3221428, upper bound: 0.3261522
time: 3.53 seconds

## BFS NS instance: NS_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -3.0578313, -2.2439690, -3.0536118, -2.2461905, -0.6534407, 0.6465242
1: -7.4380102, -6.4211349, -7.4294214, -6.4305444, -0.5746284, 0.5633197
2: -7.1076689, -6.3553371, -7.0976515, -6.3645029, -0.4117587, 0.4100220
3: -12.7438908, -11.7627325, -12.7346535, -11.7738123, -0.4913096, 0.4780840
4: -0.4477997, 0.4550433, -0.4358819, 0.4504210, -0.6945424, 0.6949134
5: -7.6663313, -6.8134894, -7.6646166, -6.8165798, -0.5437772, 0.5221529
6: 0.4321704, 1.1421008, 0.4391437, 1.1388249, -0.4676791, 0.4595258
7: -4.9095387, -3.7953434, -4.9057388, -3.7992191, -0.7761912, 0.7676105
8: -0.8755326, -0.1963959, -0.8715458, -0.1986179, -0.5280805, 0.5212231
9: -5.6759667, -4.6392989, -5.6692724, -4.6449747, -0.6658261, 0.6641624

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A1_B1_B2_B1_B1

### Relational analysis result of NS_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3219054, upper bound: 0.3259450
time: 3.63 seconds

## Relational analysis of NS_A1_B1_B2_B1_B2

### Relational analysis result of NS_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3219054, upper bound: 0.3259451
time: 3.78 seconds

## BFS NS instance: NS_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -3.0590818, -2.2437282, -3.0587311, -2.2440245, -0.6593063, 0.6518924
1: -7.4394035, -6.4203043, -7.4349170, -6.4273577, -0.5786514, 0.5830295
2: -7.1094141, -6.3550339, -7.1041975, -6.3617430, -0.4180008, 0.4146632
3: -12.7442989, -11.7610483, -12.7372885, -11.7680206, -0.4975762, 0.4973667
4: -0.4511864, 0.4553374, -0.4482071, 0.4535370, -0.7025118, 0.7029991
5: -7.6670103, -6.8128757, -7.6665235, -6.8133659, -0.5475309, 0.5454624
6: 0.4313319, 1.1437078, 0.4339457, 1.1447656, -0.4742031, 0.4752446
7: -4.9112797, -3.7949505, -4.9122128, -3.7961168, -0.7814279, 0.7728405
8: -0.8758914, -0.1953979, -0.8734832, -0.1944141, -0.5245593, 0.5242913
9: -5.6765957, -4.6375299, -5.6741428, -4.6387424, -0.6665597, 0.6710086

Time for backsubstitution: 21.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 5758
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 858
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 486

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4654

## Relational analysis of NS_A1_B1_B2_B2_B1

### Relational analysis result of NS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3239709, upper bound: 0.3261487
time: 3.69 seconds

## Relational analysis of NS_A1_B1_B2_B2_B2

### Relational analysis result of NS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269484, upper bound: 0.3261480
time: 3.69 seconds

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -3.0514500, -2.2468326, -3.0606112, -2.2413135, -0.6486168, 0.6469414
1: -7.4333563, -6.4256201, -7.4402485, -6.4171219, -0.5731277, 0.5738716
2: -7.1001072, -6.3613200, -7.1087222, -6.3536367, -0.4157108, 0.4051490
3: -12.7402039, -11.7703419, -12.7462769, -11.7628527, -0.4846342, 0.4885521
4: -0.4338958, 0.4501235, -0.4464027, 0.4634011, -0.6873937, 0.6946247
5: -7.6564741, -6.8200808, -7.6651001, -6.8116031, -0.5224159, 0.5391691
6: 0.4423175, 1.1358418, 0.4271798, 1.1484797, -0.4543127, 0.4722944
7: -4.8920126, -3.8058097, -4.9067497, -3.7924867, -0.7663488, 0.7683206
8: -0.8688965, -0.2057347, -0.8788366, -0.1934147, -0.5189354, 0.5290332
9: -5.6637630, -4.6472507, -5.6840830, -4.6385098, -0.6593537, 0.6663318

Time for backsubstitution: 21.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5772
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: A, layer: 1, pos: 875
type: B, layer: 1, pos: 875
type: A, layer: 1, pos: 4582
type: B, layer: 1, pos: 4582
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 844
type: B, layer: 1, pos: 844
type: A, layer: 1, pos: 4624
type: B, layer: 1, pos: 4624
type: A, layer: 1, pos: 81
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 858

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3257060, upper bound: 0.3212705
time: 3.82 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3257063, upper bound: 0.3212713
time: 3.70 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -3.0549803, -2.2448034, -3.0617518, -2.2409468, -0.6527450, 0.6562023
1: -7.4368424, -6.4224238, -7.4419012, -6.4162865, -0.5908484, 0.5782940
2: -7.1061783, -6.3587766, -7.1104546, -6.3531351, -0.4200162, 0.4130948
3: -12.7426682, -11.7676783, -12.7466745, -11.7599459, -0.5037169, 0.4883033
4: -0.4382700, 0.4532418, -0.4526103, 0.4637041, -0.6935916, 0.7028148
5: -7.6582260, -6.8175907, -7.6657777, -6.8109264, -0.5456524, 0.5423594
6: 0.4373260, 1.1408870, 0.4261651, 1.1500804, -0.4700184, 0.4781779
7: -4.8963585, -3.8027086, -4.9091172, -3.7920828, -0.7706404, 0.7741580
8: -0.8708363, -0.2026968, -0.8792067, -0.1921663, -0.5222456, 0.5237947
9: -5.6668363, -4.6419992, -5.6847258, -4.6363044, -0.6656437, 0.6672096

Time for backsubstitution: 21.19 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.45 + 563.84 = 620.29 seconds
