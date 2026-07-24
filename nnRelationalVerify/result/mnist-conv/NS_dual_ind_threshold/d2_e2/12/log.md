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
execution time: IAR + RelationalAnalysis = 21.53 + 33.51 = 55.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3290372, upper bound: 0.3290372

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5758
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5758

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290352, upper bound: 0.3261551
time: 3.68 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290352, upper bound: 0.3290350
time: 3.64 seconds

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

Time for backsubstitution: 20.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5772

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269549, upper bound: 0.3261539
time: 3.59 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290333, upper bound: 0.3261539
time: 3.55 seconds

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

Time for backsubstitution: 19.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261551, upper bound: 0.3290351
time: 3.55 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261551, upper bound: 0.3290358
time: 3.59 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 27.29 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 6, lower bound: -0.3269549, upper bound: 0.3261539
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 6, lower bound: -0.3290333, upper bound: 0.3261539
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 6, lower bound: -0.3261551, upper bound: 0.3290351
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 27.29
Output dim: 6, lower bound: -0.3261551, upper bound: 0.3290358

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

Time for backsubstitution: 20.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6170

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269544, upper bound: 0.3242008
time: 3.52 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3269544, upper bound: 0.3261532
time: 3.48 seconds

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

Time for backsubstitution: 21.07 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5772

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290335, upper bound: 0.3240958
time: 3.78 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3290335, upper bound: 0.3261540
time: 3.95 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -3.0641847, -2.2406852, -3.0597932, -2.2427649, -0.6599998, 0.6580329
1: -7.4437399, -6.4144535, -7.4424210, -6.4198570, -0.5917761, 0.5962381
2: -7.1122885, -6.3522439, -7.1121569, -6.3548131, -0.4269996, 0.4294763
3: -12.7483749, -11.7579327, -12.7480383, -11.7607479, -0.5087483, 0.5113338
4: -0.4580395, 0.4665931, -0.4519433, 0.4588590, -0.7163754, 0.7182190
5: -7.6689420, -6.8097658, -7.6673298, -6.8119040, -0.5520983, 0.5526669
6: 0.4248867, 1.1547046, 0.4309227, 1.1461120, -0.4849447, 0.4874856
7: -4.9145422, -3.7904549, -4.9118948, -3.7942643, -0.7882118, 0.7902870
8: -0.8799989, -0.1870546, -0.8763628, -0.1937261, -0.5331831, 0.5357285
9: -5.6880751, -4.6347718, -5.6802359, -4.6374364, -0.6859152, 0.6808205

Time for backsubstitution: 21.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 5772

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3269541
time: 4.15 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3290327
time: 3.71 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -3.0641847, -2.2406852, -3.0641847, -2.2406852, -0.6634288, 0.6634288
1: -7.4437399, -6.4144535, -7.4437399, -6.4144535, -0.5962725, 0.5962727
2: -7.1122885, -6.3522439, -7.1122885, -6.3522439, -0.4296851, 0.4296852
3: -12.7483749, -11.7579327, -12.7483749, -11.7579327, -0.5115077, 0.5115079
4: -0.4580395, 0.4665931, -0.4580395, 0.4665931, -0.7177167, 0.7177169
5: -7.6689420, -6.8097658, -7.6689420, -6.8097658, -0.5525832, 0.5525830
6: 0.4248867, 1.1547046, 0.4248867, 1.1547046, -0.4847076, 0.4847076
7: -4.9145422, -3.7904549, -4.9145422, -3.7904549, -0.7924919, 0.7924919
8: -0.8799989, -0.1870546, -0.8799989, -0.1870546, -0.5341334, 0.5341334
9: -5.6880751, -4.6347718, -5.6880751, -4.6347718, -0.6836112, 0.6836112

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5772

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3269555
time: 3.62 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3290339
time: 3.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.56 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3269544, upper bound: 0.3242008
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3269544, upper bound: 0.3261532
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3290335, upper bound: 0.3240958
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3290335, upper bound: 0.3261540
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3269541
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3290327
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3269555
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.56
Output dim: 6, lower bound: -0.3261539, upper bound: 0.3290339

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -3.0543058, -2.2457674, -3.0576613, -2.2441809, -0.6493056, 0.6489563
1: -7.4338589, -6.4228716, -7.4331064, -6.4275875, -0.5727625, 0.5777485
2: -7.1034474, -6.3589973, -7.1023765, -6.3619733, -0.4116488, 0.4130009
3: -12.7389259, -11.7679615, -12.7356701, -11.7694340, -0.4904439, 0.4881188
4: -0.4375502, 0.4497219, -0.4435571, 0.4532295, -0.6898212, 0.6908011
5: -7.6579084, -6.8185401, -7.6635160, -6.8138418, -0.5388551, 0.5401893
6: 0.4377282, 1.1385064, 0.4345038, 1.1430669, -0.4676673, 0.4680581
7: -4.8957663, -3.8033943, -4.9070168, -3.7965364, -0.7650571, 0.7714984
8: -0.8703671, -0.2043509, -0.8731444, -0.1972957, -0.5195017, 0.5150678
9: -5.6632409, -4.6420937, -5.6732697, -4.6402326, -0.6576760, 0.6657093

Time for backsubstitution: 21.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240959, upper bound: 0.3242007
time: 3.79 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240959, upper bound: 0.3242016
time: 3.52 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -3.0590789, -2.2437277, -3.0587332, -2.2440238, -0.6541078, 0.6559632
1: -7.4394016, -6.4203043, -7.4349189, -6.4273577, -0.5765071, 0.5824337
2: -7.1094117, -6.3550339, -7.1042008, -6.3617411, -0.4165386, 0.4186472
3: -12.7442942, -11.7610483, -12.7372894, -11.7680187, -0.4974203, 0.4977965
4: -0.4511825, 0.4553356, -0.4482141, 0.4535379, -0.7060769, 0.7016296
5: -7.6670074, -6.8128753, -7.6665268, -6.8133655, -0.5401626, 0.5489309
6: 0.4313331, 1.1437068, 0.4339433, 1.1447668, -0.4766060, 0.4714615
7: -4.9112754, -3.7949514, -4.9122181, -3.7961168, -0.7704291, 0.7855906
8: -0.8758891, -0.1954012, -0.8734846, -0.1944098, -0.5278935, 0.5205534
9: -5.6765947, -4.6375322, -5.6741438, -4.6387386, -0.6725185, 0.6687024

Time for backsubstitution: 22.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
time: 3.53 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -3.0558825, -2.2459977, -3.0628154, -2.2407901, -0.6549866, 0.6548805
1: -7.4336238, -6.4298224, -7.4437146, -6.4160538, -0.5851979, 0.5808351
2: -7.1040754, -6.3626781, -7.1122813, -6.3529053, -0.4208137, 0.4199730
3: -12.7370310, -11.7693176, -12.7482948, -11.7585335, -0.4996593, 0.5000907
4: -0.4434247, 0.4483824, -0.4572625, 0.4640130, -0.7074618, 0.7051973
5: -7.6650820, -6.8148079, -7.6687889, -6.8104444, -0.5498121, 0.5487826
6: 0.4383047, 1.1390941, 0.4255972, 1.1517823, -0.4775112, 0.4769897
7: -4.9097953, -3.7982588, -4.9143224, -3.7916651, -0.7859187, 0.7833047
8: -0.8709102, -0.1988478, -0.8795476, -0.1892800, -0.5275960, 0.5275965
9: -5.6688886, -4.6413603, -5.6856022, -4.6348090, -0.6688359, 0.6786330

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3240963
time: 4.59 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3240963
time: 4.30 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -3.0597906, -2.2427657, -3.0628154, -2.2407901, -0.6596091, 0.6591737
1: -7.4424157, -6.4198561, -7.4437146, -6.4160538, -0.5852597, 0.5829809
2: -7.1121545, -6.3548107, -7.1122813, -6.3529053, -0.4206494, 0.4187411
3: -12.7480345, -11.7607498, -12.7482948, -11.7585335, -0.4999511, 0.4978670
4: -0.4519434, 0.4588556, -0.4572625, 0.4640130, -0.7073822, 0.7073996
5: -7.6673298, -6.8119059, -7.6687889, -6.8104444, -0.5505033, 0.5505726
6: 0.4309235, 1.1461093, 0.4255972, 1.1517823, -0.4775826, 0.4770815
7: -4.9118938, -3.7942629, -4.9143224, -3.7916651, -0.7881269, 0.7882724
8: -0.8763609, -0.1937284, -0.8795476, -0.1892800, -0.5307150, 0.5299180
9: -5.6802335, -4.6374369, -5.6856022, -4.6348090, -0.6735909, 0.6756835

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5758
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 5758

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3261545
time: 3.49 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3261545
time: 3.76 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -3.0600011, -2.2439284, -3.0590818, -2.2437296, -0.6545167, 0.6545339
1: -7.4349422, -6.4264908, -7.4394040, -6.4203043, -0.5824530, 0.5799658
2: -7.1042080, -6.3615994, -7.1094141, -6.3550348, -0.4186549, 0.4180366
3: -12.7373610, -11.7679167, -12.7442970, -11.7610483, -0.4975078, 0.4975398
4: -0.4487015, 0.4561107, -0.4511862, 0.4553369, -0.7038460, 0.7071071
5: -7.6666784, -6.8126965, -7.6670098, -6.8128748, -0.5490658, 0.5491366
6: 0.4337521, 1.1476860, 0.4313319, 1.1437082, -0.4742181, 0.4795642
7: -4.9124336, -3.7951841, -4.9112806, -3.7949514, -0.7855434, 0.7829719
8: -0.8736076, -0.1921945, -0.8758898, -0.1953993, -0.5244832, 0.5301404
9: -5.6765518, -4.6387024, -5.6765966, -4.6375313, -0.6737785, 0.6723230

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 6170

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3242010, upper bound: 0.3269536
time: 4.54 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261533, upper bound: 0.3269538
time: 4.07 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -3.0641847, -2.2406869, -3.0597920, -2.2427657, -0.6608634, 0.6580145
1: -7.4437351, -6.4144516, -7.4424181, -6.4198575, -0.5830011, 0.5962374
2: -7.1122847, -6.3522444, -7.1121564, -6.3548121, -0.4187492, 0.4294761
3: -12.7483711, -11.7579317, -12.7480383, -11.7607498, -0.4980016, 0.5113337
4: -0.4580399, 0.4665909, -0.4519434, 0.4588579, -0.7163749, 0.7099550
5: -7.6689444, -6.8097672, -7.6673288, -6.8119035, -0.5520983, 0.5512843
6: 0.4248857, 1.1547019, 0.4309227, 1.1461124, -0.4849447, 0.4805428
7: -4.9145417, -3.7904553, -4.9118948, -3.7942634, -0.7882104, 0.7903051
8: -0.8799982, -0.1870570, -0.8763626, -0.1937265, -0.5331821, 0.5329840
9: -5.6880713, -4.6347713, -5.6802368, -4.6374364, -0.6786075, 0.6803331

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5772

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240957, upper bound: 0.3290334
time: 3.55 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240957, upper bound: 0.3290335
time: 3.58 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -3.0600011, -2.2439284, -3.0634599, -2.2416508, -0.6579728, 0.6599045
1: -7.4349422, -6.4264908, -7.4407225, -6.4149137, -0.5868864, 0.5801122
2: -7.1042080, -6.3615994, -7.1095457, -6.3525190, -0.4212967, 0.4182453
3: -12.7373610, -11.7679167, -12.7446318, -11.7582960, -0.5002053, 0.4977138
4: -0.4487015, 0.4561107, -0.4572675, 0.4630702, -0.7051811, 0.7065835
5: -7.6666784, -6.8126965, -7.6686220, -6.8107233, -0.5495534, 0.5490441
6: 0.4337521, 1.1476860, 0.4253440, 1.1523015, -0.4739794, 0.4769350
7: -4.9124336, -3.7951841, -4.9139252, -3.7911758, -0.7897902, 0.7852192
8: -0.8736076, -0.1921945, -0.8795006, -0.1887307, -0.5253358, 0.5285578
9: -5.6765518, -4.6387024, -5.6844325, -4.6348677, -0.6715195, 0.6751120

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6170

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3242127, upper bound: 0.3269541
time: 4.29 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3261650, upper bound: 0.3269541
time: 3.93 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -3.0641847, -2.2406869, -3.0641859, -2.2406862, -0.6642919, 0.6634099
1: -7.4437351, -6.4144516, -7.4437394, -6.4144540, -0.5874979, 0.5962722
2: -7.1122847, -6.3522444, -7.1122885, -6.3522444, -0.4214354, 0.4296854
3: -12.7483711, -11.7579317, -12.7483730, -11.7579327, -0.5007608, 0.5115076
4: -0.4580399, 0.4665909, -0.4580389, 0.4665928, -0.7177157, 0.7094498
5: -7.6689444, -6.8097672, -7.6689434, -6.8097663, -0.5525825, 0.5512002
6: 0.4248857, 1.1547019, 0.4248865, 1.1547043, -0.4847068, 0.4777644
7: -4.9145417, -3.7904553, -4.9145412, -3.7904534, -0.7924919, 0.7925096
8: -0.8799982, -0.1870570, -0.8799982, -0.1870542, -0.5341330, 0.5313878
9: -5.6880713, -4.6347713, -5.6880774, -4.6347733, -0.6763003, 0.6831231

Time for backsubstitution: 22.02 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5772
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 4654
type: B, layer: 1, pos: 486
type: B, layer: 1, pos: 4582
type: B, layer: 1, pos: 875
type: B, layer: 1, pos: 844
type: B, layer: 1, pos: 858
type: B, layer: 1, pos: 4624
type: B, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 5772

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3241020, upper bound: 0.3290332
time: 3.58 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3241020, upper bound: 0.3290331
time: 3.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.56 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240959, upper bound: 0.3242007
NS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240959, upper bound: 0.3242016
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3240963
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3240963
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3261545
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240962, upper bound: 0.3261545
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3242010, upper bound: 0.3269536
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3261533, upper bound: 0.3269538
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240957, upper bound: 0.3290334
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3240957, upper bound: 0.3290335
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3242127, upper bound: 0.3269541
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3261650, upper bound: 0.3269541
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3241020, upper bound: 0.3290332
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.56
Output dim: 6, lower bound: -0.3241020, upper bound: 0.3290331

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -3.0590789, -2.2437277, -3.0558832, -2.2459958, -0.6513851, 0.6540065
1: -7.4394016, -6.4203043, -7.4336247, -6.4298215, -0.5743341, 0.5811045
2: -7.1094117, -6.3550339, -7.1040764, -6.3626776, -0.4155123, 0.4185277
3: -12.7442942, -11.7610483, -12.7370300, -11.7693167, -0.4961224, 0.4975933
4: -0.4511825, 0.4553356, -0.4434242, 0.4483820, -0.7009149, 0.6969872
5: -7.6670074, -6.8128753, -7.6650801, -6.8148088, -0.5387793, 0.5474153
6: 0.4313331, 1.1437068, 0.4383030, 1.1390936, -0.4709644, 0.4668664
7: -4.9112754, -3.7949514, -4.9097929, -3.7982607, -0.7688489, 0.7828178
8: -0.8758891, -0.1954012, -0.8709102, -0.1988473, -0.5236254, 0.5178564
9: -5.6765947, -4.6375322, -5.6688886, -4.6413603, -0.6696403, 0.6640553

Time for backsubstitution: 21.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 5772

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3240957
time: 3.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
time: 3.35 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -3.0590789, -2.2437277, -3.0600011, -2.2439284, -0.6533859, 0.6574411
1: -7.4394016, -6.4203043, -7.4349418, -6.4264898, -0.5778208, 0.5824528
2: -7.1094117, -6.3550339, -7.1042070, -6.3615980, -0.4165739, 0.4186548
3: -12.7442942, -11.7610483, -12.7373648, -11.7679176, -0.4975290, 0.4979270
4: -0.4511825, 0.4553356, -0.4487004, 0.4561117, -0.7086434, 0.7020462
5: -7.6670074, -6.8128753, -7.6666784, -6.8126965, -0.5409002, 0.5490649
6: 0.4313331, 1.1437068, 0.4337521, 1.1476870, -0.4795637, 0.4718162
7: -4.9112754, -3.7949514, -4.9124327, -3.7951841, -0.7719736, 0.7855442
8: -0.8758891, -0.1954012, -0.8736079, -0.1921949, -0.5301402, 0.5206780
9: -5.6765947, -4.6375322, -5.6765518, -4.6387038, -0.6723216, 0.6714718

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5772
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 5772

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3240957
time: 3.40 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
time: 3.54 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -3.0597906, -2.2427657, -3.0597906, -2.2427657, -0.6568851, 0.6568851
1: -7.4424157, -6.4198561, -7.4424157, -6.4198561, -0.5816512, 0.5816512
2: -7.1121545, -6.3548107, -7.1121545, -6.3548107, -0.4186215, 0.4186213
3: -12.7480345, -11.7607498, -12.7480345, -11.7607498, -0.4976587, 0.4976587
4: -0.4519434, 0.4588556, -0.4519434, 0.4588556, -0.7022152, 0.7022152
5: -7.6673298, -6.8119059, -7.6673298, -6.8119059, -0.5490456, 0.5490456
6: 0.4309235, 1.1461093, 0.4309235, 1.1461093, -0.4719368, 0.4719369
7: -4.9118938, -3.7942629, -4.9118938, -3.7942629, -0.7854981, 0.7854981
8: -0.8763609, -0.1937284, -0.8763609, -0.1937284, -0.5263848, 0.5263846
9: -5.6802335, -4.6374369, -5.6802335, -4.6374369, -0.6707087, 0.6707087

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6170

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240959, upper bound: 0.3242005
time: 4.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
time: 3.48 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -3.0597906, -2.2427657, -3.0641847, -2.2406869, -0.6588957, 0.6608622
1: -7.4424157, -6.4198561, -7.4437351, -6.4144516, -0.5874631, 0.5830009
2: -7.1121545, -6.3548107, -7.1122847, -6.3522444, -0.4212265, 0.4187491
3: -12.7480345, -11.7607498, -12.7483711, -11.7579317, -0.5005867, 0.4980012
4: -0.4519434, 0.4588556, -0.4580399, 0.4665909, -0.7099547, 0.7081110
5: -7.6673298, -6.8119059, -7.6689444, -6.8097672, -0.5512836, 0.5507154
6: 0.4309235, 1.1461093, 0.4248857, 1.1547019, -0.4805424, 0.4780040
7: -4.9118938, -3.7942629, -4.9145417, -3.7904553, -0.7903047, 0.7882295
8: -0.8763609, -0.1937284, -0.8799982, -0.1870570, -0.5329833, 0.5304375
9: -5.6802335, -4.6374369, -5.6880713, -4.6347713, -0.6734004, 0.6786075

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 486
type: A, layer: 1, pos: 4582
type: A, layer: 1, pos: 875
type: A, layer: 1, pos: 844
type: A, layer: 1, pos: 858
type: A, layer: 1, pos: 4624
type: A, layer: 1, pos: 81

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 6170

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3242015
time: 3.45 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3240956, upper bound: 0.3261539
time: 3.56 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -3.0589333, -2.2440848, -3.0543058, -2.2457674, -0.6504402, 0.6485848
1: -7.4331293, -6.4267216, -7.4338589, -6.4228716, -0.5777674, 0.5740757
2: -7.1023827, -6.3618279, -7.1034474, -6.3589973, -0.4130086, 0.4116843
3: -12.7357397, -11.7693310, -12.7389259, -11.7679615, -0.4882488, 0.4905472
4: -0.4440441, 0.4558038, -0.4375502, 0.4497219, -0.6912191, 0.6923888
5: -7.6636672, -6.8131742, -7.6579084, -6.8185401, -0.5403209, 0.5395925
6: 0.4343140, 1.1459842, 0.4377282, 1.1385064, -0.4684081, 0.4706246
7: -4.9072313, -3.7956038, -4.8957663, -3.8033943, -0.7714496, 0.7666011
8: -0.8732679, -0.1950788, -0.8703671, -0.2043509, -0.5151916, 0.5217447
9: -5.6756792, -4.6401978, -5.6632409, -4.6420937, -0.6684773, 0.6574790

Time for backsubstitution: 22.09 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.04 + 556.48 = 611.52 seconds
