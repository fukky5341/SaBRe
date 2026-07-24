## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.1665576092


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.0259066, -5.5622473, -9.0259066, -5.5622473, -2.7390966, 2.7390966)
1: (-6.5765409, -3.9590964, -6.5765409, -3.9590964, -2.2134962, 2.2134960)
2: (8.3243103, 10.9320621, 8.3243103, 10.9320621, -2.2642365, 2.2642365)
3: (-6.1232839, -2.8826058, -6.1232839, -2.8826058, -2.9247456, 2.9247451)
4: (-11.8333864, -7.9824405, -11.8333864, -7.9824405, -2.9852571, 2.9852571)
5: (-13.6636562, -10.1825542, -13.6636562, -10.1825542, -2.5011797, 2.5011792)
6: (-15.6556625, -12.3171911, -15.6556625, -12.3171911, -2.3238053, 2.3238053)
7: (-5.5686188, -2.0476646, -5.5686188, -2.0476646, -3.2520328, 3.2520332)
8: (-1.9611969, 0.3840871, -1.9611969, 0.3840871, -2.0681491, 2.0681491)
9: (-7.3109264, -4.0054374, -7.3109264, -4.0054374, -2.7144065, 2.7144065)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.28 + 36.61 = 60.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -1.1688954, upper bound: 1.1688953

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4639
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6191
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4639

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -1.1648070, upper bound: 1.1577665
time: 14.49 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688675, upper bound: 1.1688662
time: 5.20 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 19.81 seconds
NS_A1, status: Status.VERIFIED, split count: 1, time: 19.81
Output dim: 2, lower bound: -1.1648070, upper bound: 1.1577665
NS_A2, status: Status.UNKNOWN, split count: 1, time: 19.81
Output dim: 2, lower bound: -1.1688675, upper bound: 1.1688662

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.0258846, -5.5622597, -9.0258932, -5.5622568, -2.7596607, 2.7251792
1: -6.5765324, -3.9591105, -6.5765352, -3.9591038, -2.2075744, 2.2108514
2: 8.3243246, 10.9320049, 8.3243179, 10.9320288, -2.2642126, 2.2531314
3: -6.1232677, -2.8826237, -6.1232758, -2.8826151, -2.9204893, 2.9423370
4: -11.8333645, -7.9824491, -11.8333731, -7.9824457, -2.9800339, 2.9852414
5: -13.6636324, -10.1825533, -13.6636429, -10.1825514, -2.4983902, 2.5127039
6: -15.6556368, -12.3172045, -15.6556492, -12.3171968, -2.3277698, 2.3129454
7: -5.5686011, -2.0476842, -5.5686102, -2.0476756, -3.2501769, 3.2462888
8: -1.9611845, 0.3840852, -1.9611893, 0.3840871, -2.0765119, 2.0609412
9: -7.3109035, -4.0054445, -7.3109140, -4.0054426, -2.7002220, 2.7123060

Time for backsubstitution: 23.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6191
type: B, layer: 1, pos: 6191
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6191

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688673, upper bound: 1.1684430
time: 5.52 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688674, upper bound: 1.1688661
time: 5.32 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 34.18 seconds
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 34.18
Output dim: 2, lower bound: -1.1688673, upper bound: 1.1684430
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 34.18
Output dim: 2, lower bound: -1.1688674, upper bound: 1.1688661

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -8.9886036, -5.5786963, -9.0120907, -5.5641479, -2.7186565, 2.6914310
1: -6.5282736, -3.9994292, -6.5585651, -3.9610293, -2.1524835, 2.1548805
2: 8.3598528, 10.9217129, 8.3297939, 10.9281387, -2.2127385, 2.2159698
3: -6.0636644, -2.9367185, -6.0927734, -2.8848248, -2.8564630, 2.8550038
4: -11.8151140, -8.0142660, -11.8309870, -7.9935989, -2.9479561, 2.9488006
5: -13.6450024, -10.1898527, -13.6593437, -10.1855869, -2.4744072, 2.4954667
6: -15.6315422, -12.3309975, -15.6488914, -12.3194504, -2.2977200, 2.2924585
7: -5.5253358, -2.0600851, -5.5589809, -2.0532866, -3.1980376, 3.2103419
8: -1.9408922, 0.3561540, -1.9568028, 0.3757181, -2.0397105, 2.0259304
9: -7.2787366, -4.0523577, -7.3072505, -4.0275269, -2.6376209, 2.6620774

Time for backsubstitution: 23.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6111

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688666, upper bound: 1.1671216
time: 5.61 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688666, upper bound: 1.1684421
time: 5.51 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -9.0258732, -5.5622635, -9.0258865, -5.5622568, -2.7569733, 2.7126312
1: -6.5765209, -3.9591103, -6.5765285, -3.9591053, -2.1910357, 2.2108355
2: 8.3243313, 10.9320002, 8.3243217, 10.9320278, -2.2374573, 2.2514141
3: -6.1232557, -2.8826251, -6.1232700, -2.8826170, -2.8599701, 2.9305978
4: -11.8333626, -7.9824605, -11.8333731, -7.9824514, -2.9774084, 2.9735065
5: -13.6636276, -10.1825562, -13.6636438, -10.1825533, -2.5000505, 2.5119996
6: -15.6556339, -12.3172016, -15.6556473, -12.3171959, -2.3256860, 2.3129411
7: -5.5685949, -2.0476885, -5.5686054, -2.0476787, -3.2347445, 3.2400579
8: -1.9611802, 0.3840699, -1.9611878, 0.3840780, -2.0706987, 2.0565906
9: -7.3108997, -4.0054688, -7.3109112, -4.0054560, -2.7073898, 2.7092838

Time for backsubstitution: 22.94 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6111
type: A, layer: 1, pos: 6111
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 6111

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675583, upper bound: 1.1688679
time: 5.14 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688666, upper bound: 1.1688653
time: 5.22 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.43 seconds
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 33.43
Output dim: 2, lower bound: -1.1688666, upper bound: 1.1671216
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 33.43
Output dim: 2, lower bound: -1.1688666, upper bound: 1.1684421
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.43
Output dim: 2, lower bound: -1.1675583, upper bound: 1.1688679
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.43
Output dim: 2, lower bound: -1.1688666, upper bound: 1.1688653

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -8.9694939, -5.5944123, -9.0096474, -5.5730467, -2.6904612, 2.6730890
1: -6.5139179, -4.0063591, -6.5539160, -3.9624176, -2.1349912, 2.1422758
2: 8.3742695, 10.9116459, 8.3316383, 10.9228764, -2.1894217, 2.2002888
3: -6.0393181, -2.9561243, -6.0874233, -2.8947575, -2.8228292, 2.8303981
4: -11.8051462, -8.0272884, -11.8255301, -7.9970236, -2.9348783, 2.9258251
5: -13.6358128, -10.1929359, -13.6556263, -10.1861668, -2.4643774, 2.4889307
6: -15.6180487, -12.3480644, -15.6478329, -12.3289862, -2.2709188, 2.2729106
7: -5.4835720, -2.0771976, -5.5368648, -2.0543559, -3.1552749, 3.1708031
8: -1.9334798, 0.3511448, -1.9550104, 0.3742814, -2.0261073, 2.0153189
9: -7.2710443, -4.0622315, -7.3040175, -4.0312476, -2.6256037, 2.6481104

Time for backsubstitution: 22.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A2_A1_A1_A1

### Relational analysis result of NS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688639, upper bound: 1.1656034
time: 17.70 seconds

## Relational analysis of NS_A2_A1_A1_A2

### Relational analysis result of NS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688639, upper bound: 1.1671189
time: 6.75 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -8.9886007, -5.5787029, -9.0120888, -5.5641518, -2.7186432, 2.6881609
1: -6.5282660, -3.9994314, -6.5585604, -3.9610298, -2.1491547, 2.1548262
2: 8.3598557, 10.9217072, 8.3297958, 10.9281359, -2.1977777, 2.2151022
3: -6.0636563, -2.9367275, -6.0927691, -2.8848302, -2.8564472, 2.8361135
4: -11.8151121, -8.0142717, -11.8309841, -7.9936008, -2.9448609, 2.9436131
5: -13.6449966, -10.1898537, -13.6593447, -10.1855888, -2.4735203, 2.4938855
6: -15.6315422, -12.3310051, -15.6488876, -12.3194580, -2.2915764, 2.2786393
7: -5.5253105, -2.0600870, -5.5589662, -2.0532875, -3.1869574, 3.2103205
8: -1.9408889, 0.3561516, -1.9568028, 0.3757162, -2.0411859, 2.0185933
9: -7.2787313, -4.0523624, -7.3072457, -4.0275311, -2.6378984, 2.6620679

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A2_A1_A2_B1

### Relational analysis result of NS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673426, upper bound: 1.1684394
time: 5.22 seconds

## Relational analysis of NS_A2_A1_A2_B2

### Relational analysis result of NS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688638, upper bound: 1.1684395
time: 5.45 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -9.0234270, -5.5711718, -9.0066652, -5.5779719, -2.7386589, 2.6843753
1: -6.5718241, -3.9605217, -6.5625620, -3.9660516, -2.1783810, 2.1938822
2: 8.3261490, 10.9267197, 8.3391075, 10.9219475, -2.2218542, 2.2281685
3: -6.1178899, -2.8925533, -6.0989065, -2.9019980, -2.8357544, 2.8945017
4: -11.8279028, -7.9858952, -11.8234081, -7.9955430, -2.9544382, 2.9604015
5: -13.6599121, -10.1831322, -13.6544819, -10.1856403, -2.4935350, 2.5019426
6: -15.6545782, -12.3267345, -15.6420603, -12.3342638, -2.3061128, 2.2846158
7: -5.5464830, -2.0487618, -5.5268345, -2.0648167, -3.1951342, 3.1972580
8: -1.9593987, 0.3826346, -1.9538302, 0.3790946, -2.0600824, 2.0428376
9: -7.3076649, -4.0092072, -7.3032079, -4.0152011, -2.6936073, 2.6972327

Time for backsubstitution: 22.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1660343, upper bound: 1.1688626
time: 18.53 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675555, upper bound: 1.1688651
time: 5.01 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -9.0258713, -5.5622692, -9.0258837, -5.5622663, -2.7537022, 2.7126174
1: -6.5765166, -3.9591117, -6.5765228, -3.9591064, -2.1910000, 2.2074685
2: 8.3243313, 10.9319973, 8.3243237, 10.9320230, -2.2281761, 2.2350690
3: -6.1232519, -2.8826301, -6.1232615, -2.8826241, -2.8410811, 2.9200597
4: -11.8333588, -7.9824643, -11.8333693, -7.9824572, -2.9695849, 2.9682596
5: -13.6636248, -10.1825562, -13.6636391, -10.1825562, -2.4985495, 2.5111117
6: -15.6556330, -12.3172054, -15.6556463, -12.3172064, -2.3118896, 2.3044448
7: -5.5685816, -2.0476890, -5.5685816, -2.0476816, -3.2347226, 3.2289772
8: -1.9611788, 0.3840690, -1.9611864, 0.3840771, -2.0633583, 2.0580873
9: -7.3108978, -4.0054717, -7.3109064, -4.0054617, -2.7073793, 2.7095633

Time for backsubstitution: 22.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673426, upper bound: 1.1688629
time: 5.85 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688638, upper bound: 1.1688626
time: 4.91 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.65 seconds
NS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1688639, upper bound: 1.1656034
NS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1688639, upper bound: 1.1671189
NS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1673426, upper bound: 1.1684394
NS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1688638, upper bound: 1.1684395
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1660343, upper bound: 1.1688626
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1675555, upper bound: 1.1688651
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1673426, upper bound: 1.1688629
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 33.65
Output dim: 2, lower bound: -1.1688638, upper bound: 1.1688626

## BFS NS instance: NS_A2_A1_A1_A1

### Backsubstitution after applying NS history:
0: -8.9550133, -5.5955644, -9.0011787, -5.5737314, -2.6748562, 2.6630435
1: -6.5120287, -4.0120845, -6.5527868, -3.9656501, -2.1293054, 2.1345022
2: 8.3756466, 10.9038877, 8.3325081, 10.9183350, -2.1827598, 2.1908767
3: -6.0365467, -2.9632716, -6.0858374, -2.8989043, -2.8162589, 2.8218503
4: -11.8038158, -8.0291271, -11.8247423, -7.9981489, -2.9323268, 2.9230199
5: -13.6342106, -10.1989613, -13.6546917, -10.1896763, -2.4585228, 2.4809690
6: -15.6005650, -12.3491697, -15.6376486, -12.3296347, -2.2519255, 2.2610598
7: -5.4785142, -2.0860751, -5.5339012, -2.0595455, -3.1446052, 3.1586914
8: -1.9218364, 0.3501940, -1.9482279, 0.3736782, -2.0135417, 2.0074439
9: -7.2661519, -4.0648670, -7.3011131, -4.0328121, -2.6182470, 2.6412735

Time for backsubstitution: 22.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of NS_A2_A1_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672378, upper bound: 1.1656001
time: 11.90 seconds

## Relational analysis of NS_A2_A1_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688608, upper bound: 1.1656006
time: 12.78 seconds

## BFS NS instance: NS_A2_A1_A1_A2

### Backsubstitution after applying NS history:
0: -8.9723568, -5.5757666, -9.0096331, -5.5730515, -2.6847124, 2.6929259
1: -6.5218019, -4.0032344, -6.5539160, -3.9624245, -2.1441965, 2.1430101
2: 8.3601351, 10.9135246, 8.3316412, 10.9228659, -2.1961751, 2.1993818
3: -6.0526533, -2.9537392, -6.0874214, -2.8947625, -2.8362474, 2.8290339
4: -11.8068790, -8.0215397, -11.8255272, -7.9970255, -2.9367642, 2.9317803
5: -13.6480083, -10.1924419, -13.6556282, -10.1861658, -2.4764395, 2.4870315
6: -15.6210432, -12.3243742, -15.6478109, -12.3289881, -2.2635322, 2.2811866
7: -5.4925299, -2.0749645, -5.5368609, -2.0543642, -3.1646385, 3.1699696
8: -1.9365983, 0.3664098, -1.9550014, 0.3742795, -2.0242157, 2.0317121
9: -7.2732754, -4.0558872, -7.3040123, -4.0312490, -2.6257911, 2.6536765

Time for backsubstitution: 22.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 498
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6219

## Relational analysis of NS_A2_A1_A1_A2_B1

### Relational analysis result of NS_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1672378, upper bound: 1.1671162
time: 8.24 seconds

## Relational analysis of NS_A2_A1_A1_A2_B2

### Relational analysis result of NS_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688608, upper bound: 1.1671160
time: 7.23 seconds

## BFS NS instance: NS_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -8.9801311, -5.5793743, -8.9976120, -5.5653143, -2.7085485, 2.6726742
1: -6.5271592, -4.0027823, -6.5566258, -3.9665387, -2.1415873, 2.1489584
2: 8.3606768, 10.9171619, 8.3312893, 10.9203815, -2.1885076, 2.2082384
3: -6.0620508, -2.9409208, -6.0900373, -2.8918893, -2.8479939, 2.8294945
4: -11.8143244, -8.0153322, -11.8296518, -7.9955249, -2.9419870, 2.9411211
5: -13.6440697, -10.1933756, -13.6577377, -10.1915903, -2.4656296, 2.4880013
6: -15.6213226, -12.3316545, -15.6314726, -12.3205652, -2.2774978, 2.2597227
7: -5.5223484, -2.0652812, -5.5539050, -2.0621605, -3.1749563, 3.1994944
8: -1.9340887, 0.3555951, -1.9452019, 0.3746805, -2.0332923, 2.0060296
9: -7.2758303, -4.0539112, -7.3022871, -4.0302105, -2.6310210, 2.6547480

Time for backsubstitution: 22.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of NS_A2_A1_A2_B1_A1

### Relational analysis result of NS_A2_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673397, upper bound: 1.1668148
time: 6.64 seconds

## Relational analysis of NS_A2_A1_A2_B1_A2

### Relational analysis result of NS_A2_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1673397, upper bound: 1.1684364
time: 5.11 seconds

## BFS NS instance: NS_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -8.9885874, -5.5787024, -9.0148621, -5.5455317, -2.7362399, 2.6825514
1: -6.5282640, -3.9994383, -6.5664415, -3.9580557, -2.1500950, 2.1639457
2: 8.3598585, 10.9216995, 8.3153515, 10.9300137, -2.1969271, 2.2229829
3: -6.0636539, -2.9367332, -6.1061354, -2.8823025, -2.8552117, 2.8495569
4: -11.8151093, -8.0142708, -11.8327122, -7.9878707, -2.9507184, 2.9454966
5: -13.6449986, -10.1898546, -13.6715078, -10.1851597, -2.4713726, 2.5059309
6: -15.6315193, -12.3310032, -15.6517534, -12.2957821, -2.2940624, 2.2708077
7: -5.5253067, -2.0600951, -5.5678520, -2.0511022, -3.1861944, 3.2195573
8: -1.9408793, 0.3561506, -1.9599085, 0.3910537, -2.0497856, 2.0166855
9: -7.2787251, -4.0523658, -7.3093958, -4.0211630, -2.6434627, 2.6622553

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4616
type: B, layer: 1, pos: 4639
type: A, layer: 1, pos: 4654
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of NS_A2_A1_A2_B2_A1

### Relational analysis result of NS_A2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688608, upper bound: 1.1668146
time: 21.13 seconds

## Relational analysis of NS_A2_A1_A2_B2_A2

### Relational analysis result of NS_A2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1688608, upper bound: 1.1684361
time: 5.44 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -9.0149469, -5.5718527, -8.9921675, -5.5791397, -2.7286129, 2.6687803
1: -6.5706949, -3.9637671, -6.5606384, -3.9715943, -2.1707630, 2.1881135
2: 8.3270216, 10.9221821, 8.3405800, 10.9141998, -2.2124882, 2.2204990
3: -6.1163011, -2.8967066, -6.0961633, -2.9090748, -2.8271832, 2.8871784
4: -11.8271151, -7.9870186, -11.8220730, -7.9974813, -2.9514537, 2.9578481
5: -13.6589718, -10.1866446, -13.6528664, -10.1916485, -2.4856243, 2.4960799
6: -15.6443872, -12.3273849, -15.6246281, -12.3353624, -2.2920961, 2.2656631
7: -5.5435038, -2.0539570, -5.5217547, -2.0736992, -3.1830997, 3.1865129
8: -1.9526172, 0.3820300, -1.9422226, 0.3780580, -2.0521097, 2.0302472
9: -7.3047523, -4.0107718, -7.2982512, -4.0178823, -2.6866903, 2.6899018

Time for backsubstitution: 23.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of NS_A2_A2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1660314, upper bound: 1.1672391
time: 6.19 seconds

## Relational analysis of NS_A2_A2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1660314, upper bound: 1.1688597
time: 8.43 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -9.0234127, -5.5711713, -9.0093937, -5.5593462, -2.7537632, 2.6784296
1: -6.5718226, -3.9605293, -6.5704594, -3.9630592, -2.1794043, 2.2030292
2: 8.3261490, 10.9267130, 8.3246670, 10.9238291, -2.2209945, 2.2332258
3: -6.1178865, -2.8925605, -6.1122904, -2.8992994, -2.8341827, 2.8994660
4: -11.8279037, -7.9858975, -11.8251438, -7.9898801, -2.9579849, 2.9623013
5: -13.6599102, -10.1831341, -13.6666660, -10.1852198, -2.4913406, 2.5138731
6: -15.6545515, -12.3267355, -15.6448517, -12.3105831, -2.3086073, 2.2768722
7: -5.5464783, -2.0487709, -5.5357409, -2.0626440, -3.1944146, 3.2066073
8: -1.9593873, 0.3826337, -1.9570322, 0.3943276, -2.0678873, 2.0408101
9: -7.3076582, -4.0092087, -7.3053770, -4.0088539, -2.6991415, 2.6974277

Time for backsubstitution: 23.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6219
type: B, layer: 1, pos: 6219
type: B, layer: 1, pos: 4639
type: B, layer: 1, pos: 4616
type: A, layer: 1, pos: 4616
type: B, layer: 1, pos: 4654
type: A, layer: 1, pos: 4654
type: A, layer: 1, pos: 4666
type: B, layer: 1, pos: 4666
type: B, layer: 1, pos: 520
type: A, layer: 1, pos: 520
type: B, layer: 1, pos: 6231
type: A, layer: 1, pos: 6231
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 498
type: B, layer: 1, pos: 6170
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 6170
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 498
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 929
type: B, layer: 1, pos: 929
type: A, layer: 1, pos: 115
type: B, layer: 1, pos: 115
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 6191

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 6219

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675525, upper bound: 1.1672368
time: 7.15 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -1.1675525, upper bound: 1.1688620
time: 5.49 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -9.0173931, -5.5629473, -9.0113888, -5.5634260, -2.7436600, 2.6970329
1: -6.5753880, -3.9623566, -6.5745907, -3.9646373, -2.1833973, 2.2016909
2: 8.3252048, 10.9274588, 8.3258200, 10.9242716, -2.2188110, 2.2273643
3: -6.1216631, -2.8867831, -6.1205325, -2.8896933, -2.8325253, 2.9127250
4: -11.8325682, -7.9835806, -11.8320341, -7.9843812, -2.9666033, 2.9656942
5: -13.6626863, -10.1860666, -13.6620274, -10.1885614, -2.4906464, 2.5052447
6: -15.6454420, -12.3178549, -15.6382227, -12.3183136, -2.2978656, 2.2854950
7: -5.5656061, -2.0528860, -5.5634956, -2.0565619, -3.2226734, 3.2182078
8: -1.9543977, 0.3834639, -1.9495912, 0.3830357, -2.0553808, 2.0455012
9: -7.3079810, -4.0070357, -7.3059359, -4.0081358, -2.7004733, 2.7022233

Time for backsubstitution: 23.56 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 60.89 + 550.77 = 611.66 seconds
