## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.579334386


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-13.3209696, -8.9732342, -13.3209696, -8.9732342, -3.5872164, 3.5872157)
1: (-7.3048649, -3.5086851, -7.3048649, -3.5086851, -3.3974328, 3.3974328)
2: (-10.0570250, -7.2594433, -10.0570250, -7.2594433, -2.7975817, 2.7975817)
3: (-12.5703182, -9.4160776, -12.5703182, -9.4160776, -2.7899914, 2.7899911)
4: (5.3104172, 8.7127075, 5.3104172, 8.7127075, -3.2979751, 3.2979755)
5: (-8.9787188, -5.6989875, -8.9787188, -5.6989875, -2.6998472, 2.6998470)
6: (-12.5030499, -8.9509468, -12.5030499, -8.9509468, -2.3918996, 2.3918998)
7: (-5.7039032, -2.7505307, -5.7039032, -2.7505307, -2.7433391, 2.7433386)
8: (-1.2158759, 2.0059929, -1.2158759, 2.0059929, -3.2218688, 3.2218688)
9: (-6.5885067, -3.8328815, -6.5885067, -3.8328815, -2.5790672, 2.5790672)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.04 + 34.21 = 56.25 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.6115657, upper bound: 1.6115677

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016583, upper bound: 1.5823062
time: 3.85 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115522, upper bound: 1.6115494
time: 6.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.41
Output dim: 4, lower bound: -1.6016583, upper bound: 1.5823062
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.41
Output dim: 4, lower bound: -1.6115522, upper bound: 1.6115494

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -13.2443447, -9.1210403, -13.2960911, -9.0433998, -3.2316456, 3.4012194
1: -7.1863804, -3.5408008, -7.2447991, -3.5148525, -3.2092571, 3.2778387
2: -10.0132504, -7.2979498, -10.0359735, -7.2747846, -2.7384658, 2.7380238
3: -12.5013952, -9.4653015, -12.5336771, -9.4353428, -2.6954241, 2.6575134
4: 5.4078059, 8.5721321, 5.3407826, 8.6435699, -3.1013346, 3.1224790
5: -8.9391508, -5.7702608, -8.9638920, -5.7333994, -2.5981913, 2.6041982
6: -12.4553852, -8.9797421, -12.4801865, -8.9622803, -2.3313751, 2.3328364
7: -5.5616722, -2.8372817, -5.6334686, -2.7762895, -2.5753503, 2.5207434
8: -1.1583524, 1.9369693, -1.1986213, 1.9725802, -3.1309326, 3.1355906
9: -6.5155478, -3.9069533, -6.5652218, -3.8694615, -2.4450278, 2.4883509

Time for backsubstitution: 20.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823065, upper bound: 1.5823086
time: 4.51 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823065, upper bound: 1.5823061
time: 4.29 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -13.3209591, -8.9732571, -13.3209686, -8.9732380, -3.5977516, 3.5866730
1: -7.3048439, -3.5086889, -7.3048587, -3.5086868, -3.3502531, 3.3905711
2: -10.0570183, -7.2594490, -10.0570230, -7.2594447, -2.7975736, 2.7975740
3: -12.5703058, -9.4160833, -12.5703154, -9.4160795, -2.7884665, 2.8214250
4: 5.3104267, 8.7126923, 5.3104191, 8.7127018, -3.2979603, 3.2933283
5: -8.9787159, -5.6990023, -8.9787188, -5.6989908, -2.6940875, 2.6828928
6: -12.5030422, -8.9513426, -12.5030479, -8.9510469, -2.3910875, 2.4086299
7: -5.7038751, -2.7505383, -5.7038975, -2.7505329, -2.7138257, 2.7433295
8: -1.2158709, 2.0059824, -1.2158730, 2.0059879, -3.2218587, 3.2218554
9: -6.5884991, -3.8328958, -6.5885034, -3.8328857, -2.6015449, 2.5779171

Time for backsubstitution: 20.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823086, upper bound: 1.6016604
time: 4.75 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823064, upper bound: 1.6016579
time: 4.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.52 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.52
Output dim: 4, lower bound: -1.5823065, upper bound: 1.5823086
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.52
Output dim: 4, lower bound: -1.5823065, upper bound: 1.5823061
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.52
Output dim: 4, lower bound: -1.5823086, upper bound: 1.6016604
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.52
Output dim: 4, lower bound: -1.5823064, upper bound: 1.6016579

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -13.2443447, -9.1210403, -13.2443447, -9.1210403, -3.1453581, 3.1453581
1: -7.1863804, -3.5408008, -7.1863804, -3.5408008, -3.1798306, 3.1798301
2: -10.0132504, -7.2979498, -10.0132504, -7.2979498, -2.7153006, 2.7153006
3: -12.5013952, -9.4653015, -12.5013952, -9.4653015, -2.6198587, 2.6198585
4: 5.4078059, 8.5721321, 5.4078059, 8.5721321, -3.0476036, 3.0476036
5: -8.9391508, -5.7702608, -8.9391508, -5.7702608, -2.5618534, 2.5618539
6: -12.4553852, -8.9797421, -12.4553852, -8.9797421, -2.3097830, 2.3097830
7: -5.5616722, -2.8372817, -5.5616722, -2.8372817, -2.4739885, 2.4739881
8: -1.1583524, 1.9369693, -1.1583524, 1.9369693, -3.0953217, 3.0953217
9: -6.5155478, -3.9069533, -6.5155478, -3.9069533, -2.4142606, 2.4142604

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823086
time: 4.51 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823081
time: 4.39 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -13.2443447, -9.1210403, -13.3151941, -8.9740057, -3.2457228, 3.4196458
1: -7.1863804, -3.5408008, -7.2944493, -3.5088406, -3.2199392, 3.3048539
2: -10.0132504, -7.2979498, -10.0566406, -7.2625899, -2.7506604, 2.7586908
3: -12.5013952, -9.4653015, -12.5696926, -9.4197998, -2.7022209, 2.6809349
4: 5.4078059, 8.5721321, 5.3136616, 8.7125702, -3.1098795, 3.1482012
5: -8.9391508, -5.7702608, -8.9769821, -5.6994271, -2.6135564, 2.6149435
6: -12.4553852, -8.9797421, -12.5007954, -8.9518223, -2.3365192, 2.3590930
7: -5.5616722, -2.8372817, -5.7031469, -2.7565625, -2.5936894, 2.5307386
8: -1.1583524, 1.9369693, -1.2118330, 2.0057664, -3.1641188, 3.1488023
9: -6.5155478, -3.9069533, -6.5837092, -3.8341627, -2.4487886, 2.4971895

Time for backsubstitution: 22.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823062
time: 4.44 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823507, upper bound: 1.5823058
time: 4.21 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -13.3151941, -8.9740057, -13.2443447, -9.1210403, -3.4196463, 3.2457225
1: -7.2944493, -3.5088406, -7.1863804, -3.5408008, -3.3048544, 3.2199388
2: -10.0566406, -7.2625899, -10.0132504, -7.2979498, -2.7586908, 2.7506604
3: -12.5696926, -9.4197998, -12.5013952, -9.4653015, -2.6809349, 2.7022212
4: 5.3136616, 8.7125702, 5.4078059, 8.5721321, -3.1482000, 3.1098793
5: -8.9769821, -5.6994271, -8.9391508, -5.7702608, -2.6149440, 2.6135566
6: -12.5007954, -8.9518223, -12.4553852, -8.9797421, -2.3590927, 2.3365195
7: -5.7031469, -2.7565625, -5.5616722, -2.8372817, -2.5307388, 2.5936899
8: -1.2118330, 2.0057664, -1.1583524, 1.9369693, -3.1488023, 3.1641188
9: -6.5837092, -3.8341627, -6.5155478, -3.9069533, -2.4971895, 2.4487889

Time for backsubstitution: 22.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823058, upper bound: 1.6016576
time: 4.86 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
time: 4.75 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -13.3209591, -8.9732571, -13.3209591, -8.9732571, -3.5977278, 3.5977275
1: -7.3048439, -3.5086889, -7.3048439, -3.5086889, -3.3502464, 3.3502455
2: -10.0570183, -7.2594490, -10.0570183, -7.2594490, -2.7975693, 2.7975693
3: -12.5703058, -9.4160833, -12.5703058, -9.4160833, -2.8214116, 2.8214116
4: 5.3104267, 8.7126923, 5.3104267, 8.7126923, -3.2933207, 3.2933207
5: -8.9787159, -5.6990023, -8.9787159, -5.6990023, -2.6828852, 2.6828856
6: -12.5030422, -8.9513426, -12.5030422, -8.9513426, -2.4086218, 2.4086218
7: -5.7038751, -2.7505383, -5.7038751, -2.7505383, -2.7138214, 2.7138209
8: -1.2158709, 2.0059824, -1.2158709, 2.0059824, -3.2218533, 3.2218533
9: -6.5884991, -3.8328958, -6.5884991, -3.8328958, -2.6015444, 2.6027341

Time for backsubstitution: 23.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90

Time for candidate selection: 0.07 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6115519
time: 4.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823059, upper bound: 1.6115514
time: 4.46 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.25 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823086
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823081
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823062
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823507, upper bound: 1.5823058
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823058, upper bound: 1.6016576
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6115519
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.25
Output dim: 4, lower bound: -1.5823059, upper bound: 1.6115514

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -13.1549864, -9.1414547, -13.2115250, -9.1251020, -3.0518789, 3.0779858
1: -7.1668262, -3.5483453, -7.1797132, -3.5424819, -3.1557193, 3.1617203
2: -9.9397202, -7.3311028, -9.9866438, -7.3062530, -2.6334672, 2.6555409
3: -12.4820805, -9.5253382, -12.4963055, -9.4877586, -2.5596256, 2.5522819
4: 5.4385414, 8.5623474, 5.4172578, 8.5695362, -3.0147815, 3.0274529
5: -8.9218502, -5.8169279, -8.9341793, -5.7880025, -2.5262523, 2.5111792
6: -12.3745794, -8.9950533, -12.4254637, -8.9821262, -2.2289424, 2.2631421
7: -5.5388203, -2.8817830, -5.5570507, -2.8535597, -2.4314773, 2.4248130
8: -1.1019380, 1.9122748, -1.1380420, 1.9311433, -3.0330813, 3.0503168
9: -6.5032811, -3.9169121, -6.5121675, -3.9104037, -2.3929186, 2.3957481

Time for backsubstitution: 22.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 90

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823510
time: 4.66 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823530
time: 4.62 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -13.2443409, -9.1210403, -13.2443447, -9.1210403, -3.0934458, 3.1439376
1: -7.1863785, -3.5408010, -7.1863804, -3.5408008, -3.1798296, 3.1815991
2: -10.0132475, -7.2979488, -10.0132504, -7.2979498, -2.6905251, 2.7153015
3: -12.5013914, -9.4653053, -12.5013952, -9.4653015, -2.6186085, 2.5955935
4: 5.4078054, 8.5721302, 5.4078059, 8.5721321, -3.0476027, 3.0476046
5: -8.9391508, -5.7702627, -8.9391508, -5.7702608, -2.5618529, 2.5555360
6: -12.4553833, -8.9797421, -12.4553852, -8.9797421, -2.2492852, 2.3097823
7: -5.5616713, -2.8372827, -5.5616722, -2.8372817, -2.4739876, 2.4468451
8: -1.1583488, 1.9369683, -1.1583524, 1.9369693, -3.0953181, 3.0953207
9: -6.5155468, -3.9069557, -6.5155478, -3.9069533, -2.4139535, 2.4177794

Time for backsubstitution: 22.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 90

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823529
time: 4.56 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823530
time: 4.54 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -13.1549864, -9.1414547, -13.2824707, -8.9780760, -3.1522746, 3.3441315
1: -7.1668262, -3.5483453, -7.2876906, -3.5105517, -3.1958017, 3.2867517
2: -9.9397202, -7.3311028, -10.0300140, -7.2708406, -2.6688795, 2.6926892
3: -12.4820805, -9.5253382, -12.5642185, -9.4422235, -2.6421213, 2.6137576
4: 5.4385414, 8.5623474, 5.3231382, 8.7100353, -3.0770068, 3.1277711
5: -8.9218502, -5.8169279, -8.9720955, -5.7173038, -2.5655422, 2.5643370
6: -12.3745794, -8.9950533, -12.4707870, -8.9542913, -2.2557011, 2.3111701
7: -5.5388203, -2.8817830, -5.6984344, -2.7728167, -2.5380807, 2.4816382
8: -1.1019380, 1.9122748, -1.1915166, 1.9998770, -3.1018150, 3.1037915
9: -6.5032811, -3.9169121, -6.5802317, -3.8376265, -2.4274840, 2.4786291

Time for backsubstitution: 22.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 90

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016580, upper bound: 1.5823059
time: 4.13 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016552, upper bound: 1.5823058
time: 4.34 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -13.2443409, -9.1210403, -13.3151941, -8.9740057, -3.1910315, 3.4182246
1: -7.1863785, -3.5408010, -7.2944493, -3.5088406, -3.2199383, 3.3045919
2: -10.0132475, -7.2979488, -10.0566406, -7.2625899, -2.7396173, 2.7586918
3: -12.5013914, -9.4653053, -12.5696926, -9.4197998, -2.7009726, 2.6549060
4: 5.4078054, 8.5721302, 5.3136616, 8.7125702, -3.1091051, 3.1475704
5: -8.9391508, -5.7702627, -8.9769821, -5.6994271, -2.6102633, 2.6086257
6: -12.4553833, -8.9797421, -12.5007954, -8.9518223, -2.2760215, 2.3590922
7: -5.5616713, -2.8372827, -5.7031469, -2.7565625, -2.5882092, 2.5034361
8: -1.1583488, 1.9369683, -1.2118330, 2.0057664, -3.1641152, 3.1488013
9: -6.5155468, -3.9069557, -6.5837092, -3.8341627, -2.4469924, 2.5004601

Time for backsubstitution: 23.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016552, upper bound: 1.5823058
time: 4.31 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6016576, upper bound: 1.5823058
time: 4.28 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -13.2260571, -8.9944620, -13.2115250, -9.1251020, -3.3256063, 3.1530426
1: -7.2746181, -3.5164695, -7.1797132, -3.5424819, -3.2809467, 3.2017603
2: -9.9830484, -7.2956901, -9.9866438, -7.3062530, -2.6767955, 2.6909537
3: -12.5499249, -9.4796896, -12.4963055, -9.4877586, -2.6114440, 2.6348670
4: 5.3445454, 8.7028675, 5.4172578, 8.5695362, -3.1152263, 3.0896902
5: -8.9599285, -5.7464423, -8.9341793, -5.7880025, -2.5755723, 2.5626240
6: -12.4197655, -8.9672499, -12.4254637, -8.9821262, -2.2779603, 2.2899163
7: -5.6799755, -2.8010485, -5.5570507, -2.8535597, -2.4754319, 2.5441077
8: -1.1553361, 1.9808795, -1.1380420, 1.9311433, -3.0864794, 3.1189215
9: -6.5711861, -3.8441567, -6.5121675, -3.9104037, -2.4757204, 2.4303696

Time for backsubstitution: 23.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598134, upper bound: 1.6014895
time: 4.62 seconds

## Relational analysis of NS_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
time: 4.85 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016571
time: 4.32 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -13.3151875, -8.9740067, -13.2443447, -9.1210403, -3.3674164, 3.2356436
1: -7.2944498, -3.5088406, -7.1863804, -3.5408008, -3.3048534, 3.2217093
2: -10.0566368, -7.2625904, -10.0132504, -7.2979498, -2.7374105, 2.7506599
3: -12.5696917, -9.4198017, -12.5013952, -9.4653015, -2.6745205, 2.6779313
4: 5.3136616, 8.7125702, 5.4078059, 8.5721321, -3.1473894, 3.1092637
5: -8.9769812, -5.6994276, -8.9391508, -5.7702608, -2.6149435, 2.6072340
6: -12.5007906, -8.9518223, -12.4553852, -8.9797421, -2.2982469, 2.3365197
7: -5.7031479, -2.7565618, -5.5616722, -2.8372817, -2.5252056, 2.5659339
8: -1.2118309, 2.0057654, -1.1583524, 1.9369693, -3.1488001, 3.1641178
9: -6.5837088, -3.8341637, -6.5155478, -3.9069533, -2.4969101, 2.4501567

Time for backsubstitution: 25.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5598135, upper bound: 1.6014889
time: 4.53 seconds

## Relational analysis of NS_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
time: 4.77 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823059, upper bound: 1.6016571
time: 4.59 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -13.2316933, -8.9937153, -13.2881813, -8.9773293, -3.5037212, 3.5302494
1: -7.2850342, -3.5163178, -7.2980933, -3.5104027, -3.3264437, 3.3321958
2: -9.9834309, -7.2925563, -10.0303936, -7.2677021, -2.7157288, 2.7378373
3: -12.5505314, -9.4759722, -12.5648260, -9.4385080, -2.7613688, 2.7540858
4: 5.3413172, 8.7029877, 5.3199091, 8.7101574, -3.2604671, 3.2730427
5: -8.9616642, -5.7460160, -8.9738293, -5.7168794, -2.6470027, 2.6315091
6: -12.4220238, -8.9667826, -12.4730415, -8.9538145, -2.3275213, 2.3620379
7: -5.6806850, -2.7950187, -5.6991549, -2.7667911, -2.6671629, 2.6644111
8: -1.1593807, 1.9810987, -1.1955559, 2.0000944, -3.1594751, 3.1766546
9: -6.5759697, -3.8428788, -6.5850210, -3.8363566, -2.5800619, 2.5841770

Time for backsubstitution: 25.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.08 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046491
time: 4.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954327, upper bound: 1.6115529
time: 4.70 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -13.3209553, -8.9732580, -13.3209591, -8.9732571, -3.5454283, 3.5963130
1: -7.3048415, -3.5086896, -7.3048439, -3.5086889, -3.3502455, 3.3520174
2: -10.0570164, -7.2594490, -10.0570183, -7.2594490, -2.7975674, 2.7975693
3: -12.5703068, -9.4160852, -12.5703058, -9.4160833, -2.8201146, 2.7975953
4: 5.3104286, 8.7126913, 5.3104267, 8.7126923, -3.2933216, 3.2933245
5: -8.9787159, -5.6990042, -8.9787159, -5.6990023, -2.6828847, 2.6765676
6: -12.5030403, -8.9513445, -12.5030422, -8.9513426, -2.3477721, 2.4086213
7: -5.7038751, -2.7505386, -5.7038751, -2.7505383, -2.7138214, 2.6861773
8: -1.2158663, 2.0059819, -1.2158709, 2.0059824, -3.2218487, 3.2218528
9: -6.5884986, -3.8328953, -6.5884991, -3.8328958, -2.5997481, 2.6059999

Time for backsubstitution: 25.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90

Time for candidate selection: 0.07 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954348, upper bound: 1.6115535
time: 4.60 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5954349, upper bound: 1.6115516
time: 4.96 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 35.48 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823510
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823530
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823529
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823530
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.6016580, upper bound: 1.5823059
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.6016552, upper bound: 1.5823058
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.6016552, upper bound: 1.5823058
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.6016576, upper bound: 1.5823058
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016571
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5823059, upper bound: 1.6016571
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046491
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5954327, upper bound: 1.6115529
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5954348, upper bound: 1.6115535
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.48
Output dim: 4, lower bound: -1.5954349, upper bound: 1.6115516

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -13.1549864, -9.1414547, -13.1549864, -9.1414547, -3.0317001, 3.0317001
1: -7.1668262, -3.5483453, -7.1668262, -3.5483453, -3.1479931, 3.1479931
2: -9.9397202, -7.3311028, -9.9397202, -7.3311028, -2.6086173, 2.6086173
3: -12.4820805, -9.5253382, -12.4820805, -9.5253382, -2.5318356, 2.5318358
4: 5.4385414, 8.5623474, 5.4385414, 8.5623474, -3.0068064, 3.0068064
5: -8.9218502, -5.8169279, -8.9218502, -5.8169279, -2.4983168, 2.4983170
6: -12.3745794, -8.9950533, -12.3745794, -8.9950533, -2.2163706, 2.2163703
7: -5.5388203, -2.8817830, -5.5388203, -2.8817830, -2.4050360, 2.4050357
8: -1.1019380, 1.9122748, -1.1019380, 1.9122748, -3.0142128, 3.0142128
9: -6.5032811, -3.9169121, -6.5032811, -3.9169121, -2.3829005, 2.3829007

Time for backsubstitution: 25.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90

Time for candidate selection: 0.08 seconds

### Candidate
type: A, layer: 1, pos: 90

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -13.1549864, -9.1414547, -13.2442541, -9.1223497, -3.0540061, 3.1003737
1: -7.1668262, -3.5483453, -7.1861629, -3.5412786, -3.1572104, 3.1686239
2: -9.9397202, -7.3311028, -10.0130043, -7.3020520, -2.6376681, 2.6704602
3: -12.4820805, -9.5253382, -12.5013123, -9.4687214, -2.5746920, 2.5577486
4: 5.4385414, 8.5623474, 5.4085531, 8.5710878, -3.0166645, 3.0359015
5: -8.9218502, -5.8169279, -8.9390898, -5.7728987, -2.5319943, 2.5151274
6: -12.3745794, -8.9950533, -12.4539175, -8.9801407, -2.2310228, 2.2747288
7: -5.5388203, -2.8817830, -5.5602965, -2.8373380, -2.4374263, 2.4281750
8: -1.1019380, 1.9122748, -1.1582377, 1.9351873, -3.0371253, 3.0705125
9: -6.5032811, -3.9169121, -6.5150533, -3.9069667, -2.3936901, 2.3987713

Time for backsubstitution: 25.31 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.25 + 545.52 = 601.77 seconds
