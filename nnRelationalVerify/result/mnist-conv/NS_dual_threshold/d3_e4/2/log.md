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
execution time: IAR + RelationalAnalysis = 22.43 + 34.16 = 56.58 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -1.6115657, upper bound: 1.6115677

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 495
type: A, layer: 1, pos: 495
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 495

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823086, upper bound: 1.6016604
time: 4.76 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115498, upper bound: 1.6115526
time: 4.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.83 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 9.83
Output dim: 4, lower bound: -1.5823086, upper bound: 1.6016604
NS_B2, status: Status.UNKNOWN, split count: 1, time: 9.83
Output dim: 4, lower bound: -1.6115498, upper bound: 1.6115526

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -13.2960911, -9.0433998, -13.2443447, -9.1210403, -3.4012194, 3.2316446
1: -7.2447991, -3.5148525, -7.1863804, -3.5408008, -3.2778378, 3.2092566
2: -10.0359735, -7.2747846, -10.0132504, -7.2979498, -2.7380238, 2.7384658
3: -12.5336771, -9.4353428, -12.5013952, -9.4653015, -2.6575131, 2.6954241
4: 5.3407826, 8.6435699, 5.4078059, 8.5721321, -3.1224794, 3.1013336
5: -8.9638920, -5.7333994, -8.9391508, -5.7702608, -2.6041985, 2.5981908
6: -12.4801865, -8.9622803, -12.4553852, -8.9797421, -2.3328362, 2.3313751
7: -5.6334686, -2.7762895, -5.5616722, -2.8372817, -2.5207434, 2.5753505
8: -1.1986213, 1.9725802, -1.1583524, 1.9369693, -3.1355906, 3.1309326
9: -6.5652218, -3.8694615, -6.5155478, -3.9069533, -2.4883504, 2.4450278

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 495

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823065, upper bound: 1.5823086
time: 4.79 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823065, upper bound: 1.6016604
time: 4.68 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -13.3209686, -8.9732380, -13.3209591, -8.9732571, -3.5866737, 3.5977523
1: -7.3048587, -3.5086868, -7.3048439, -3.5086889, -3.3905706, 3.3502531
2: -10.0570230, -7.2594447, -10.0570183, -7.2594490, -2.7975740, 2.7975736
3: -12.5703154, -9.4160795, -12.5703058, -9.4160833, -2.8214250, 2.7884665
4: 5.3104191, 8.7127018, 5.3104267, 8.7126923, -3.2933273, 3.2979608
5: -8.9787188, -5.6989908, -8.9787159, -5.6990023, -2.6828928, 2.6940873
6: -12.5030479, -8.9510469, -12.5030422, -8.9513426, -2.4086299, 2.3910868
7: -5.7038975, -2.7505329, -5.7038751, -2.7505383, -2.7433290, 2.7138262
8: -1.2158730, 2.0059879, -1.2158709, 2.0059824, -3.2218554, 3.2218587
9: -6.5885034, -3.8328857, -6.5884991, -3.8328958, -2.5779176, 2.6015453

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115491, upper bound: 1.6115542
time: 4.86 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115492, upper bound: 1.6115537
time: 5.31 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.15 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 32.15
Output dim: 4, lower bound: -1.5823065, upper bound: 1.5823086
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 32.15
Output dim: 4, lower bound: -1.5823065, upper bound: 1.6016604
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 32.15
Output dim: 4, lower bound: -1.6115491, upper bound: 1.6115542
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 32.15
Output dim: 4, lower bound: -1.6115492, upper bound: 1.6115537

## BFS NS instance: NS_B1_A1

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

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.5823534
time: 4.51 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.5823530
time: 4.47 seconds

## BFS NS instance: NS_B1_A2

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

Time for backsubstitution: 22.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823065, upper bound: 1.6016599
time: 4.59 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823059, upper bound: 1.6016600
time: 4.78 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -13.2316999, -8.9936981, -13.2881813, -8.9773293, -3.4926662, 3.5227904
1: -7.2850509, -3.5163152, -7.2980933, -3.5104027, -3.3664904, 3.3322020
2: -9.9834347, -7.2925525, -10.0303936, -7.2677021, -2.7157326, 2.7378411
3: -12.5505381, -9.4759684, -12.5648260, -9.4385080, -2.7613816, 2.7211413
4: 5.3413095, 8.7029972, 5.3199091, 8.7101574, -3.2604728, 3.2775850
5: -8.9616642, -5.7460065, -8.9738293, -5.7168794, -2.6470103, 2.6427846
6: -12.4220314, -8.9665146, -12.4730415, -8.9538145, -2.3275290, 2.3445015
7: -5.6807051, -2.7950132, -5.6991549, -2.7667911, -2.6906929, 2.6644154
8: -1.1593854, 1.9811063, -1.1955559, 2.0000944, -3.1594799, 3.1766622
9: -6.5759768, -3.8428702, -6.5850210, -3.8363566, -2.5564380, 2.5830150

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115490, upper bound: 1.6115513
time: 6.83 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115490, upper bound: 1.6115537
time: 4.82 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -13.3209639, -8.9732389, -13.3209591, -8.9732571, -3.5343742, 3.5963390
1: -7.3048592, -3.5086877, -7.3048439, -3.5086889, -3.3905706, 3.3520241
2: -10.0570202, -7.2594452, -10.0570183, -7.2594490, -2.7975712, 2.7975731
3: -12.5703144, -9.4160805, -12.5703058, -9.4160833, -2.8201265, 2.7646508
4: 5.3104210, 8.7127028, 5.3104267, 8.7126923, -3.2933273, 3.2973976
5: -8.9787178, -5.6989937, -8.9787159, -5.6990023, -2.6828928, 2.6877697
6: -12.5030451, -8.9510479, -12.5030422, -8.9513426, -2.3477793, 2.3910873
7: -5.7038960, -2.7505326, -5.7038751, -2.7505383, -2.7409410, 2.6861830
8: -1.2158694, 2.0059886, -1.2158709, 2.0059824, -3.2218518, 3.2218595
9: -6.5885034, -3.8328862, -6.5884991, -3.8328958, -2.5776381, 2.6026559

Time for backsubstitution: 22.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046489
time: 4.28 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115479, upper bound: 1.6115525
time: 4.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.71 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.5823060, upper bound: 1.5823534
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.5823060, upper bound: 1.5823530
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.5823065, upper bound: 1.6016599
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.5823059, upper bound: 1.6016600
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.6115490, upper bound: 1.6115513
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.6115490, upper bound: 1.6115537
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046489
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 31.71
Output dim: 4, lower bound: -1.6115479, upper bound: 1.6115525

## BFS NS instance: NS_B1_A1_A1

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

Time for backsubstitution: 22.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6250
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823510
time: 4.61 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823530
time: 4.61 seconds

## BFS NS instance: NS_B1_A1_A2

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

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_B1_A1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

### Candidate
type: B, layer: 1, pos: 6250

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823529
time: 4.56 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823530, upper bound: 1.5823530
time: 4.71 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -13.2824707, -8.9780760, -13.1549864, -9.1414547, -3.3441319, 3.1522748
1: -7.2876906, -3.5105517, -7.1668262, -3.5483453, -3.2867517, 3.1958027
2: -10.0300140, -7.2708406, -9.9397202, -7.3311028, -2.6926889, 2.6688795
3: -12.5642185, -9.4422235, -12.4820805, -9.5253382, -2.6137576, 2.6421216
4: 5.3231382, 8.7100353, 5.4385414, 8.5623474, -3.1277709, 3.0770075
5: -8.9720955, -5.7173038, -8.9218502, -5.8169279, -2.5643373, 2.5655422
6: -12.4707870, -8.9542913, -12.3745794, -8.9950533, -2.3111701, 2.2557008
7: -5.6984344, -2.7728167, -5.5388203, -2.8817830, -2.4816384, 2.5380807
8: -1.1915166, 1.9998770, -1.1019380, 1.9122748, -3.1037915, 3.1018150
9: -6.5802317, -3.8376265, -6.5032811, -3.9169121, -2.4786291, 2.4274840

Time for backsubstitution: 22.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6250
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
time: 4.90 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016600
time: 4.71 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -13.3151941, -8.9740057, -13.2443409, -9.1210403, -3.4182243, 3.1910315
1: -7.2944493, -3.5088406, -7.1863785, -3.5408010, -3.3045917, 3.2199378
2: -10.0566406, -7.2625899, -10.0132475, -7.2979488, -2.7586918, 2.7396173
3: -12.5696926, -9.4197998, -12.5013914, -9.4653053, -2.6549063, 2.7009723
4: 5.3136616, 8.7125702, 5.4078054, 8.5721302, -3.1475706, 3.1091053
5: -8.9769821, -5.6994271, -8.9391508, -5.7702627, -2.6086254, 2.6102633
6: -12.5007954, -8.9518223, -12.4553833, -8.9797421, -2.3590922, 2.2760217
7: -5.7031469, -2.7565625, -5.5616713, -2.8372827, -2.5034363, 2.5882094
8: -1.2118330, 2.0057664, -1.1583488, 1.9369683, -3.1488013, 3.1641152
9: -6.5837092, -3.8341627, -6.5155468, -3.9069557, -2.5004601, 2.4469924

Time for backsubstitution: 23.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 6250

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 90

### Candidate
type: A, layer: 1, pos: 6250

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
time: 4.39 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016599
time: 4.45 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -13.2316999, -8.9936981, -13.2316933, -8.9937153, -3.4724922, 3.4835727
1: -7.2850509, -3.5163152, -7.2850342, -3.5163178, -3.3587313, 3.3187008
2: -9.9834347, -7.2925525, -9.9834309, -7.2925563, -2.6908784, 2.6908784
3: -12.5505381, -9.4759684, -12.5505314, -9.4759722, -2.7336788, 2.7007220
4: 5.3413095, 8.7029972, 5.3413172, 8.7029877, -3.2525244, 3.2569611
5: -8.9616642, -5.7460065, -8.9616642, -5.7460160, -2.6186075, 2.6300113
6: -12.4220314, -8.9665146, -12.4220238, -8.9667826, -2.3149695, 2.2974262
7: -5.6807051, -2.7950132, -5.6806850, -2.7950187, -2.6738997, 2.6443958
8: -1.1593854, 1.9811063, -1.1593807, 1.9810987, -3.1404841, 3.1404870
9: -6.5759768, -3.8428702, -6.5759697, -3.8428788, -2.5464382, 2.5700588

Time for backsubstitution: 25.09 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_B2_A1_B1_B1

### Relational analysis result of NS_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046492
time: 4.50 seconds

## Relational analysis of NS_B2_A1_B1_B2

### Relational analysis result of NS_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115477, upper bound: 1.6115530
time: 4.92 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -13.2316999, -8.9936981, -13.3208590, -8.9745789, -3.4948292, 3.5279891
1: -7.2850509, -3.5163152, -7.3046246, -3.5091662, -3.3679900, 3.3390369
2: -9.9834347, -7.2925525, -10.0567713, -7.2635641, -2.7198706, 2.7642188
3: -12.5505381, -9.4759684, -12.5702248, -9.4195328, -2.7737947, 2.7264979
4: 5.3413095, 8.7029972, 5.3111925, 8.7117100, -3.2623501, 3.2853708
5: -8.9616642, -5.7460065, -8.9786558, -5.7017021, -2.6528773, 2.6466658
6: -12.4220314, -8.9665146, -12.5015793, -8.9517307, -2.3295860, 2.3568022
7: -5.6807051, -2.7950132, -5.7024679, -2.7505960, -2.6951945, 2.6678190
8: -1.1593854, 1.9811063, -1.2157483, 2.0041814, -3.1635668, 3.1968546
9: -6.5759768, -3.8428702, -6.5879679, -3.8329101, -2.5572085, 2.5844550

Time for backsubstitution: 25.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_B2_A1_B2_B1

### Relational analysis result of NS_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046493
time: 4.44 seconds

## Relational analysis of NS_B2_A1_B2_B2

### Relational analysis result of NS_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6115478, upper bound: 1.6115530
time: 5.09 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -13.3205423, -8.9737053, -13.3198261, -8.9742212, -3.5330839, 3.5940609
1: -7.3046312, -3.5087929, -7.3059473, -3.5089908, -3.3900728, 3.3499041
2: -10.0569420, -7.2597094, -10.0566912, -7.2602029, -2.7967391, 2.7969818
3: -12.5702438, -9.4163332, -12.5745258, -9.4167023, -2.8191371, 2.7686129
4: 5.3109541, 8.7124157, 5.3120403, 8.7121515, -3.2922621, 3.2950678
5: -8.9782963, -5.7001648, -8.9768791, -5.7011065, -2.6803803, 2.6848540
6: -12.5011902, -8.9511881, -12.4998302, -8.9507818, -2.3443131, 2.3877413
7: -5.7036972, -2.7507019, -5.7047644, -2.7509089, -2.7389894, 2.6868582
8: -1.2157340, 2.0058107, -1.2152553, 2.0056152, -3.2213492, 3.2210660
9: -6.5882530, -3.8329301, -6.5877085, -3.8327808, -2.5775514, 2.6015706

Time for backsubstitution: 26.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857455, upper bound: 1.5857457
time: 4.71 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046495
time: 4.46 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -13.3209620, -8.9732428, -13.3209562, -8.9732628, -3.5341759, 3.5972219
1: -7.3048563, -3.5090611, -7.3048429, -3.5094151, -3.3922420, 3.3520117
2: -10.0570183, -7.2594476, -10.0570164, -7.2594509, -2.7975674, 2.7975688
3: -12.5703144, -9.4171247, -12.5703049, -9.4181175, -2.8255224, 2.7635114
4: 5.3104248, 8.7127018, 5.3104324, 8.7126904, -3.2930846, 3.2976687
5: -8.9787159, -5.6989989, -8.9787092, -5.6990137, -2.6831150, 2.6877563
6: -12.5030422, -8.9510489, -12.5030365, -8.9513435, -2.3477731, 2.3879740
7: -5.7038951, -2.7508724, -5.7038727, -2.7512002, -2.7395158, 2.6857090
8: -1.2158699, 2.0059867, -1.2158687, 2.0059776, -3.2218475, 3.2218554
9: -6.5885038, -3.8328872, -6.5884972, -3.8328972, -2.5774670, 2.6024930

Time for backsubstitution: 25.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 495
type: B, layer: 1, pos: 6250

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6046486, upper bound: 1.5857457
time: 4.49 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -1.6046484, upper bound: 1.6115506
time: 5.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 35.77 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823510
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823530
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823508, upper bound: 1.5823529
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823530, upper bound: 1.5823530
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016600
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016572
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5823060, upper bound: 1.6016599
NS_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046492
NS_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.6115477, upper bound: 1.6115530
NS_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046493
NS_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.6115478, upper bound: 1.6115530
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5857455, upper bound: 1.5857457
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.5857455, upper bound: 1.6046495
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.6046486, upper bound: 1.5857457
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 35.77
Output dim: 4, lower bound: -1.6046484, upper bound: 1.6115506

## BFS NS instance: NS_B1_A1_A1_B1

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

Time for backsubstitution: 24.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 90

## BFS NS instance: NS_B1_A1_A1_B2

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

Time for backsubstitution: 24.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_B1_A1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.58 + 543.94 = 600.52 seconds
