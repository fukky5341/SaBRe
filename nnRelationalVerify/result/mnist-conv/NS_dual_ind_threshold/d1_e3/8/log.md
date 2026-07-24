## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.171128188


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.1082363, -8.0845366, -9.1082363, -8.0845366, -0.7749262, 0.7749262)
1: (-13.5773735, -12.4230623, -13.5773735, -12.4230623, -0.6608329, 0.6608324)
2: (-6.0350904, -5.1668196, -6.0350904, -5.1668196, -0.4653335, 0.4653327)
3: (-8.3463717, -7.4084873, -8.3463717, -7.4084873, -0.7160082, 0.7160082)
4: (-10.1672878, -9.1996689, -10.1672878, -9.1996689, -0.4615841, 0.4615841)
5: (-8.9240236, -8.0944824, -8.9240236, -8.0944824, -0.4377449, 0.4377451)
6: (-11.1597443, -10.2314358, -11.1597443, -10.2314358, -0.5711021, 0.5711021)
7: (-12.6991444, -11.8378401, -12.6991444, -11.8378401, -0.4696124, 0.4696121)
8: (11.9400930, 12.5971460, 11.9400930, 12.5971460, -0.4565544, 0.4565542)
9: (-5.7965894, -4.9964890, -5.7965894, -4.9964890, -0.4133899, 0.4133902)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.00 + 33.60 = 55.60 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1746203, upper bound: 0.1746206

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4574
type: A, layer: 1, pos: 444

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 4574

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744166, upper bound: 0.1746202
time: 5.19 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746199, upper bound: 0.1746202
time: 10.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 16.17 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 16.17
Output dim: 8, lower bound: -0.1744166, upper bound: 0.1746202
NS_A2, status: Status.UNKNOWN, split count: 1, time: 16.17
Output dim: 8, lower bound: -0.1746199, upper bound: 0.1746202

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.1071882, -8.0876436, -9.1081104, -8.0858078, -0.7719812, 0.7716351
1: -13.5765257, -12.4258137, -13.5773010, -12.4241848, -0.6583266, 0.6579776
2: -6.0344195, -5.1673446, -6.0349598, -5.1670322, -0.4643116, 0.4646928
3: -8.3461752, -7.4096885, -8.3463726, -7.4089594, -0.7153125, 0.7147627
4: -10.1668844, -9.2013674, -10.1672659, -9.2003584, -0.4604819, 0.4598360
5: -8.9229603, -8.0946026, -8.9236050, -8.0944939, -0.4365458, 0.4371457
6: -11.1590519, -10.2331362, -11.1596346, -10.2321224, -0.5692539, 0.5692554
7: -12.6985264, -11.8396225, -12.6990852, -11.8385592, -0.4683013, 0.4677856
8: 11.9414721, 12.5967922, 11.9406404, 12.5971260, -0.4550962, 0.4556046
9: -5.7964439, -4.9966192, -5.7965455, -4.9965281, -0.4131544, 0.4131892

Time for backsubstitution: 20.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 444
type: B, layer: 1, pos: 4574

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1745895
time: 3.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744164, upper bound: 0.1746201
time: 3.33 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.1082354, -8.0845404, -9.1082344, -8.0845356, -0.7747526, 0.7728028
1: -13.5773735, -12.4230652, -13.5773735, -12.4230661, -0.6607056, 0.6589031
2: -6.0350928, -5.1668210, -6.0350924, -5.1668196, -0.4653320, 0.4649618
3: -8.3463726, -7.4084873, -8.3463726, -7.4084868, -0.7160072, 0.7159004
4: -10.1672888, -9.1996689, -10.1672869, -9.1996670, -0.4615834, 0.4603000
5: -8.9240217, -8.0944824, -8.9240227, -8.0944834, -0.4375956, 0.4377451
6: -11.1597424, -10.2314377, -11.1597443, -10.2314358, -0.5709834, 0.5701628
7: -12.6991425, -11.8378420, -12.6991444, -11.8378391, -0.4696116, 0.4681275
8: 11.9400921, 12.5971451, 11.9400921, 12.5971460, -0.4559393, 0.4565549
9: -5.7965899, -4.9964886, -5.7965879, -4.9964890, -0.4134932, 0.4133661

Time for backsubstitution: 20.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4574
type: B, layer: 1, pos: 444

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4574

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746200, upper bound: 0.1744151
time: 4.17 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746200, upper bound: 0.1746197
time: 4.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.91 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.91
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1745895
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.91
Output dim: 8, lower bound: -0.1744164, upper bound: 0.1746201
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.91
Output dim: 8, lower bound: -0.1746200, upper bound: 0.1744151
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.91
Output dim: 8, lower bound: -0.1746200, upper bound: 0.1746197

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -9.1049967, -8.0876942, -9.1035700, -8.0859118, -0.7681894, 0.7658157
1: -13.5763531, -12.4282360, -13.5769367, -12.4292126, -0.6532192, 0.6553588
2: -6.0333481, -5.1674190, -6.0327291, -5.1671877, -0.4627814, 0.4621642
3: -8.3446865, -7.4098544, -8.3432732, -7.4093018, -0.7134805, 0.7114868
4: -10.1668110, -9.2020845, -10.1671095, -9.2018433, -0.4588675, 0.4589612
5: -8.9228249, -8.0951891, -8.9233265, -8.0957165, -0.4349461, 0.4357405
6: -11.1588917, -10.2336311, -11.1592989, -10.2331524, -0.5680609, 0.5684738
7: -12.6984138, -11.8402271, -12.6988430, -11.8398027, -0.4664974, 0.4664991
8: 11.9416409, 12.5964651, 11.9409943, 12.5964479, -0.4537430, 0.4545221
9: -5.7947431, -4.9966741, -5.7930083, -4.9966421, -0.4113233, 0.4095809

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 444

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1745015
time: 6.48 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1745890
time: 5.37 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -9.1071873, -8.0876436, -9.1084747, -8.0814552, -0.7751713, 0.7700071
1: -13.5765276, -12.4258175, -13.5831604, -12.4239740, -0.6565504, 0.6636853
2: -6.0344191, -5.1673470, -6.0351968, -5.1649547, -0.4661674, 0.4638102
3: -8.3461752, -7.4096904, -8.3463879, -7.4052830, -0.7189636, 0.7134457
4: -10.1668873, -9.2013702, -10.1693602, -9.2000284, -0.4601026, 0.4619963
5: -8.9229603, -8.0946035, -8.9246788, -8.0944414, -0.4359529, 0.4376607
6: -11.1590519, -10.2331390, -11.1604519, -10.2319736, -0.5694900, 0.5701394
7: -12.6985302, -11.8396244, -12.7006512, -11.8384285, -0.4682732, 0.4691324
8: 11.9414721, 12.5967913, 11.9400311, 12.5972481, -0.4548302, 0.4565375
9: -5.7964435, -4.9966183, -5.7969003, -4.9926496, -0.4170434, 0.4116979

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1745018
time: 5.97 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1746201
time: 4.80 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -9.1082354, -8.0845404, -9.1071882, -8.0876436, -0.7716417, 0.7735996
1: -13.5773735, -12.4230652, -13.5765257, -12.4258137, -0.6579514, 0.6598067
2: -6.0350928, -5.1668210, -6.0344195, -5.1673446, -0.4648023, 0.4646821
3: -8.3463726, -7.4084873, -8.3461752, -7.4096885, -0.7147627, 0.7158051
4: -10.1672888, -9.1996689, -10.1668844, -9.2013674, -0.4598498, 0.4611812
5: -8.9240217, -8.0944824, -8.9229603, -8.0946026, -0.4376268, 0.4365582
6: -11.1597424, -10.2314377, -11.1590519, -10.2331362, -0.5692945, 0.5702076
7: -12.6991425, -11.8378420, -12.6985264, -11.8396225, -0.4678142, 0.4690275
8: 11.9400921, 12.5971451, 11.9414721, 12.5967922, -0.4561844, 0.4551187
9: -5.7965899, -4.9964886, -5.7964439, -4.9966192, -0.4131939, 0.4131734

Time for backsubstitution: 21.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 444

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745892, upper bound: 0.1742970
time: 3.80 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746197, upper bound: 0.1744145
time: 5.36 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.1082354, -8.0845404, -9.1082354, -8.0845404, -0.7728033, 0.7728033
1: -13.5773735, -12.4230652, -13.5773735, -12.4230652, -0.6589031, 0.6589031
2: -6.0350928, -5.1668210, -6.0350928, -5.1668210, -0.4649611, 0.4649611
3: -8.3463726, -7.4084873, -8.3463726, -7.4084873, -0.7159009, 0.7159004
4: -10.1672888, -9.1996689, -10.1672888, -9.1996689, -0.4602997, 0.4602997
5: -8.9240217, -8.0944824, -8.9240217, -8.0944824, -0.4375956, 0.4375954
6: -11.1597424, -10.2314377, -11.1597424, -10.2314377, -0.5701618, 0.5701618
7: -12.6991425, -11.8378420, -12.6991425, -11.8378420, -0.4681263, 0.4681263
8: 11.9400921, 12.5971451, 11.9400921, 12.5971451, -0.4559393, 0.4559393
9: -5.7965899, -4.9964886, -5.7965899, -4.9964886, -0.4134929, 0.4134929

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 444

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745893, upper bound: 0.1742965
time: 5.66 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1744149
time: 3.31 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.59 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1745015
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1745890
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1745018
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1746201
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1745892, upper bound: 0.1742970
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1746197, upper bound: 0.1744145
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1745893, upper bound: 0.1742965
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.59
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1744149

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.1026506, -8.0877495, -9.1035700, -8.0859118, -0.7654934, 0.7651405
1: -13.5761662, -12.4308462, -13.5769367, -12.4292126, -0.6531105, 0.6527581
2: -6.0321922, -5.1674986, -6.0327291, -5.1671877, -0.4615712, 0.4619534
3: -8.3430786, -7.4100327, -8.3432732, -7.4093018, -0.7118640, 0.7113113
4: -10.1667280, -9.2028522, -10.1671095, -9.2018433, -0.4588022, 0.4581544
5: -8.9226818, -8.0958195, -8.9233265, -8.0957165, -0.4345169, 0.4351158
6: -11.1587162, -10.2341671, -11.1592989, -10.2331524, -0.5679212, 0.5679212
7: -12.6982880, -11.8408699, -12.6988430, -11.8398027, -0.4662127, 0.4656928
8: 11.9418249, 12.5961180, 11.9409943, 12.5964479, -0.4534516, 0.4539609
9: -5.7929134, -4.9967308, -5.7930083, -4.9966421, -0.4094882, 0.4095190

Time for backsubstitution: 21.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4574

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4574

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1742969
time: 3.13 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1745020
time: 3.54 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.1075544, -8.0832930, -9.1035700, -8.0859118, -0.7703505, 0.7696314
1: -13.5823898, -12.4256077, -13.5769367, -12.4292126, -0.6583300, 0.6580667
2: -6.0346603, -5.1652675, -6.0327291, -5.1671877, -0.4641290, 0.4642203
3: -8.3461914, -7.4060163, -8.3432732, -7.4093018, -0.7149892, 0.7153006
4: -10.1689806, -9.2010403, -10.1671095, -9.2018433, -0.4610896, 0.4598372
5: -8.9240341, -8.0945492, -8.9233265, -8.0957165, -0.4358602, 0.4363518
6: -11.1598740, -10.2329845, -11.1592989, -10.2331524, -0.5690422, 0.5692229
7: -12.7000952, -11.8394928, -12.6988430, -11.8398027, -0.4681082, 0.4669747
8: 11.9408627, 12.5969133, 11.9409943, 12.5964479, -0.4543300, 0.4547768
9: -5.7968006, -4.9927382, -5.7930083, -4.9966421, -0.4133980, 0.4135287

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4574

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4574

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1743843
time: 3.03 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1745895
time: 3.45 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.1026506, -8.0877495, -9.1084747, -8.0814552, -0.7699814, 0.7700005
1: -13.5761662, -12.4308462, -13.5831604, -12.4239740, -0.6584148, 0.6586623
2: -6.0321922, -5.1674986, -6.0351968, -5.1649547, -0.4638376, 0.4645119
3: -8.3430786, -7.4100327, -8.3463879, -7.4052830, -0.7158537, 0.7144384
4: -10.1667280, -9.2028522, -10.1693602, -9.2000284, -0.4604831, 0.4604409
5: -8.9226818, -8.0958195, -8.9246788, -8.0944414, -0.4357543, 0.4364591
6: -11.1587162, -10.2341671, -11.1604519, -10.2319736, -0.5692225, 0.5690413
7: -12.6982880, -11.8408699, -12.7006512, -11.8384285, -0.4674926, 0.4675889
8: 11.9418249, 12.5961180, 11.9400311, 12.5972481, -0.4542675, 0.4548397
9: -5.7929134, -4.9967308, -5.7969003, -4.9926496, -0.4134979, 0.4134340

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4574

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 4574

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1742965
time: 9.14 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1745021
time: 4.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.1075544, -8.0832930, -9.1084747, -8.0814552, -0.7704639, 0.7701149
1: -13.5823898, -12.4256077, -13.5831604, -12.4239740, -0.6583538, 0.6580033
2: -6.0346603, -5.1652675, -6.0351968, -5.1649547, -0.4635653, 0.4639473
3: -8.3461914, -7.4060163, -8.3463879, -7.4052830, -0.7149868, 0.7144375
4: -10.1689806, -9.2010403, -10.1693602, -9.2000284, -0.4612532, 0.4606063
5: -8.9240341, -8.0945492, -8.9246788, -8.0944414, -0.4360521, 0.4366512
6: -11.1598740, -10.2329845, -11.1604519, -10.2319736, -0.5703950, 0.5703959
7: -12.7000952, -11.8394928, -12.7006512, -11.8384285, -0.4689155, 0.4684007
8: 11.9408627, 12.5969133, 11.9400311, 12.5972481, -0.4561410, 0.4566500
9: -5.7968006, -4.9927382, -5.7969003, -4.9926496, -0.4118462, 0.4118812

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4574

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4574

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1744145
time: 5.53 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1742984, upper bound: 0.1746201
time: 3.95 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -9.1036949, -8.0846424, -9.1049967, -8.0876942, -0.7658205, 0.7698078
1: -13.5770121, -12.4280920, -13.5763531, -12.4282360, -0.6553311, 0.6546988
2: -6.0328646, -5.1669750, -6.0333481, -5.1674190, -0.4622726, 0.4631529
3: -8.3432713, -7.4088316, -8.3446865, -7.4098544, -0.7114882, 0.7139730
4: -10.1671295, -9.2011538, -10.1668110, -9.2020845, -0.4589756, 0.4595683
5: -8.9237442, -8.0957031, -8.9228249, -8.0951891, -0.4362230, 0.4349575
6: -11.1594076, -10.2324657, -11.1588917, -10.2336311, -0.5685124, 0.5690155
7: -12.6989040, -11.8390856, -12.6984138, -11.8402271, -0.4665277, 0.4672246
8: 11.9404449, 12.5964699, 11.9416409, 12.5964651, -0.4551010, 0.4537649
9: -5.7930479, -4.9966021, -5.7947431, -4.9966741, -0.4095840, 0.4113421

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 444

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1742987
time: 3.84 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1742987
time: 4.14 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.1086006, -8.0801849, -9.1071873, -8.0876436, -0.7700157, 0.7767906
1: -13.5832357, -12.4228535, -13.5765276, -12.4258175, -0.6630244, 0.6580310
2: -6.0353317, -5.1647420, -6.0344191, -5.1673470, -0.4639182, 0.4665384
3: -8.3463879, -7.4048138, -8.3461752, -7.4096904, -0.7134466, 0.7194557
4: -10.1693783, -9.1993399, -10.1668873, -9.2013702, -0.4620106, 0.4608030
5: -8.9250946, -8.0944271, -8.9229603, -8.0946035, -0.4381428, 0.4359651
6: -11.1605644, -10.2312851, -11.1590519, -10.2331390, -0.5701776, 0.5704441
7: -12.7007093, -11.8377113, -12.6985302, -11.8396244, -0.4691610, 0.4689999
8: 11.9394836, 12.5972672, 11.9414721, 12.5967913, -0.4571171, 0.4548516
9: -5.7969441, -4.9926085, -5.7964435, -4.9966183, -0.4117019, 0.4170620

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 444

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1743861
time: 3.51 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1743861
time: 4.15 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.1036949, -8.0846424, -9.1060448, -8.0845900, -0.7669897, 0.7690125
1: -13.5770121, -12.4280920, -13.5772009, -12.4254847, -0.6562834, 0.6537890
2: -6.0328646, -5.1669750, -6.0340190, -5.1668925, -0.4624357, 0.4634321
3: -8.3432713, -7.4088316, -8.3448830, -7.4086518, -0.7126245, 0.7140656
4: -10.1671295, -9.2011538, -10.1672115, -9.2003851, -0.4594264, 0.4586861
5: -8.9237442, -8.0957031, -8.9238911, -8.0950680, -0.4361911, 0.4359937
6: -11.1594076, -10.2324657, -11.1595821, -10.2319345, -0.5693822, 0.5689683
7: -12.6989040, -11.8390856, -12.6990280, -11.8384418, -0.4668417, 0.4663231
8: 11.9404449, 12.5964699, 11.9402618, 12.5968199, -0.4548559, 0.4545856
9: -5.7930479, -4.9966021, -5.7948823, -4.9965425, -0.4098833, 0.4116604

Time for backsubstitution: 21.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1742967
time: 6.69 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1742969
time: 3.54 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.1086006, -8.0801849, -9.1082354, -8.0845375, -0.7711782, 0.7759891
1: -13.5832357, -12.4228535, -13.5773735, -12.4230690, -0.6648502, 0.6571259
2: -6.0353317, -5.1647420, -6.0350919, -5.1668196, -0.4640794, 0.4668171
3: -8.3463879, -7.4048138, -8.3463697, -7.4084897, -0.7145824, 0.7195511
4: -10.1693783, -9.1993399, -10.1672869, -9.1996717, -0.4624608, 0.4599202
5: -8.9250946, -8.0944271, -8.9240208, -8.0944843, -0.4381108, 0.4370017
6: -11.1605644, -10.2312851, -11.1597424, -10.2314358, -0.5710468, 0.5703983
7: -12.7007093, -11.8377113, -12.6991425, -11.8378410, -0.4694724, 0.4680979
8: 11.9394836, 12.5972672, 11.9400930, 12.5971460, -0.4568701, 0.4556727
9: -5.7969441, -4.9926085, -5.7965856, -4.9964886, -0.4120009, 0.4173839

Time for backsubstitution: 21.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1743843
time: 4.31 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1744149
time: 3.92 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.88 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1742969
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1745020
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1743843
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742986, upper bound: 0.1745895
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1742965
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1745021
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742985, upper bound: 0.1744145
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1742984, upper bound: 0.1746201
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1742987
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1742987
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1743861
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745017, upper bound: 0.1743861
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1742967
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1742969
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1743843
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.88
Output dim: 8, lower bound: -0.1745018, upper bound: 0.1744149

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -9.1026506, -8.0877495, -9.1026506, -8.0877495, -0.7640009, 0.7640009
1: -13.5761662, -12.4308462, -13.5761662, -12.4308462, -0.6518345, 0.6518350
2: -6.0321922, -5.1674986, -6.0321922, -5.1674986, -0.4614134, 0.4614134
3: -8.3430786, -7.4100327, -8.3430786, -7.4100327, -0.7111125, 0.7111130
4: -10.1667280, -9.2028522, -10.1667280, -9.2028522, -0.4577670, 0.4577668
5: -8.9226818, -8.0958195, -8.9226818, -8.0958195, -0.4344132, 0.4344130
6: -11.1587162, -10.2341671, -11.1587162, -10.2341671, -0.5671859, 0.5671859
7: -12.6982880, -11.8408699, -12.6982880, -11.8408699, -0.4651384, 0.4651384
8: 11.9418249, 12.5961180, 11.9418249, 12.5961180, -0.4531050, 0.4531052
9: -5.7929134, -4.9967308, -5.7929134, -4.9967308, -0.4093733, 0.4093733

Time for backsubstitution: 21.44 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 2523
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 2879
type: A, layer: 3, pos: 2559
type: A, layer: 3, pos: 1513
type: A, layer: 3, pos: 1417
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 3123
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 902
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 437
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 2004
type: A, layer: 3, pos: 306

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1621984, upper bound: 0.1697495
time: 3.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1713451, upper bound: 0.1713439
time: 4.42 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.1026506, -8.0877495, -9.1036949, -8.0846424, -0.7671118, 0.7651453
1: -13.5761662, -12.4308462, -13.5770121, -12.4280920, -0.6545906, 0.6527305
2: -6.0321922, -5.1674986, -6.0328646, -5.1669750, -0.4619427, 0.4620614
3: -8.3430786, -7.4100327, -8.3432713, -7.4088316, -0.7123566, 0.7113123
4: -10.1667280, -9.2028522, -10.1671295, -9.2011538, -0.4595032, 0.4581687
5: -8.9226818, -8.0958195, -8.9237442, -8.0957031, -0.4345281, 0.4355984
6: -11.1587162, -10.2341671, -11.1594076, -10.2324657, -0.5688753, 0.5679598
7: -12.6982880, -11.8408699, -12.6989040, -11.8390856, -0.4669399, 0.4657218
8: 11.9418249, 12.5961180, 11.9404449, 12.5964699, -0.4534731, 0.4545400
9: -5.7929134, -4.9967308, -5.7930479, -4.9966021, -0.4095068, 0.4095218

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 2523
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 2879
type: A, layer: 3, pos: 2559
type: A, layer: 3, pos: 1513
type: A, layer: 3, pos: 1417
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 3123
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 902
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 437
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 2004
type: A, layer: 3, pos: 306

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1621984, upper bound: 0.1699546
time: 7.48 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1713451, upper bound: 0.1715487
time: 5.51 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.1075544, -8.0832930, -9.1026506, -8.0877495, -0.7688580, 0.7684927
1: -13.5823898, -12.4256077, -13.5761662, -12.4308462, -0.6577306, 0.6571441
2: -6.0346603, -5.1652675, -6.0321922, -5.1674986, -0.4639721, 0.4636807
3: -8.3461914, -7.4060163, -8.3430786, -7.4100327, -0.7142367, 0.7151012
4: -10.1689806, -9.2010403, -10.1667280, -9.2028522, -0.4600544, 0.4594498
5: -8.9240341, -8.0945492, -8.9226818, -8.0958195, -0.4357564, 0.4356489
6: -11.1598740, -10.2329845, -11.1587162, -10.2341671, -0.5683069, 0.5684876
7: -12.7000952, -11.8394928, -12.6982880, -11.8408699, -0.4670339, 0.4664204
8: 11.9408627, 12.5969133, 11.9418249, 12.5961180, -0.4539838, 0.4539211
9: -5.7968006, -4.9927382, -5.7929134, -4.9967308, -0.4132831, 0.4133830

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 2523
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: A, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 2879
type: A, layer: 3, pos: 2559
type: A, layer: 3, pos: 1513
type: A, layer: 3, pos: 1417
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 3123
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 902
type: A, layer: 3, pos: 437
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2004

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1621984, upper bound: 0.1698368
time: 3.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1713451, upper bound: 0.1714312
time: 4.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.1075544, -8.0832930, -9.1036949, -8.0846424, -0.7719688, 0.7696371
1: -13.5823898, -12.4256077, -13.5770121, -12.4280920, -0.6583538, 0.6580396
2: -6.0346603, -5.1652675, -6.0328646, -5.1669750, -0.4645009, 0.4643288
3: -8.3461914, -7.4060163, -8.3432713, -7.4088316, -0.7154818, 0.7153025
4: -10.1689806, -9.2010403, -10.1671295, -9.2011538, -0.4617906, 0.4598515
5: -8.9240341, -8.0945492, -8.9237442, -8.0957031, -0.4358714, 0.4368343
6: -11.1598740, -10.2329845, -11.1594076, -10.2324657, -0.5699964, 0.5692616
7: -12.7000952, -11.8394928, -12.6989040, -11.8390856, -0.4688354, 0.4670036
8: 11.9408627, 12.5969133, 11.9404449, 12.5964699, -0.4543519, 0.4553559
9: -5.7968006, -4.9927382, -5.7930479, -4.9966021, -0.4134166, 0.4135318

Time for backsubstitution: 21.92 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.60 + 563.44 = 619.03 seconds
