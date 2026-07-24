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
execution time: IAR + RelationalAnalysis = 21.25 + 33.44 = 54.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1746203, upper bound: 0.1746206

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4574
type: B, layer: 1, pos: 4574
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 4574

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744166, upper bound: 0.1746202
time: 5.06 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746199, upper bound: 0.1746202
time: 10.49 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 15.72 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 15.72
Output dim: 8, lower bound: -0.1744166, upper bound: 0.1746202
NS_A2, status: Status.UNKNOWN, split count: 1, time: 15.72
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

Time for backsubstitution: 19.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4574
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4574

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744148, upper bound: 0.1744150
time: 3.02 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744148, upper bound: 0.1746202
time: 3.04 seconds

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

Time for backsubstitution: 20.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4574
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444

Time for candidate selection: 0.15 seconds

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
time: 4.45 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.21 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 8, lower bound: -0.1744148, upper bound: 0.1744150
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 8, lower bound: -0.1744148, upper bound: 0.1746202
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 8, lower bound: -0.1746200, upper bound: 0.1744151
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 29.21
Output dim: 8, lower bound: -0.1746200, upper bound: 0.1746197

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -9.1071882, -8.0876436, -9.1071882, -8.0876436, -0.7704926, 0.7704926
1: -13.5765257, -12.4258137, -13.5765257, -12.4258137, -0.6570549, 0.6570544
2: -6.0344195, -5.1673446, -6.0344195, -5.1673446, -0.4641528, 0.4641528
3: -8.3461752, -7.4096885, -8.3461752, -7.4096885, -0.7145615, 0.7145615
4: -10.1668844, -9.2013674, -10.1668844, -9.2013674, -0.4594479, 0.4594481
5: -8.9229603, -8.0946026, -8.9229603, -8.0946026, -0.4364414, 0.4364414
6: -11.1590519, -10.2331362, -11.1590519, -10.2331362, -0.5685191, 0.5685196
7: -12.6985264, -11.8396225, -12.6985264, -11.8396225, -0.4672306, 0.4672306
8: 11.9414721, 12.5967922, 11.9414721, 12.5967922, -0.4547482, 0.4547482
9: -5.7964439, -4.9966192, -5.7964439, -4.9966192, -0.4130387, 0.4130390

Time for backsubstitution: 19.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1742970
time: 3.24 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744165, upper bound: 0.1744145
time: 7.74 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -9.1071882, -8.0876436, -9.1082354, -8.0845404, -0.7735996, 0.7716417
1: -13.5765257, -12.4258137, -13.5773735, -12.4230652, -0.6598067, 0.6579518
2: -6.0344195, -5.1673446, -6.0350928, -5.1668210, -0.4646821, 0.4648023
3: -8.3461752, -7.4096885, -8.3463726, -7.4084873, -0.7158051, 0.7147636
4: -10.1668844, -9.2013674, -10.1672888, -9.1996689, -0.4611814, 0.4598498
5: -8.9229603, -8.0946026, -8.9240217, -8.0944824, -0.4365582, 0.4376268
6: -11.1590519, -10.2331362, -11.1597424, -10.2314377, -0.5702076, 0.5692945
7: -12.6985264, -11.8396225, -12.6991425, -11.8378420, -0.4690275, 0.4678142
8: 11.9414721, 12.5967922, 11.9400921, 12.5971451, -0.4551187, 0.4561841
9: -5.7964439, -4.9966192, -5.7965899, -4.9964886, -0.4131734, 0.4131937

Time for backsubstitution: 20.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1745017
time: 5.54 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1744165, upper bound: 0.1746196
time: 4.90 seconds

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

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 444
type: A, layer: 1, pos: 444

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 444

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745019, upper bound: 0.1743837
time: 5.21 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746197, upper bound: 0.1744144
time: 4.68 seconds

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

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 444
type: B, layer: 1, pos: 444

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 444

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1745893, upper bound: 0.1742965
time: 5.68 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1744149
time: 3.36 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.64 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1742970
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1744165, upper bound: 0.1744145
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1743859, upper bound: 0.1745017
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1744165, upper bound: 0.1746196
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1745019, upper bound: 0.1743837
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1746197, upper bound: 0.1744144
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1745893, upper bound: 0.1742965
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.64
Output dim: 8, lower bound: -0.1746198, upper bound: 0.1744149

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -9.1026506, -8.0877495, -9.1049967, -8.0876942, -0.7646761, 0.7666979
1: -13.5761662, -12.4308462, -13.5763531, -12.4282360, -0.6544356, 0.6519432
2: -6.0321922, -5.1674986, -6.0333481, -5.1674190, -0.4616246, 0.4626236
3: -8.3430786, -7.4100327, -8.3446865, -7.4098544, -0.7112880, 0.7127285
4: -10.1667280, -9.2028522, -10.1668110, -9.2020845, -0.4585736, 0.4578321
5: -8.9226818, -8.0958195, -8.9228249, -8.0951891, -0.4350381, 0.4348421
6: -11.1587162, -10.2341671, -11.1588917, -10.2336311, -0.5677385, 0.5673256
7: -12.6982880, -11.8408699, -12.6984138, -11.8402271, -0.4659445, 0.4654233
8: 11.9418249, 12.5961180, 11.9416409, 12.5964651, -0.4536667, 0.4533968
9: -5.7929134, -4.9967308, -5.7947431, -4.9966741, -0.4094353, 0.4112084

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2523
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 2811
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 3123
type: B, layer: 3, pos: 3123
type: B, layer: 3, pos: 703
type: A, layer: 3, pos: 703
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 2559
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 902
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 437
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 2004

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2585

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1698383, upper bound: 0.1621988
time: 4.39 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1714327, upper bound: 0.1713453
time: 3.19 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -9.1075544, -8.0832930, -9.1071873, -8.0876436, -0.7688675, 0.7736816
1: -13.5823898, -12.4256077, -13.5765276, -12.4258175, -0.6627541, 0.6552777
2: -6.0346603, -5.1652675, -6.0344191, -5.1673470, -0.4632707, 0.4660110
3: -8.3461914, -7.4060163, -8.3461752, -7.4096904, -0.7132444, 0.7182112
4: -10.1689806, -9.2010403, -10.1668873, -9.2013702, -0.4616096, 0.4590683
5: -8.9240341, -8.0945492, -8.9229603, -8.0946035, -0.4369578, 0.4358492
6: -11.1598740, -10.2329845, -11.1590519, -10.2331390, -0.5694036, 0.5687551
7: -12.7000952, -11.8394928, -12.6985302, -11.8396244, -0.4685779, 0.4672031
8: 11.9408627, 12.5969133, 11.9414721, 12.5967913, -0.4556808, 0.4544830
9: -5.7968006, -4.9927382, -5.7964435, -4.9966183, -0.4115477, 0.4169285

Time for backsubstitution: 21.54 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2523
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: B, layer: 3, pos: 1682
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 3123
type: B, layer: 3, pos: 3123
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 1417
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 2004

Time for candidate selection: 0.33 seconds

### Candidate
type: B, layer: 3, pos: 2585

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1698685, upper bound: 0.1623159
time: 3.50 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1714633, upper bound: 0.1714631
time: 4.48 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.1026506, -8.0877495, -9.1060448, -8.0845900, -0.7677851, 0.7678471
1: -13.5761662, -12.4308462, -13.5772009, -12.4254847, -0.6571894, 0.6528397
2: -6.0321922, -5.1674986, -6.0340190, -5.1668925, -0.4621544, 0.4632726
3: -8.3430786, -7.4100327, -8.3448830, -7.4086518, -0.7125311, 0.7129288
4: -10.1667280, -9.2028522, -10.1672115, -9.2003851, -0.4603081, 0.4582338
5: -8.9226818, -8.0958195, -8.9238911, -8.0950680, -0.4351542, 0.4360275
6: -11.1587162, -10.2341671, -11.1595821, -10.2319345, -0.5694270, 0.5681000
7: -12.6982880, -11.8408699, -12.6990280, -11.8384418, -0.4677439, 0.4660065
8: 11.9418249, 12.5961180, 11.9402618, 12.5968199, -0.4540348, 0.4548318
9: -5.7929134, -4.9967308, -5.7948823, -4.9965425, -0.4095695, 0.4113605

Time for backsubstitution: 21.65 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2523
type: B, layer: 3, pos: 2523
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: B, layer: 3, pos: 179
type: B, layer: 3, pos: 2879
type: A, layer: 3, pos: 2879
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 2811
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 1513
type: B, layer: 3, pos: 3123
type: A, layer: 3, pos: 3123
type: B, layer: 3, pos: 703
type: A, layer: 3, pos: 703
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 2004
type: B, layer: 3, pos: 2004

Time for candidate selection: 0.32 seconds

### Candidate
type: B, layer: 3, pos: 2585

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1698383, upper bound: 0.1624020
time: 4.55 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1714327, upper bound: 0.1715481
time: 4.45 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.1075544, -8.0832930, -9.1082354, -8.0845375, -0.7719746, 0.7748299
1: -13.5823898, -12.4256077, -13.5773735, -12.4230690, -0.6633801, 0.6561747
2: -6.0346603, -5.1652675, -6.0350919, -5.1668196, -0.4637995, 0.4666612
3: -8.3461914, -7.4060163, -8.3463697, -7.4084897, -0.7144866, 0.7184143
4: -10.1689806, -9.2010403, -10.1672869, -9.1996717, -0.4633427, 0.4594693
5: -8.9240341, -8.0945492, -8.9240208, -8.0944843, -0.4370742, 0.4370346
6: -11.1598740, -10.2329845, -11.1597424, -10.2314358, -0.5710921, 0.5695305
7: -12.7000952, -11.8394928, -12.6991425, -11.8378410, -0.4703739, 0.4677866
8: 11.9408627, 12.5969133, 11.9400930, 12.5971460, -0.4560494, 0.4559186
9: -5.7968006, -4.9927382, -5.7965856, -4.9964886, -0.4116826, 0.4170837

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2523
type: B, layer: 3, pos: 2523
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 2811
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 3123
type: B, layer: 3, pos: 3123
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 902
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 2004

Time for candidate selection: 0.43 seconds

### Candidate
type: B, layer: 3, pos: 2585

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1698685, upper bound: 0.1625191
time: 3.71 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1714633, upper bound: 0.1716664
time: 3.52 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -9.1060448, -8.0845900, -9.1026506, -8.0877495, -0.7678471, 0.7677851
1: -13.5772009, -12.4254847, -13.5761662, -12.4308462, -0.6528397, 0.6571894
2: -6.0340190, -5.1668925, -6.0321922, -5.1674986, -0.4632726, 0.4621544
3: -8.3448830, -7.4086518, -8.3430786, -7.4100327, -0.7129283, 0.7125311
4: -10.1672115, -9.2003851, -10.1667280, -9.2028522, -0.4582338, 0.4603083
5: -8.9238911, -8.0950680, -8.9226818, -8.0958195, -0.4360278, 0.4351540
6: -11.1595821, -10.2319345, -11.1587162, -10.2341671, -0.5681000, 0.5694265
7: -12.6990280, -11.8384418, -12.6982880, -11.8408699, -0.4660068, 0.4677439
8: 11.9402618, 12.5968199, 11.9418249, 12.5961180, -0.4548321, 0.4540350
9: -5.7948823, -4.9965425, -5.7929134, -4.9967308, -0.4113605, 0.4095693

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 1991
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: B, layer: 3, pos: 1264
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 2811
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 1513
type: A, layer: 3, pos: 3123
type: B, layer: 3, pos: 3123
type: A, layer: 3, pos: 703
type: B, layer: 3, pos: 703
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 2004

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1624016, upper bound: 0.1698386
time: 3.57 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1715483, upper bound: 0.1714330
time: 3.63 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -9.1082354, -8.0845375, -9.1075544, -8.0832930, -0.7748299, 0.7719746
1: -13.5773735, -12.4230690, -13.5823898, -12.4256077, -0.6561747, 0.6633801
2: -6.0350919, -5.1668196, -6.0346603, -5.1652675, -0.4666610, 0.4637997
3: -8.3463697, -7.4084897, -8.3461914, -7.4060163, -0.7184148, 0.7144871
4: -10.1672869, -9.1996717, -10.1689806, -9.2010403, -0.4594693, 0.4633427
5: -8.9240208, -8.0944843, -8.9240341, -8.0945492, -0.4370344, 0.4370747
6: -11.1597424, -10.2314358, -11.1598740, -10.2329845, -0.5695305, 0.5710921
7: -12.6991425, -11.8378410, -12.7000952, -11.8394928, -0.4677868, 0.4703741
8: 11.9400930, 12.5971460, 11.9408627, 12.5969133, -0.4559188, 0.4560494
9: -5.7965856, -4.9964886, -5.7968006, -4.9927382, -0.4170837, 0.4116828

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 1991
type: B, layer: 3, pos: 1991
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: B, layer: 3, pos: 1264
type: A, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: B, layer: 3, pos: 179
type: B, layer: 3, pos: 2879
type: A, layer: 3, pos: 2879
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 1682
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 2811
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 3123
type: A, layer: 3, pos: 3123
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 2377
type: B, layer: 3, pos: 2832
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 902
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: A, layer: 3, pos: 2004
type: B, layer: 3, pos: 2004

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1625189, upper bound: 0.1698688
time: 4.51 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1716662, upper bound: 0.1714636
time: 3.25 seconds

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

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2523
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 2811
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 3123
type: B, layer: 3, pos: 3123
type: B, layer: 3, pos: 703
type: A, layer: 3, pos: 703
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 2559
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 902
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 437
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 2004

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2585

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1700417, upper bound: 0.1621970
time: 4.63 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1716361, upper bound: 0.1713435
time: 3.58 seconds

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

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2585
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 2523
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 179
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: B, layer: 3, pos: 1682
type: B, layer: 3, pos: 2811
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 3123
type: B, layer: 3, pos: 3123
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 2377
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 1417
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: B, layer: 3, pos: 208
type: A, layer: 3, pos: 208
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 2004

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2585

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1700717, upper bound: 0.1623142
time: 3.56 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1716667, upper bound: 0.1714613
time: 3.40 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.75 seconds
NS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1698383, upper bound: 0.1621988
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1714327, upper bound: 0.1713453
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1698685, upper bound: 0.1623159
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1714633, upper bound: 0.1714631
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1698383, upper bound: 0.1624020
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1714327, upper bound: 0.1715481
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1698685, upper bound: 0.1625191
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1714633, upper bound: 0.1716664
NS_A2_B1_B1_A1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1624016, upper bound: 0.1698386
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1715483, upper bound: 0.1714330
NS_A2_B1_B2_A1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1625189, upper bound: 0.1698688
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1716662, upper bound: 0.1714636
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1700417, upper bound: 0.1621970
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1716361, upper bound: 0.1713435
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1700717, upper bound: 0.1623142
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.75
Output dim: 8, lower bound: -0.1716667, upper bound: 0.1714613

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -9.1011906, -8.0877943, -9.1008949, -8.0878220, -0.7400999, 0.7382679
1: -13.5760794, -12.4320793, -13.5761375, -12.4315472, -0.5773787, 0.6511931
2: -6.0310292, -5.1675854, -6.0300283, -5.1676507, -0.4600883, 0.4414692
3: -8.3430786, -7.4135032, -8.3446865, -7.4197216, -0.5338001, 0.7033272
4: -10.1664524, -9.2036810, -10.1660318, -9.2044430, -0.4543085, 0.4548957
5: -8.9226770, -8.0983496, -8.9228134, -8.1021986, -0.3367616, 0.4340272
6: -11.1581364, -10.2342377, -11.1572809, -10.2338238, -0.5660567, 0.5278010
7: -12.6972523, -11.8410006, -12.6954594, -11.8405552, -0.4644420, 0.4576349
8: 11.9438076, 12.5960712, 11.9472027, 12.5963316, -0.4423571, 0.4001007
9: -5.7927036, -4.9975805, -5.7941837, -4.9990954, -0.4060171, 0.4162989

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1682
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 3123
type: A, layer: 3, pos: 3123
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 1417
type: B, layer: 3, pos: 703
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 2004
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 208
type: B, layer: 3, pos: 2811
type: B, layer: 3, pos: 2377

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1619265, upper bound: 0.1697514
time: 3.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1619265, upper bound: 0.1697514
time: 3.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.1060810, -8.0833368, -9.1030731, -8.0877695, -0.7442980, 0.7451735
1: -13.5823069, -12.4268351, -13.5763121, -12.4291306, -0.5857577, 0.6545310
2: -6.0334954, -5.1653571, -6.0311007, -5.1675758, -0.4617338, 0.4448600
3: -8.3461914, -7.4094858, -8.3461752, -7.4195571, -0.5357237, 0.7088137
4: -10.1687088, -9.2018642, -10.1661053, -9.2037296, -0.4573712, 0.4561315
5: -8.9240265, -8.0970774, -8.9229469, -8.1016150, -0.3386794, 0.4350357
6: -11.1592941, -10.2330589, -11.1574392, -10.2333298, -0.5677247, 0.5292301
7: -12.6990604, -11.8396244, -12.6955738, -11.8399515, -0.4670737, 0.4594176
8: 11.9428482, 12.5968676, 11.9470367, 12.5966587, -0.4443727, 0.4011617
9: -5.7965899, -4.9935894, -5.7958851, -4.9990425, -0.4081304, 0.4220173

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 2523
type: A, layer: 3, pos: 1508
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: A, layer: 3, pos: 2377
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 2832
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 703
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 1513
type: B, layer: 3, pos: 3123
type: A, layer: 3, pos: 3123
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 703
type: B, layer: 3, pos: 1417
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 306
type: B, layer: 3, pos: 208
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 2004
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 2811
type: B, layer: 3, pos: 2377

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1619565, upper bound: 0.1698688
time: 6.21 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1619565, upper bound: 0.1698688
time: 3.67 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.1011906, -8.0877943, -9.1019411, -8.0847149, -0.7432089, 0.7394028
1: -13.5760794, -12.4320793, -13.5769815, -12.4287939, -0.5801344, 0.6520886
2: -6.0310292, -5.1675854, -6.0307021, -5.1671247, -0.4606171, 0.4421179
3: -8.3430786, -7.4135032, -8.3448830, -7.4185185, -0.5350475, 0.7035275
4: -10.1664524, -9.2036810, -10.1664314, -9.2027435, -0.4560466, 0.4552965
5: -8.9226770, -8.0983496, -8.9238758, -8.1020832, -0.3368744, 0.4352131
6: -11.1581364, -10.2342377, -11.1579704, -10.2321224, -0.5677466, 0.5285721
7: -12.6972523, -11.8410006, -12.6960735, -11.8387718, -0.4662414, 0.4582195
8: 11.9438076, 12.5960712, 11.9458237, 12.5966854, -0.4427261, 0.4015281
9: -5.7927036, -4.9975805, -5.7943239, -4.9989653, -0.4061501, 0.4164486

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2585
type: B, layer: 3, pos: 1991
type: A, layer: 3, pos: 1991
type: A, layer: 3, pos: 1844
type: B, layer: 3, pos: 1844
type: B, layer: 3, pos: 1698
type: A, layer: 3, pos: 1698
type: B, layer: 3, pos: 2523
type: A, layer: 3, pos: 2523
type: B, layer: 3, pos: 1508
type: A, layer: 3, pos: 1508
type: A, layer: 3, pos: 1264
type: B, layer: 3, pos: 1264
type: A, layer: 3, pos: 179
type: B, layer: 3, pos: 179
type: A, layer: 3, pos: 2811
type: A, layer: 3, pos: 2879
type: B, layer: 3, pos: 2879
type: A, layer: 3, pos: 2377
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 2832
type: B, layer: 3, pos: 1682
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 703
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 3123
type: B, layer: 3, pos: 1513
type: A, layer: 3, pos: 1513
type: A, layer: 3, pos: 3123
type: A, layer: 3, pos: 1417
type: B, layer: 3, pos: 1417
type: B, layer: 3, pos: 703
type: A, layer: 3, pos: 2559
type: B, layer: 3, pos: 2559
type: A, layer: 3, pos: 208
type: B, layer: 3, pos: 902
type: A, layer: 3, pos: 902
type: B, layer: 3, pos: 437
type: A, layer: 3, pos: 437
type: B, layer: 3, pos: 306
type: A, layer: 3, pos: 2004
type: B, layer: 3, pos: 2004
type: A, layer: 3, pos: 306
type: B, layer: 3, pos: 2832
type: B, layer: 3, pos: 208
type: B, layer: 3, pos: 2811
type: B, layer: 3, pos: 2377

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2585

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1619265, upper bound: 0.1699547
time: 5.65 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1619265, upper bound: 0.1699547
time: 3.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.1060810, -8.0833368, -9.1041212, -8.0846634, -0.7474079, 0.7463112
1: -13.5823069, -12.4268351, -13.5771599, -12.4263811, -0.5863833, 0.6554275
2: -6.0334954, -5.1653571, -6.0317736, -5.1670527, -0.4622641, 0.4455109
3: -8.3461914, -7.4094858, -8.3463697, -7.4183578, -0.5369701, 0.7090139
4: -10.1687088, -9.2018642, -10.1665087, -9.2020292, -0.4591067, 0.4565310
5: -8.9240265, -8.0970774, -8.9240065, -8.1014938, -0.3387930, 0.4362202
6: -11.1592941, -10.2330589, -11.1581306, -10.2316284, -0.5694137, 0.5300016
7: -12.6990604, -11.8396244, -12.6961908, -11.8381701, -0.4688702, 0.4600019
8: 11.9428482, 12.5968676, 11.9456549, 12.5970116, -0.4447422, 0.4025893
9: -5.7965899, -4.9935894, -5.7960281, -4.9989114, -0.4082644, 0.4221683

Time for backsubstitution: 21.94 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 54.69 + 547.97 = 602.66 seconds
