## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.263905785


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1160488, -4.3704209, -6.1160488, -4.3704209, -1.1343257, 1.1343260)
1: (-6.7619667, -5.4906173, -6.7619667, -5.4906173, -0.9096284, 0.9096284)
2: (-0.4435998, 0.8881155, -0.4435998, 0.8881155, -0.7806470, 0.7806470)
3: (-2.9973392, -1.8161306, -2.9973392, -1.8161306, -0.6144085, 0.6144085)
4: (-9.0586786, -7.8379421, -9.0586786, -7.8379421, -0.8248234, 0.8248234)
5: (-8.8680315, -7.5075006, -8.8680315, -7.5075006, -0.5613703, 0.5613702)
6: (-10.9432974, -9.3195362, -10.9432974, -9.3195362, -0.7603209, 0.7603209)
7: (3.2328248, 4.1301155, 3.2328248, 4.1301155, -0.6888497, 0.6888497)
8: (-4.0541353, -2.7991476, -4.0541353, -2.7991476, -0.6941956, 0.6941956)
9: (-3.4196138, -2.1753459, -3.4196138, -2.1753459, -0.9527485, 0.9527488)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.67 + 35.09 = 57.76 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.2665715, upper bound: 0.2665707

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6135

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643296, upper bound: 0.2665698
time: 3.35 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665690, upper bound: 0.2665699
time: 3.48 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.06 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.06
Output dim: 7, lower bound: -0.2643296, upper bound: 0.2665698
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.06
Output dim: 7, lower bound: -0.2665690, upper bound: 0.2665699

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.1124649, -4.3745470, -6.1147275, -4.3717451, -1.1278043, 1.1277854
1: -6.7347665, -5.5038552, -6.7461548, -5.4914222, -0.8816285, 0.8772826
2: -0.4398823, 0.8824776, -0.4426168, 0.8848209, -0.7782929, 0.7788055
3: -2.9795551, -1.8252256, -2.9874241, -1.8167076, -0.5936635, 0.5936629
4: -9.0401068, -7.8727212, -9.0569534, -7.8578568, -0.7746680, 0.7760465
5: -8.8537626, -7.5333920, -8.8671370, -7.5224104, -0.5302874, 0.5339344
6: -10.9346380, -9.3371983, -10.9430151, -9.3296490, -0.7383478, 0.7430892
7: 3.2470202, 4.1229382, 3.2403572, 4.1292014, -0.6719079, 0.6705356
8: -4.0493174, -2.8040218, -4.0527897, -2.8019123, -0.6723533, 0.6740795
9: -3.4100895, -2.1858225, -3.4177771, -2.1819963, -0.9335263, 0.9385146

Time for backsubstitution: 21.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2643283
time: 3.59 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2665698
time: 3.51 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.1160474, -4.3704243, -6.1160488, -4.3704224, -1.1407061, 1.1306837
1: -6.7619572, -5.4906197, -6.7619610, -5.4906173, -0.8832939, 0.9051821
2: -0.4435993, 0.8881130, -0.4435997, 0.8881146, -0.7804878, 0.7759268
3: -2.9973316, -1.8161314, -2.9973333, -1.8161321, -0.5964303, 0.6144042
4: -9.0586767, -7.8379598, -9.0586777, -7.8379507, -0.8200490, 0.7914462
5: -8.8680325, -7.5075116, -8.8680325, -7.5075068, -0.5560055, 0.5302438
6: -10.9432945, -9.3195448, -10.9432964, -9.3195410, -0.7562803, 0.7433980
7: 3.2328300, 4.1301150, 3.2328279, 4.1301150, -0.6770077, 0.6873443
8: -4.0541344, -2.7991509, -4.0541353, -2.7991509, -0.6927762, 0.6940451
9: -3.4196134, -2.1753507, -3.4196134, -2.1753492, -0.9527462, 0.9490570

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6135
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665687, upper bound: 0.2643283
time: 3.45 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665691, upper bound: 0.2665699
time: 3.43 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.43 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 28.43
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2643283
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 28.43
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2665698
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 28.43
Output dim: 7, lower bound: -0.2665687, upper bound: 0.2643283
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 28.43
Output dim: 7, lower bound: -0.2665691, upper bound: 0.2665699

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.1124649, -4.3745470, -6.1124649, -4.3745470, -1.1230001, 1.1230004
1: -6.7347665, -5.5038552, -6.7347665, -5.5038552, -0.8670626, 0.8670623
2: -0.4398823, 0.8824776, -0.4398823, 0.8824776, -0.7759655, 0.7759655
3: -2.9795551, -1.8252256, -2.9795551, -1.8252256, -0.5851529, 0.5851529
4: -9.0401068, -7.8727212, -9.0401068, -7.8727212, -0.7592714, 0.7592714
5: -8.8537626, -7.5333920, -8.8537626, -7.5333920, -0.5199096, 0.5199096
6: -10.9346380, -9.3371983, -10.9346380, -9.3371983, -0.7326837, 0.7326837
7: 3.2470202, 4.1229382, 3.2470202, 4.1229382, -0.6643677, 0.6643677
8: -4.0493174, -2.8040218, -4.0493174, -2.8040218, -0.6702892, 0.6702893
9: -3.4100895, -2.1858225, -3.4100895, -2.1858225, -0.9304855, 0.9304855

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642892, upper bound: 0.2641714
time: 5.49 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643289, upper bound: 0.2643275
time: 3.41 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.1124649, -4.3745470, -6.1158543, -4.3704453, -1.1270802, 1.1257825
1: -6.7347665, -5.5038552, -6.7616320, -5.4906511, -0.8783793, 0.8825257
2: -0.4398823, 0.8824776, -0.4434599, 0.8874973, -0.7808805, 0.7797716
3: -2.9795551, -1.8252256, -2.9970329, -1.8161564, -0.5939929, 0.6004374
4: -9.0401068, -7.8727212, -9.0584888, -7.8379841, -0.7780247, 0.7715220
5: -8.8537626, -7.5333920, -8.8677197, -7.5075359, -0.5323037, 0.5283329
6: -10.9346380, -9.3371983, -10.9431963, -9.3195457, -0.7423910, 0.7391212
7: 3.2470202, 4.1229382, 3.2329042, 4.1300812, -0.6714015, 0.6769214
8: -4.0493174, -2.8040218, -4.0540771, -2.7996769, -0.6743753, 0.6748657
9: -3.4100895, -2.1858225, -3.4196038, -2.1765537, -0.9389350, 0.9386156

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642896, upper bound: 0.2664148
time: 3.43 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643293, upper bound: 0.2665691
time: 3.61 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.1160474, -4.3704243, -6.1124649, -4.3745470, -1.1250563, 1.1270974
1: -6.7619572, -5.4906197, -6.7347665, -5.5038552, -0.8829396, 0.8784201
2: -0.4435993, 0.8881130, -0.4398823, 0.8824776, -0.7798965, 0.7765162
3: -2.9973316, -1.8161314, -2.9795551, -1.8252256, -0.6029370, 0.5940125
4: -9.0586767, -7.8379598, -9.0401068, -7.8727212, -0.7847505, 0.7780411
5: -8.8680325, -7.5075116, -8.8537626, -7.5333920, -0.5304101, 0.5323249
6: -10.9432945, -9.3195448, -10.9346380, -9.3371983, -0.7387254, 0.7423913
7: 3.2328300, 4.1301150, 3.2470202, 4.1229382, -0.6788507, 0.6714373
8: -4.0541344, -2.7991509, -4.0493174, -2.8040218, -0.6749104, 0.6881468
9: -3.4196134, -2.1753507, -3.4100895, -2.1858225, -0.9386208, 0.9436221

Time for backsubstitution: 22.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665284, upper bound: 0.2641733
time: 3.33 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665679, upper bound: 0.2643276
time: 3.52 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.1160474, -4.3704243, -6.1160474, -4.3704243, -1.1407042, 1.1407042
1: -6.7619572, -5.4906197, -6.7619572, -5.4906197, -0.8832927, 0.8832924
2: -0.4435993, 0.8881130, -0.4435993, 0.8881130, -0.7759264, 0.7759264
3: -2.9973316, -1.8161314, -2.9973316, -1.8161314, -0.5964302, 0.5964301
4: -9.0586767, -7.8379598, -9.0586767, -7.8379598, -0.7914462, 0.7914462
5: -8.8680325, -7.5075116, -8.8680325, -7.5075116, -0.5302436, 0.5302436
6: -10.9432945, -9.3195448, -10.9432945, -9.3195448, -0.7433965, 0.7433968
7: 3.2328300, 4.1301150, 3.2328300, 4.1301150, -0.6770067, 0.6770067
8: -4.0541344, -2.7991509, -4.0541344, -2.7991509, -0.6940441, 0.6940441
9: -3.4196134, -2.1753507, -3.4196134, -2.1753507, -0.9490571, 0.9490570

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665289, upper bound: 0.2641733
time: 3.32 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665684, upper bound: 0.2643276
time: 3.42 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.11 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2642892, upper bound: 0.2641714
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2643289, upper bound: 0.2643275
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2642896, upper bound: 0.2664148
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2643293, upper bound: 0.2665691
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2665284, upper bound: 0.2641733
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2665679, upper bound: 0.2643276
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2665289, upper bound: 0.2641733
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.11
Output dim: 7, lower bound: -0.2665684, upper bound: 0.2643276

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.1120853, -4.3801355, -6.1123905, -4.3756781, -1.1213942, 1.1171927
1: -6.7344117, -5.5044546, -6.7346931, -5.5039773, -0.8661368, 0.8657660
2: -0.4390981, 0.8787858, -0.4397250, 0.8817317, -0.7731998, 0.7712321
3: -2.9789286, -1.8290541, -2.9794288, -1.8260009, -0.5831852, 0.5809985
4: -9.0375748, -7.8728867, -9.0395937, -7.8727551, -0.7563684, 0.7583454
5: -8.8536501, -7.5341215, -8.8537407, -7.5335398, -0.5193782, 0.5187863
6: -10.9345188, -9.3418531, -10.9346151, -9.3381405, -0.7316349, 0.7280922
7: 3.2480803, 4.1221790, 3.2472355, 4.1227846, -0.6622074, 0.6624858
8: -4.0478287, -2.8050623, -4.0490179, -2.8042326, -0.6684999, 0.6688607
9: -3.4094234, -2.1859660, -3.4099536, -2.1858530, -0.9286480, 0.9290977

Time for backsubstitution: 21.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641707
time: 3.61 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641725
time: 3.63 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.1419420, -4.3729477, -6.1124668, -4.3745584, -1.1302710, 1.1243505
1: -6.7393022, -5.5024004, -6.7347651, -5.5038586, -0.8707378, 0.8712261
2: -0.4587880, 0.8829716, -0.4398813, 0.8824706, -0.7824551, 0.7803507
3: -2.9975815, -1.8229480, -2.9795544, -1.8252349, -0.5961010, 0.5863820
4: -9.0438995, -7.8578291, -9.0401001, -7.8727236, -0.7605274, 0.7704802
5: -8.8562002, -7.5316782, -8.8537626, -7.5333943, -0.5232133, 0.5207051
6: -10.9599323, -9.3360462, -10.9346380, -9.3372126, -0.7396268, 0.7331972
7: 3.2456298, 4.1276102, 3.2470238, 4.1229353, -0.6646614, 0.6727653
8: -4.0543966, -2.8026772, -4.0493145, -2.8040266, -0.6751366, 0.6712402
9: -3.4143806, -2.1839986, -3.4100871, -2.1858244, -0.9323518, 0.9367173

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
time: 3.47 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
time: 3.48 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.1120853, -4.3801355, -6.1157742, -4.3715739, -1.1254768, 1.1199727
1: -6.7344117, -5.5044546, -6.7615604, -5.4907727, -0.8774295, 0.8812034
2: -0.4390981, 0.8787858, -0.4433006, 0.8867483, -0.7781146, 0.7750387
3: -2.9789286, -1.8290541, -2.9969094, -1.8169312, -0.5920252, 0.5962785
4: -9.0375748, -7.8728867, -9.0579767, -7.8380156, -0.7751310, 0.7705388
5: -8.8536501, -7.5341215, -8.8676958, -7.5076818, -0.5317634, 0.5272025
6: -10.9345188, -9.3418531, -10.9431705, -9.3204851, -0.7412143, 0.7345346
7: 3.2480803, 4.1221790, 3.2331152, 4.1299281, -0.6692410, 0.6750031
8: -4.0478287, -2.8050623, -4.0537763, -2.7998857, -0.6725873, 0.6734346
9: -3.4094234, -2.1859660, -3.4194670, -2.1765828, -0.9370608, 0.9372259

Time for backsubstitution: 22.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641718, upper bound: 0.2664113
time: 8.03 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641718, upper bound: 0.2664127
time: 5.74 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.1419420, -4.3729477, -6.1158524, -4.3704567, -1.1330400, 1.1271324
1: -6.7393022, -5.5024004, -6.7616310, -5.4906530, -0.8801467, 0.8837483
2: -0.4587880, 0.8829716, -0.4434575, 0.8874890, -0.7839022, 0.7841563
3: -2.9975815, -1.8229480, -2.9970331, -1.8161659, -0.5973213, 0.6016543
4: -9.0438995, -7.8578291, -9.0584831, -7.8379827, -0.7793887, 0.7725346
5: -8.8562002, -7.5316782, -8.8677197, -7.5075369, -0.5329286, 0.5290866
6: -10.9599323, -9.3360462, -10.9431973, -9.3195591, -0.7438527, 0.7397199
7: 3.2456298, 4.1276102, 3.2329075, 4.1300793, -0.6716948, 0.6782777
8: -4.0543966, -2.8026772, -4.0540733, -2.7996807, -0.6788325, 0.6758163
9: -3.4143806, -2.1839986, -3.4196024, -2.1765552, -0.9398186, 0.9418690

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665292
time: 3.60 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665291
time: 3.80 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.1156607, -4.3760061, -6.1123905, -4.3756781, -1.1234188, 1.1213031
1: -6.7616014, -5.4912167, -6.7346931, -5.5039773, -0.8819921, 0.8770969
2: -0.4428141, 0.8844187, -0.4397250, 0.8817317, -0.7771335, 0.7717786
3: -2.9967186, -1.8199615, -2.9794288, -1.8260009, -0.6008035, 0.5898564
4: -9.0561619, -7.8381205, -9.0395937, -7.8727551, -0.7818522, 0.7770550
5: -8.8679161, -7.5082326, -8.8537407, -7.5335398, -0.5298605, 0.5311979
6: -10.9431763, -9.3241940, -10.9346151, -9.3381405, -0.7375512, 0.7378052
7: 3.2338686, 4.1293550, 3.2472355, 4.1227846, -0.6766658, 0.6695535
8: -4.0526433, -2.8001852, -4.0490179, -2.8042326, -0.6731100, 0.6867303
9: -3.4189343, -2.1754947, -3.4099536, -2.1858530, -0.9367776, 0.9422214

Time for backsubstitution: 22.87 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641714
time: 5.92 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641724
time: 5.56 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.1455116, -4.3688011, -6.1124668, -4.3745584, -1.1319628, 1.1284411
1: -6.7665529, -5.4891520, -6.7347651, -5.5038586, -0.8847311, 0.8796611
2: -0.4624935, 0.8886094, -0.4398813, 0.8824706, -0.7846440, 0.7793700
3: -3.0153670, -1.8138747, -2.9795544, -1.8252349, -0.6039821, 0.5952520
4: -9.0624657, -7.8230505, -9.0401001, -7.8727236, -0.7860796, 0.7790596
5: -8.8704576, -7.5057778, -8.8537626, -7.5333943, -0.5310104, 0.5330930
6: -10.9685879, -9.3183460, -10.9346380, -9.3372126, -0.7401903, 0.7430109
7: 3.2313447, 4.1347685, 3.2470238, 4.1229353, -0.6792409, 0.6760273
8: -4.0591750, -2.7977901, -4.0493145, -2.8040266, -0.6797144, 0.6891323
9: -3.4238811, -2.1735291, -3.4100871, -2.1858244, -0.9405241, 0.9439042

Time for backsubstitution: 23.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664136, upper bound: 0.2642903
time: 3.99 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664136, upper bound: 0.2643279
time: 5.49 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.1156607, -4.3760061, -6.1159706, -4.3715534, -1.1390691, 1.1349072
1: -6.7616014, -5.4912167, -6.7618847, -5.4907408, -0.8823662, 0.8819957
2: -0.4428141, 0.8844187, -0.4434410, 0.8873663, -0.7731440, 0.7711890
3: -2.9967186, -1.8199615, -2.9972072, -1.8169079, -0.5944343, 0.5922877
4: -9.0561619, -7.8381205, -9.0581636, -7.8379922, -0.7885435, 0.7905278
5: -8.8679161, -7.5082326, -8.8680096, -7.5076580, -0.5297046, 0.5291361
6: -10.9431763, -9.3241940, -10.9432707, -9.3204861, -0.7423522, 0.7388146
7: 3.2338686, 4.1293550, 3.2330408, 4.1299605, -0.6748459, 0.6751325
8: -4.0526433, -2.8001852, -4.0538335, -2.7993603, -0.6922834, 0.6926204
9: -3.4189343, -2.1754947, -3.4194760, -2.1753802, -0.9472611, 0.9476805

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641682
time: 4.14 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641701
time: 4.30 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.1455116, -4.3688011, -6.1160469, -4.3704348, -1.1441956, 1.1420457
1: -6.7665529, -5.4891520, -6.7619557, -5.4906211, -0.8869810, 0.8874750
2: -0.4624935, 0.8886094, -0.4435978, 0.8881059, -0.7835904, 0.7803063
3: -3.0153670, -1.8138747, -2.9973292, -1.8161423, -0.6065874, 0.5977311
4: -9.0624657, -7.8230505, -9.0586710, -7.8379593, -0.7926726, 0.8003321
5: -8.8704576, -7.5057778, -8.8680325, -7.5075140, -0.5335114, 0.5311432
6: -10.9685879, -9.3183460, -10.9432955, -9.3195572, -0.7507811, 0.7439685
7: 3.2313447, 4.1347685, 3.2328339, 4.1301126, -0.6772730, 0.6854110
8: -4.0591750, -2.7977901, -4.0541310, -2.7991552, -0.6989337, 0.6949732
9: -3.4238811, -2.1735291, -3.4196115, -2.1753502, -0.9511707, 0.9537542

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664139, upper bound: 0.2642880
time: 3.48 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664139, upper bound: 0.2642879
time: 3.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.53 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641707
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641725
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641718, upper bound: 0.2664113
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641718, upper bound: 0.2664127
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665292
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665291
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641714
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641724
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664136, upper bound: 0.2642903
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664136, upper bound: 0.2643279
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641682
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641701
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664139, upper bound: 0.2642880
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.53
Output dim: 7, lower bound: -0.2664139, upper bound: 0.2642879

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.1120853, -4.3801355, -6.1120853, -4.3801355, -1.1168337, 1.1168334
1: -6.7344117, -5.5044546, -6.7344117, -5.5044546, -0.8652153, 0.8652153
2: -0.4390981, 0.8787858, -0.4390981, 0.8787858, -0.7697291, 0.7697291
3: -2.9789286, -1.8290541, -2.9789286, -1.8290541, -0.5800719, 0.5800720
4: -9.0375748, -7.8728867, -9.0375748, -7.8728867, -0.7560873, 0.7560873
5: -8.8536501, -7.5341215, -8.8536501, -7.5341215, -0.5185337, 0.5185337
6: -10.9345188, -9.3418531, -10.9345188, -9.3418531, -0.7279930, 0.7279932
7: 3.2480803, 4.1221790, 3.2480803, 4.1221790, -0.6610062, 0.6610062
8: -4.0478287, -2.8050623, -4.0478287, -2.8050623, -0.6676121, 0.6676121
9: -3.4094234, -2.1859660, -3.4094234, -2.1859660, -0.9278026, 0.9278028

Time for backsubstitution: 21.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641410, upper bound: 0.2630745
time: 3.52 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641711, upper bound: 0.2641701
time: 4.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.1120853, -4.3801355, -6.1419420, -4.3729477, -1.1243625, 1.1245761
1: -6.7344117, -5.5044546, -6.7393022, -5.5024004, -0.8672628, 0.8695860
2: -0.4390981, 0.8787858, -0.4587880, 0.8829716, -0.7745969, 0.7781061
3: -2.9789286, -1.8290541, -2.9975815, -1.8229480, -0.5859351, 0.5921990
4: -9.0375748, -7.8728867, -9.0438995, -7.8578291, -0.7676589, 0.7605047
5: -8.8536501, -7.5341215, -8.8562002, -7.5316782, -0.5203890, 0.5210241
6: -10.9345188, -9.3418531, -10.9599323, -9.3360462, -0.7340484, 0.7350657
7: 3.2480803, 4.1221790, 3.2456298, 4.1276102, -0.6665289, 0.6631591
8: -4.0478287, -2.8050623, -4.0543966, -2.8026772, -0.6697108, 0.6740263
9: -3.4094234, -2.1859660, -3.4143806, -2.1839986, -0.9297681, 0.9312953

Time for backsubstitution: 22.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641410, upper bound: 0.2630769
time: 3.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641711, upper bound: 0.2641720
time: 3.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.1419420, -4.3729477, -6.1120853, -4.3801355, -1.1245761, 1.1243629
1: -6.7393022, -5.5024004, -6.7344117, -5.5044546, -0.8695860, 0.8672626
2: -0.4587880, 0.8829716, -0.4390981, 0.8787858, -0.7781062, 0.7745969
3: -2.9975815, -1.8229480, -2.9789286, -1.8290541, -0.5921988, 0.5859350
4: -9.0438995, -7.8578291, -9.0375748, -7.8728867, -0.7605047, 0.7676589
5: -8.8562002, -7.5316782, -8.8536501, -7.5341215, -0.5210241, 0.5203890
6: -10.9599323, -9.3360462, -10.9345188, -9.3418531, -0.7350659, 0.7340484
7: 3.2456298, 4.1276102, 3.2480803, 4.1221790, -0.6631591, 0.6665289
8: -4.0543966, -2.8026772, -4.0478287, -2.8050623, -0.6740263, 0.6697108
9: -3.4143806, -2.1839986, -3.4094234, -2.1859660, -0.9312949, 0.9297678

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641455, upper bound: 0.2631917
time: 3.55 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641744, upper bound: 0.2642898
time: 3.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.1419420, -4.3729477, -6.1419420, -4.3729477, -1.1271653, 1.1271653
1: -6.7393022, -5.5024004, -6.7393022, -5.5024004, -0.8732219, 0.8732219
2: -0.4587880, 0.8829716, -0.4587880, 0.8829716, -0.7817702, 0.7817702
3: -2.9975815, -1.8229480, -2.9975815, -1.8229480, -0.5875919, 0.5875918
4: -9.0438995, -7.8578291, -9.0438995, -7.8578291, -0.7679224, 0.7679224
5: -8.8562002, -7.5316782, -8.8562002, -7.5316782, -0.5242033, 0.5242033
6: -10.9599323, -9.3360462, -10.9599323, -9.3360462, -0.7367413, 0.7367415
7: 3.2456298, 4.1276102, 3.2456298, 4.1276102, -0.6738071, 0.6738071
8: -4.0543966, -2.8026772, -4.0543966, -2.8026772, -0.6754053, 0.6754053
9: -3.4143806, -2.1839986, -3.4143806, -2.1839986, -0.9384012, 0.9384012

Time for backsubstitution: 22.06 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.76 + 545.26 = 603.02 seconds
