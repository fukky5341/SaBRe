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
execution time: IAR + RelationalAnalysis = 21.67 + 35.44 = 57.11 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.2665715, upper bound: 0.2665707

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6135
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 6135

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665691, upper bound: 0.2643307
time: 3.42 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665691, upper bound: 0.2665698
time: 3.38 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.04 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 7.04
Output dim: 7, lower bound: -0.2665691, upper bound: 0.2643307
NS_B2, status: Status.UNKNOWN, split count: 1, time: 7.04
Output dim: 7, lower bound: -0.2665691, upper bound: 0.2665698

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -6.1147275, -4.3717451, -6.1124649, -4.3745470, -1.1277857, 1.1278043
1: -6.7461548, -5.4914222, -6.7347665, -5.5038552, -0.8772826, 0.8816290
2: -0.4426168, 0.8848209, -0.4398823, 0.8824776, -0.7788055, 0.7782929
3: -2.9874241, -1.8167076, -2.9795551, -1.8252256, -0.5936627, 0.5936635
4: -9.0569534, -7.8578568, -9.0401068, -7.8727212, -0.7760465, 0.7746680
5: -8.8671370, -7.5224104, -8.8537626, -7.5333920, -0.5339344, 0.5302874
6: -10.9430151, -9.3296490, -10.9346380, -9.3371983, -0.7430892, 0.7383478
7: 3.2403572, 4.1292014, 3.2470202, 4.1229382, -0.6705356, 0.6719079
8: -4.0527897, -2.8019123, -4.0493174, -2.8040218, -0.6740795, 0.6723533
9: -3.4177771, -2.1819963, -3.4100895, -2.1858225, -0.9385147, 0.9335268

Time for backsubstitution: 20.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6135
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6135

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2643283
time: 3.27 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2643262
time: 3.62 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -6.1160488, -4.3704224, -6.1160474, -4.3704243, -1.1306837, 1.1407063
1: -6.7619610, -5.4906173, -6.7619572, -5.4906197, -0.9051821, 0.8832941
2: -0.4435997, 0.8881146, -0.4435993, 0.8881130, -0.7759268, 0.7804878
3: -2.9973333, -1.8161321, -2.9973316, -1.8161314, -0.6144042, 0.5964303
4: -9.0586777, -7.8379507, -9.0586767, -7.8379598, -0.7914464, 0.8200490
5: -8.8680325, -7.5075068, -8.8680325, -7.5075116, -0.5302438, 0.5560055
6: -10.9432964, -9.3195410, -10.9432945, -9.3195448, -0.7433980, 0.7562802
7: 3.2328279, 4.1301150, 3.2328300, 4.1301150, -0.6873443, 0.6770077
8: -4.0541353, -2.7991509, -4.0541344, -2.7991509, -0.6940451, 0.6927762
9: -3.4196134, -2.1753492, -3.4196134, -2.1753507, -0.9490571, 0.9527464

Time for backsubstitution: 20.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6135
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 161

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 6135

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2665698
time: 3.50 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2665678
time: 3.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.00 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2643283
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2643262
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2665698
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 28.00
Output dim: 7, lower bound: -0.2643272, upper bound: 0.2665678

## BFS NS instance: NS_B1_A1

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

Time for backsubstitution: 20.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642868, upper bound: 0.2641756
time: 3.46 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643265, upper bound: 0.2643299
time: 3.31 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -6.1158543, -4.3704453, -6.1124649, -4.3745470, -1.1257825, 1.1270802
1: -6.7616320, -5.4906511, -6.7347665, -5.5038552, -0.8825259, 0.8783793
2: -0.4434599, 0.8874973, -0.4398823, 0.8824776, -0.7797716, 0.7808807
3: -2.9970329, -1.8161564, -2.9795551, -1.8252256, -0.6004374, 0.5939929
4: -9.0584888, -7.8379841, -9.0401068, -7.8727212, -0.7715220, 0.7780247
5: -8.8677197, -7.5075359, -8.8537626, -7.5333920, -0.5283329, 0.5323037
6: -10.9431963, -9.3195457, -10.9346380, -9.3371983, -0.7391212, 0.7423910
7: 3.2329042, 4.1300812, 3.2470202, 4.1229382, -0.6769214, 0.6714015
8: -4.0540771, -2.7996769, -4.0493174, -2.8040218, -0.6748656, 0.6743753
9: -3.4196038, -2.1765537, -3.4100895, -2.1858225, -0.9386151, 0.9389350

Time for backsubstitution: 21.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641722, upper bound: 0.2642903
time: 3.43 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643269, upper bound: 0.2643300
time: 3.58 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -6.1124649, -4.3745470, -6.1160474, -4.3704243, -1.1270976, 1.1250565
1: -6.7347665, -5.5038552, -6.7619572, -5.4906197, -0.8784204, 0.8829398
2: -0.4398823, 0.8824776, -0.4435993, 0.8881130, -0.7765160, 0.7798965
3: -2.9795551, -1.8252256, -2.9973316, -1.8161314, -0.5940124, 0.6029370
4: -9.0401068, -7.8727212, -9.0586767, -7.8379598, -0.7780411, 0.7847505
5: -8.8537626, -7.5333920, -8.8680325, -7.5075116, -0.5323249, 0.5304101
6: -10.9346380, -9.3371983, -10.9432945, -9.3195448, -0.7423913, 0.7387253
7: 3.2470202, 4.1229382, 3.2328300, 4.1301150, -0.6714373, 0.6788507
8: -4.0493174, -2.8040218, -4.0541344, -2.7991509, -0.6881467, 0.6749104
9: -3.4100895, -2.1858225, -3.4196134, -2.1753507, -0.9436221, 0.9386208

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_B2_A1_A1

### Relational analysis result of NS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642868, upper bound: 0.2664148
time: 3.43 seconds

## Relational analysis of NS_B2_A1_A2

### Relational analysis result of NS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643265, upper bound: 0.2665691
time: 3.48 seconds

## BFS NS instance: NS_B2_A2

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

Time for backsubstitution: 21.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641722, upper bound: 0.2642860
time: 3.68 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2643269, upper bound: 0.2643256
time: 3.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.00 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2642868, upper bound: 0.2641756
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2643265, upper bound: 0.2643299
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2641722, upper bound: 0.2642903
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2643269, upper bound: 0.2643300
NS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2642868, upper bound: 0.2664148
NS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2643265, upper bound: 0.2665691
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2641722, upper bound: 0.2642860
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 29.00
Output dim: 7, lower bound: -0.2643269, upper bound: 0.2643256

## BFS NS instance: NS_B1_A1_A1

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

Time for backsubstitution: 20.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641707
time: 3.69 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641725
time: 3.62 seconds

## BFS NS instance: NS_B1_A1_A2

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

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
time: 3.42 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
time: 3.47 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.1157742, -4.3715739, -6.1120853, -4.3801355, -1.1199727, 1.1254766
1: -6.7615604, -5.4907727, -6.7344117, -5.5044546, -0.8812032, 0.8774297
2: -0.4433006, 0.8867483, -0.4390981, 0.8787858, -0.7750387, 0.7781148
3: -2.9969094, -1.8169312, -2.9789286, -1.8290541, -0.5962787, 0.5920252
4: -9.0579767, -7.8380156, -9.0375748, -7.8728867, -0.7705388, 0.7751310
5: -8.8676958, -7.5076818, -8.8536501, -7.5341215, -0.5272025, 0.5317634
6: -10.9431705, -9.3204851, -10.9345188, -9.3418531, -0.7345347, 0.7412143
7: 3.2331152, 4.1299281, 3.2480803, 4.1221790, -0.6750031, 0.6692410
8: -4.0537763, -2.7998857, -4.0478287, -2.8050623, -0.6734345, 0.6725874
9: -3.4194670, -2.1765828, -3.4094234, -2.1859660, -0.9372256, 0.9370611

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641706
time: 4.02 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641725
time: 3.49 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.1158524, -4.3704567, -6.1419420, -4.3729477, -1.1271324, 1.1330400
1: -6.7616310, -5.4906530, -6.7393022, -5.5024004, -0.8837483, 0.8801470
2: -0.4434575, 0.8874890, -0.4587880, 0.8829716, -0.7841563, 0.7839023
3: -2.9970331, -1.8161659, -2.9975815, -1.8229480, -0.6016541, 0.5973214
4: -9.0584831, -7.8379827, -9.0438995, -7.8578291, -0.7725346, 0.7793887
5: -8.8677197, -7.5075369, -8.8562002, -7.5316782, -0.5290866, 0.5329286
6: -10.9431973, -9.3195591, -10.9599323, -9.3360462, -0.7397199, 0.7438526
7: 3.2329075, 4.1300793, 3.2456298, 4.1276102, -0.6782777, 0.6716948
8: -4.0540733, -2.7996807, -4.0543966, -2.8026772, -0.6758163, 0.6788325
9: -3.4196024, -2.1765552, -3.4143806, -2.1839986, -0.9418693, 0.9398184

Time for backsubstitution: 21.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665284, upper bound: 0.2641752
time: 3.56 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665284, upper bound: 0.2641757
time: 3.51 seconds

## BFS NS instance: NS_B2_A1_A1

### Backsubstitution after applying NS history:
0: -6.1120853, -4.3801355, -6.1159706, -4.3715534, -1.1254938, 1.1192479
1: -6.7344117, -5.5044546, -6.7618847, -5.4907408, -0.8774705, 0.8816178
2: -0.4390981, 0.8787858, -0.4434410, 0.8873663, -0.7737331, 0.7751641
3: -2.9789286, -1.8290541, -2.9972072, -1.8169079, -0.5920444, 0.5987856
4: -9.0375748, -7.8728867, -9.0581636, -7.8379922, -0.7751467, 0.7837653
5: -8.8536501, -7.5341215, -8.8680096, -7.5076580, -0.5317844, 0.5292844
6: -10.9345188, -9.3418531, -10.9432707, -9.3204861, -0.7412145, 0.7341481
7: 3.2480803, 4.1221790, 3.2330408, 4.1299605, -0.6692767, 0.6769326
8: -4.0478287, -2.8050623, -4.0538335, -2.7993603, -0.6863792, 0.6734793
9: -3.4094234, -2.1859660, -3.4194760, -2.1753802, -0.9417958, 0.9372313

Time for backsubstitution: 21.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 540

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_B2_A1_A1_B1

### Relational analysis result of NS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2631910, upper bound: 0.2663837
time: 7.47 seconds

## Relational analysis of NS_B2_A1_A1_B2

### Relational analysis result of NS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2642891, upper bound: 0.2664141
time: 3.54 seconds

## BFS NS instance: NS_B2_A1_A2

### Backsubstitution after applying NS history:
0: -6.1419420, -4.3729477, -6.1160469, -4.3704348, -1.1330566, 1.1264040
1: -6.7393022, -5.5024004, -6.7619557, -5.4906211, -0.8801880, 0.8841627
2: -0.4587880, 0.8829716, -0.4435978, 0.8881059, -0.7794673, 0.7842817
3: -2.9975815, -1.8229480, -2.9973292, -1.8161423, -0.5973413, 0.6042168
4: -9.0438995, -7.8578291, -9.0586710, -7.8379593, -0.7794049, 0.7857630
5: -8.8562002, -7.5316782, -8.8680325, -7.5075140, -0.5329497, 0.5312266
6: -10.9599323, -9.3360462, -10.9432955, -9.3195572, -0.7438526, 0.7393663
7: 3.2456298, 4.1276102, 3.2328339, 4.1301126, -0.6717308, 0.6802070
8: -4.0543966, -2.8026772, -4.0541310, -2.7991552, -0.6927776, 0.6758614
9: -3.4143806, -2.1839986, -3.4196115, -2.1753502, -0.9447205, 0.9418747

Time for backsubstitution: 21.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_B2_A1_A2_B1

### Relational analysis result of NS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665292
time: 3.46 seconds

## Relational analysis of NS_B2_A1_A2_B2

### Relational analysis result of NS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665291
time: 3.51 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.1159706, -4.3715534, -6.1156607, -4.3760061, -1.1349072, 1.1390688
1: -6.7618847, -5.4907408, -6.7616014, -5.4912167, -0.8819957, 0.8823659
2: -0.4434410, 0.8873663, -0.4428141, 0.8844187, -0.7711890, 0.7731440
3: -2.9972072, -1.8169079, -2.9967186, -1.8199615, -0.5922877, 0.5944343
4: -9.0581636, -7.8379922, -9.0561619, -7.8381205, -0.7905278, 0.7885435
5: -8.8680096, -7.5076580, -8.8679161, -7.5082326, -0.5291361, 0.5297046
6: -10.9432707, -9.3204861, -10.9431763, -9.3241940, -0.7388148, 0.7423522
7: 3.2330408, 4.1299605, 3.2338686, 4.1293550, -0.6751325, 0.6748459
8: -4.0538335, -2.7993603, -4.0526433, -2.8001852, -0.6926205, 0.6922835
9: -3.4194760, -2.1753802, -3.4189343, -2.1754947, -0.9476805, 0.9472609

Time for backsubstitution: 21.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641682
time: 4.08 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641701
time: 3.72 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.1160469, -4.3704348, -6.1455116, -4.3688011, -1.1420455, 1.1482244
1: -6.7619557, -5.4906211, -6.7665529, -5.4891520, -0.8874753, 0.8869808
2: -0.4435978, 0.8881059, -0.4624935, 0.8886094, -0.7803063, 0.7824481
3: -2.9973292, -1.8161423, -3.0153670, -1.8138747, -0.5977312, 0.6076453
4: -9.0586710, -7.8379593, -9.0624657, -7.8230505, -0.8041356, 0.7926726
5: -8.8680325, -7.5075140, -8.8704576, -7.5057778, -0.5311431, 0.5335114
6: -10.9432955, -9.3195572, -10.9685879, -9.3183460, -0.7439685, 0.7489234
7: 3.2328339, 4.1301126, 3.2313447, 4.1347685, -0.6854110, 0.6772730
8: -4.0541310, -2.7991552, -4.0591750, -2.7977901, -0.6949732, 0.6989337
9: -3.4196115, -2.1753502, -3.4238811, -2.1735291, -0.9552884, 0.9511706

Time for backsubstitution: 21.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 540

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665289, upper bound: 0.2641733
time: 3.55 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2665289, upper bound: 0.2641714
time: 5.61 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.65 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641707
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2641713, upper bound: 0.2641725
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2641746, upper bound: 0.2642904
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641706
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2664105, upper bound: 0.2641725
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2665284, upper bound: 0.2641752
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2665284, upper bound: 0.2641757
NS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2631910, upper bound: 0.2663837
NS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2642891, upper bound: 0.2664141
NS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665292
NS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2641745, upper bound: 0.2665291
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641682
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2664109, upper bound: 0.2641701
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2665289, upper bound: 0.2641733
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 30.65
Output dim: 7, lower bound: -0.2665289, upper bound: 0.2641714

## BFS NS instance: NS_B1_A1_A1_B1

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

Time for backsubstitution: 21.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641410, upper bound: 0.2630745
time: 3.55 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641711, upper bound: 0.2641701
time: 4.02 seconds

## BFS NS instance: NS_B1_A1_A1_B2

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

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_B1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2630738, upper bound: 0.2641444
time: 4.68 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641709, upper bound: 0.2641736
time: 3.75 seconds

## BFS NS instance: NS_B1_A1_A2_B1

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

Time for backsubstitution: 22.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 108
type: A, layer: 1, pos: 108

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 161

## Relational analysis of NS_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641455, upper bound: 0.2631917
time: 3.57 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641744, upper bound: 0.2642898
time: 3.71 seconds

## BFS NS instance: NS_B1_A1_A2_B2

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

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 108
type: B, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 161

## Relational analysis of NS_B1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2630778, upper bound: 0.2642589
time: 3.49 seconds

## Relational analysis of NS_B1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.2641741, upper bound: 0.2642901
time: 3.57 seconds

## BFS NS instance: NS_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.1154671, -4.3760219, -6.1120853, -4.3801355, -1.1196084, 1.1209273
1: -6.7612762, -5.4912500, -6.7344117, -5.5044546, -0.8806586, 0.8765101
2: -0.4426749, 0.8838021, -0.4390981, 0.8787858, -0.7735379, 0.7746432
3: -2.9964204, -1.8199852, -2.9789286, -1.8290541, -0.5952921, 0.5889106
4: -9.0559731, -7.8381453, -9.0375748, -7.8728867, -0.7683549, 0.7748518
5: -8.8676023, -7.5082555, -8.8536501, -7.5341215, -0.5269564, 0.5309318
6: -10.9430761, -9.3241949, -10.9345188, -9.3418531, -0.7344360, 0.7377059
7: 3.2339430, 4.1293211, 3.2480803, 4.1221790, -0.6735592, 0.6680379
8: -4.0525861, -2.8007092, -4.0478287, -2.8050623, -0.6721787, 0.6717055
9: -3.4189262, -2.1766968, -3.4094234, -2.1859660, -0.9359269, 0.9362421

Time for backsubstitution: 22.19 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.11 + 552.95 = 610.06 seconds
