## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0014790720000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0065673, 0.0065673)
1: (-0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018516, 0.0018516)
2: (-0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0136613, 0.0136613)
3: (0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0018079, 0.0018079)
4: (-0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0102096, 0.0102096)
5: (0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028365, 0.0028365)
6: (0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025747, 0.0025747)
7: (-0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0096084, 0.0096084)
8: (-0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074782, 0.0074782)
9: (-0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006452, 0.0006452)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.56 + 2.76 = 4.32 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0015904, upper bound: 0.0015903

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015478, upper bound: 0.0015255
time: 1.83 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015524, upper bound: 0.0015525
time: 1.92 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 3.88 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 3.88
Output dim: 5, lower bound: -0.0015478, upper bound: 0.0015255
NS_A2, status: Status.UNKNOWN, split count: 1, time: 3.88
Output dim: 5, lower bound: -0.0015524, upper bound: 0.0015525

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0088162, -0.0008473, -0.0090725, -0.0009228, -0.0061497, 0.0065136
1: -0.0054243, -0.0031775, -0.0054965, -0.0031988, -0.0017338, 0.0018364
2: -0.0014617, 0.0151154, -0.0019949, 0.0149583, -0.0127927, 0.0135496
3: 0.0014339, 0.0036276, 0.0013633, 0.0036068, -0.0016929, 0.0017931
4: -0.0052044, 0.0071842, -0.0050871, 0.0075827, -0.0101261, 0.0095605
5: 0.9940603, 0.9975023, 0.9940929, 0.9976130, -0.0028133, 0.0026562
6: 0.0024922, 0.0056164, 0.0025218, 0.0057169, -0.0025537, 0.0024110
7: -0.0140811, -0.0024220, -0.0139706, -0.0020470, -0.0095298, 0.0089975
8: -0.0073078, 0.0017665, -0.0075997, 0.0016805, -0.0070027, 0.0074171
9: -0.0041621, -0.0033793, -0.0041547, -0.0033541, -0.0006399, 0.0006042

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015044, upper bound: 0.0014971
time: 1.83 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015187, upper bound: 0.0014973
time: 1.43 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0091315, -0.0009223, -0.0092295, -0.0009076, -0.0063430, 0.0065128
1: -0.0055132, -0.0031987, -0.0055408, -0.0031945, -0.0017883, 0.0018362
2: -0.0021175, 0.0149593, -0.0023214, 0.0149898, -0.0131947, 0.0135479
3: 0.0013471, 0.0036069, 0.0013201, 0.0036110, -0.0017461, 0.0017929
4: -0.0050878, 0.0076743, -0.0051106, 0.0078267, -0.0101249, 0.0098609
5: 0.9940927, 0.9976385, 0.9940863, 0.9976807, -0.0028130, 0.0027396
6: 0.0025216, 0.0057400, 0.0025158, 0.0057785, -0.0025533, 0.0024868
7: -0.0139713, -0.0019607, -0.0139928, -0.0018173, -0.0095286, 0.0092802
8: -0.0076668, 0.0016810, -0.0077784, 0.0016977, -0.0072228, 0.0074161
9: -0.0041548, -0.0033483, -0.0041562, -0.0033386, -0.0006398, 0.0006231

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015096, upper bound: 0.0015235
time: 1.65 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015235, upper bound: 0.0015235
time: 2.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.18 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.18
Output dim: 5, lower bound: -0.0015044, upper bound: 0.0014971
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.18
Output dim: 5, lower bound: -0.0015187, upper bound: 0.0014973
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.18
Output dim: 5, lower bound: -0.0015096, upper bound: 0.0015235
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.18
Output dim: 5, lower bound: -0.0015235, upper bound: 0.0015235

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0087041, -0.0008756, -0.0088102, -0.0009452, -0.0059908, 0.0062229
1: -0.0053927, -0.0031855, -0.0054226, -0.0032051, -0.0016890, 0.0017545
2: -0.0012285, 0.0150565, -0.0014491, 0.0149116, -0.0124621, 0.0129448
3: 0.0014647, 0.0036198, 0.0014355, 0.0036006, -0.0016492, 0.0017130
4: -0.0051604, 0.0070099, -0.0050522, 0.0071748, -0.0096742, 0.0093134
5: 0.9940725, 0.9974538, 0.9941025, 0.9974996, -0.0026878, 0.0025875
6: 0.0025033, 0.0055725, 0.0025306, 0.0056141, -0.0024397, 0.0023487
7: -0.0140397, -0.0025860, -0.0139378, -0.0024308, -0.0091045, 0.0087650
8: -0.0071802, 0.0017342, -0.0073009, 0.0016549, -0.0068218, 0.0070860
9: -0.0041594, -0.0033903, -0.0041525, -0.0033798, -0.0006113, 0.0005886

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014707, upper bound: 0.0014711
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014797, upper bound: 0.0014711
time: 1.54 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0087740, -0.0008590, -0.0089446, -0.0009578, -0.0060751, 0.0062689
1: -0.0054124, -0.0031808, -0.0054605, -0.0032087, -0.0017128, 0.0017674
2: -0.0013739, 0.0150910, -0.0017287, 0.0148854, -0.0126374, 0.0130405
3: 0.0014455, 0.0036243, 0.0013985, 0.0035971, -0.0016724, 0.0017257
4: -0.0051862, 0.0071186, -0.0050326, 0.0073838, -0.0097457, 0.0094444
5: 0.9940653, 0.9974840, 0.9941080, 0.9975576, -0.0027076, 0.0026239
6: 0.0024968, 0.0055999, 0.0025355, 0.0056668, -0.0024577, 0.0023818
7: -0.0140639, -0.0024837, -0.0139193, -0.0022342, -0.0091718, 0.0088883
8: -0.0072598, 0.0017531, -0.0074540, 0.0016406, -0.0069178, 0.0071384
9: -0.0041610, -0.0033834, -0.0041513, -0.0033666, -0.0006159, 0.0005968

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014858, upper bound: 0.0014713
time: 1.42 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014918, upper bound: 0.0014713
time: 1.49 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090195, -0.0009507, -0.0089665, -0.0009307, -0.0061635, 0.0062264
1: -0.0054816, -0.0032067, -0.0054666, -0.0032010, -0.0017377, 0.0017555
2: -0.0018846, 0.0149003, -0.0017743, 0.0149419, -0.0128213, 0.0129522
3: 0.0013779, 0.0035991, 0.0013925, 0.0036046, -0.0016967, 0.0017140
4: -0.0050437, 0.0075003, -0.0050748, 0.0074178, -0.0096797, 0.0095819
5: 0.9941050, 0.9975901, 0.9940963, 0.9975671, -0.0026893, 0.0026621
6: 0.0025327, 0.0056961, 0.0025249, 0.0056753, -0.0024411, 0.0024164
7: -0.0139298, -0.0021245, -0.0139591, -0.0022021, -0.0091097, 0.0090176
8: -0.0075394, 0.0016487, -0.0074789, 0.0016715, -0.0070184, 0.0070901
9: -0.0041520, -0.0033593, -0.0041539, -0.0033645, -0.0006117, 0.0006055

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014758, upper bound: 0.0014967
time: 2.02 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014850, upper bound: 0.0014967
time: 1.84 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090897, -0.0009339, -0.0091022, -0.0009429, -0.0062696, 0.0062993
1: -0.0055014, -0.0032020, -0.0055049, -0.0032045, -0.0017676, 0.0017760
2: -0.0020306, 0.0149352, -0.0020565, 0.0149165, -0.0130421, 0.0131039
3: 0.0013586, 0.0036037, 0.0013551, 0.0036013, -0.0017259, 0.0017341
4: -0.0050698, 0.0076094, -0.0050558, 0.0076288, -0.0097930, 0.0097468
5: 0.9940977, 0.9976203, 0.9941016, 0.9976258, -0.0027208, 0.0027080
6: 0.0025261, 0.0057237, 0.0025297, 0.0057285, -0.0024697, 0.0024580
7: -0.0139544, -0.0020218, -0.0139412, -0.0020036, -0.0092163, 0.0091729
8: -0.0076193, 0.0016678, -0.0076335, 0.0016576, -0.0071393, 0.0071731
9: -0.0041536, -0.0033524, -0.0041527, -0.0033512, -0.0006189, 0.0006159

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014907, upper bound: 0.0014967
time: 1.55 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014967, upper bound: 0.0014967
time: 1.85 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.89 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014707, upper bound: 0.0014711
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014797, upper bound: 0.0014711
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014858, upper bound: 0.0014713
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014918, upper bound: 0.0014713
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014758, upper bound: 0.0014967
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014850, upper bound: 0.0014967
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014907, upper bound: 0.0014967
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.89
Output dim: 5, lower bound: -0.0014967, upper bound: 0.0014967

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0087020, -0.0010490, -0.0088094, -0.0010118, -0.0059269, 0.0059735
1: -0.0053921, -0.0032344, -0.0054224, -0.0032239, -0.0016710, 0.0016842
2: -0.0012242, 0.0146956, -0.0014475, 0.0147731, -0.0123291, 0.0124261
3: 0.0014653, 0.0035720, 0.0014357, 0.0035823, -0.0016316, 0.0016444
4: -0.0048907, 0.0070067, -0.0049486, 0.0071737, -0.0092865, 0.0092140
5: 0.9941474, 0.9974529, 0.9941314, 0.9974993, -0.0025801, 0.0025599
6: 0.0025713, 0.0055717, 0.0025567, 0.0056138, -0.0023419, 0.0023236
7: -0.0137859, -0.0025890, -0.0138404, -0.0024319, -0.0087396, 0.0086714
8: -0.0071778, 0.0015367, -0.0073001, 0.0015791, -0.0067489, 0.0068021
9: -0.0041423, -0.0033905, -0.0041460, -0.0033799, -0.0005868, 0.0005823

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014317, upper bound: 0.0014181
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014317, upper bound: 0.0014224
time: 1.92 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0087563, -0.0011464, -0.0089435, -0.0010698, -0.0059041, 0.0059572
1: -0.0054074, -0.0032619, -0.0054602, -0.0032403, -0.0016646, 0.0016796
2: -0.0013371, 0.0144931, -0.0017264, 0.0146525, -0.0122817, 0.0123922
3: 0.0014504, 0.0035452, 0.0013988, 0.0035663, -0.0016253, 0.0016399
4: -0.0047394, 0.0070911, -0.0048585, 0.0073821, -0.0092612, 0.0091786
5: 0.9941894, 0.9974763, 0.9941564, 0.9975572, -0.0025730, 0.0025501
6: 0.0026095, 0.0055929, 0.0025794, 0.0056663, -0.0023355, 0.0023147
7: -0.0136434, -0.0025096, -0.0137555, -0.0022358, -0.0087158, 0.0086380
8: -0.0072396, 0.0014258, -0.0074527, 0.0015131, -0.0067230, 0.0067835
9: -0.0041327, -0.0033851, -0.0041403, -0.0033667, -0.0005852, 0.0005800

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014196
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014244
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0087719, -0.0010323, -0.0089438, -0.0010277, -0.0060108, 0.0060132
1: -0.0054118, -0.0032297, -0.0054603, -0.0032284, -0.0016947, 0.0016953
2: -0.0013695, 0.0147305, -0.0017271, 0.0147401, -0.0125036, 0.0125086
3: 0.0014461, 0.0035766, 0.0013987, 0.0035779, -0.0016547, 0.0016553
4: -0.0049168, 0.0071153, -0.0049240, 0.0073826, -0.0093482, 0.0093444
5: 0.9941401, 0.9974831, 0.9941382, 0.9975573, -0.0025972, 0.0025962
6: 0.0025647, 0.0055990, 0.0025629, 0.0056664, -0.0023575, 0.0023565
7: -0.0138104, -0.0024868, -0.0138171, -0.0022353, -0.0087977, 0.0087941
8: -0.0072574, 0.0015558, -0.0074531, 0.0015610, -0.0068445, 0.0068472
9: -0.0041440, -0.0033836, -0.0041444, -0.0033667, -0.0005907, 0.0005905

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014195
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014244
time: 1.91 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0090017, -0.0012406, -0.0089655, -0.0010464, -0.0059960, 0.0059282
1: -0.0054766, -0.0032884, -0.0054664, -0.0032337, -0.0016905, 0.0016714
2: -0.0018476, 0.0142971, -0.0017723, 0.0147012, -0.0124730, 0.0123319
3: 0.0013828, 0.0035193, 0.0013928, 0.0035728, -0.0016506, 0.0016319
4: -0.0045929, 0.0074727, -0.0048949, 0.0074163, -0.0092161, 0.0093215
5: 0.9942302, 0.9975824, 0.9941463, 0.9975668, -0.0025605, 0.0025898
6: 0.0026464, 0.0056892, 0.0025703, 0.0056750, -0.0023242, 0.0023508
7: -0.0135056, -0.0021505, -0.0137898, -0.0022035, -0.0086734, 0.0087726
8: -0.0075191, 0.0013185, -0.0074778, 0.0015397, -0.0068277, 0.0067505
9: -0.0041235, -0.0033610, -0.0041426, -0.0033646, -0.0005824, 0.0005891

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014480, upper bound: 0.0014916
time: 1.46 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014480, upper bound: 0.0014967
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0090175, -0.0011276, -0.0089657, -0.0009971, -0.0061031, 0.0059903
1: -0.0054810, -0.0032566, -0.0054664, -0.0032198, -0.0017207, 0.0016889
2: -0.0018804, 0.0145322, -0.0017727, 0.0148036, -0.0126956, 0.0124611
3: 0.0013784, 0.0035504, 0.0013927, 0.0035863, -0.0016801, 0.0016490
4: -0.0047686, 0.0074972, -0.0049714, 0.0074167, -0.0093126, 0.0094879
5: 0.9941814, 0.9975892, 0.9941250, 0.9975668, -0.0025873, 0.0026360
6: 0.0026021, 0.0056953, 0.0025510, 0.0056750, -0.0023485, 0.0023927
7: -0.0136709, -0.0021275, -0.0138618, -0.0022032, -0.0087642, 0.0089292
8: -0.0075371, 0.0014472, -0.0074781, 0.0015958, -0.0069496, 0.0068212
9: -0.0041346, -0.0033595, -0.0041474, -0.0033646, -0.0005885, 0.0005996

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014575, upper bound: 0.0014916
time: 1.96 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014575, upper bound: 0.0014967
time: 2.33 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0090767, -0.0012231, -0.0091011, -0.0010548, -0.0061000, 0.0060006
1: -0.0054977, -0.0032835, -0.0055046, -0.0032360, -0.0017198, 0.0016918
2: -0.0020036, 0.0143337, -0.0020542, 0.0146837, -0.0126893, 0.0124825
3: 0.0013621, 0.0035241, 0.0013555, 0.0035705, -0.0016792, 0.0016519
4: -0.0046202, 0.0075892, -0.0048818, 0.0076270, -0.0093287, 0.0094832
5: 0.9942226, 0.9976147, 0.9941500, 0.9976252, -0.0025918, 0.0026347
6: 0.0026395, 0.0057186, 0.0025735, 0.0057281, -0.0023526, 0.0023915
7: -0.0135313, -0.0020408, -0.0137775, -0.0020052, -0.0087793, 0.0089247
8: -0.0076045, 0.0013385, -0.0076322, 0.0015302, -0.0069461, 0.0068330
9: -0.0041252, -0.0033537, -0.0041418, -0.0033513, -0.0005895, 0.0005993

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014644, upper bound: 0.0014918
time: 2.24 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014644, upper bound: 0.0014967
time: 2.07 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0090876, -0.0011104, -0.0091014, -0.0010128, -0.0062062, 0.0060618
1: -0.0055008, -0.0032517, -0.0055047, -0.0032242, -0.0017498, 0.0017090
2: -0.0020262, 0.0145679, -0.0020549, 0.0147711, -0.0129102, 0.0126098
3: 0.0013592, 0.0035551, 0.0013554, 0.0035820, -0.0017085, 0.0016687
4: -0.0047953, 0.0076062, -0.0049471, 0.0076276, -0.0094238, 0.0096483
5: 0.9941739, 0.9976195, 0.9941318, 0.9976254, -0.0026182, 0.0026806
6: 0.0025954, 0.0057228, 0.0025571, 0.0057282, -0.0023765, 0.0024332
7: -0.0136960, -0.0020249, -0.0138389, -0.0020048, -0.0088688, 0.0090801
8: -0.0076169, 0.0014668, -0.0076326, 0.0015780, -0.0070671, 0.0069026
9: -0.0041363, -0.0033526, -0.0041459, -0.0033512, -0.0005955, 0.0006097

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 177
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014713, upper bound: 0.0014918
time: 2.06 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014713, upper bound: 0.0014967
time: 2.25 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.91 seconds
NS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014317, upper bound: 0.0014181
NS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014317, upper bound: 0.0014224
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014196
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014244
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014195
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014244
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014480, upper bound: 0.0014916
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014480, upper bound: 0.0014967
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014575, upper bound: 0.0014916
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014575, upper bound: 0.0014967
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014644, upper bound: 0.0014918
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014644, upper bound: 0.0014967
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014713, upper bound: 0.0014918
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.91
Output dim: 5, lower bound: -0.0014713, upper bound: 0.0014967

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090017, -0.0012406, -0.0085537, -0.0009848, -0.0062336, 0.0055598
1: -0.0054766, -0.0032884, -0.0053503, -0.0032163, -0.0017575, 0.0015675
2: -0.0018476, 0.0142971, -0.0009157, 0.0148293, -0.0129672, 0.0115655
3: 0.0013828, 0.0035193, 0.0015061, 0.0035897, -0.0017160, 0.0015305
4: -0.0045929, 0.0074727, -0.0049906, 0.0067762, -0.0086433, 0.0096908
5: 0.9942302, 0.9975824, 0.9941198, 0.9973888, -0.0024014, 0.0026924
6: 0.0026464, 0.0056892, 0.0025461, 0.0055135, -0.0021797, 0.0024439
7: -0.0135056, -0.0021505, -0.0138798, -0.0028060, -0.0081343, 0.0091202
8: -0.0075191, 0.0013185, -0.0070090, 0.0016098, -0.0070982, 0.0063310
9: -0.0041235, -0.0033610, -0.0041486, -0.0034050, -0.0005462, 0.0006124

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013925, upper bound: 0.0014442
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013925, upper bound: 0.0014442
time: 2.28 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0090017, -0.0012406, -0.0088689, -0.0010609, -0.0059824, 0.0057315
1: -0.0054766, -0.0032884, -0.0054391, -0.0032378, -0.0016867, 0.0016159
2: -0.0018476, 0.0142971, -0.0015713, 0.0146710, -0.0124446, 0.0119226
3: 0.0013828, 0.0035193, 0.0014194, 0.0035688, -0.0016468, 0.0015778
4: -0.0045929, 0.0074727, -0.0048723, 0.0072661, -0.0089102, 0.0093003
5: 0.9942302, 0.9975824, 0.9941525, 0.9975249, -0.0024755, 0.0025839
6: 0.0026464, 0.0056892, 0.0025759, 0.0056371, -0.0022470, 0.0023454
7: -0.0135056, -0.0021505, -0.0137685, -0.0023449, -0.0083855, 0.0087527
8: -0.0075191, 0.0013185, -0.0073678, 0.0015232, -0.0068122, 0.0065264
9: -0.0041235, -0.0033610, -0.0041412, -0.0033741, -0.0005631, 0.0005877

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013925, upper bound: 0.0014529
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013974, upper bound: 0.0014529
time: 2.26 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090175, -0.0011276, -0.0085540, -0.0009380, -0.0063272, 0.0056029
1: -0.0054810, -0.0032566, -0.0053503, -0.0032031, -0.0017839, 0.0015797
2: -0.0018804, 0.0145322, -0.0009162, 0.0149266, -0.0131619, 0.0116552
3: 0.0013784, 0.0035504, 0.0015061, 0.0036026, -0.0017418, 0.0015424
4: -0.0047686, 0.0074972, -0.0050633, 0.0067765, -0.0087103, 0.0098364
5: 0.9941814, 0.9975892, 0.9940996, 0.9973889, -0.0024200, 0.0027328
6: 0.0026021, 0.0056953, 0.0025278, 0.0055136, -0.0021966, 0.0024806
7: -0.0136709, -0.0021275, -0.0139483, -0.0028057, -0.0081974, 0.0092571
8: -0.0075371, 0.0014472, -0.0070092, 0.0016631, -0.0072048, 0.0063800
9: -0.0041346, -0.0033595, -0.0041532, -0.0034050, -0.0005504, 0.0006216

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014007, upper bound: 0.0014442
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014080, upper bound: 0.0014442
time: 1.88 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090175, -0.0011276, -0.0088691, -0.0010115, -0.0060892, 0.0057836
1: -0.0054810, -0.0032566, -0.0054392, -0.0032238, -0.0017168, 0.0016306
2: -0.0018804, 0.0145322, -0.0015717, 0.0147736, -0.0126668, 0.0120311
3: 0.0013784, 0.0035504, 0.0014193, 0.0035824, -0.0016762, 0.0015921
4: -0.0047686, 0.0074972, -0.0049490, 0.0072665, -0.0089913, 0.0094664
5: 0.9941814, 0.9975892, 0.9941312, 0.9975250, -0.0024981, 0.0026300
6: 0.0026021, 0.0056953, 0.0025566, 0.0056372, -0.0022675, 0.0023873
7: -0.0136709, -0.0021275, -0.0138407, -0.0023446, -0.0084618, 0.0089089
8: -0.0075371, 0.0014472, -0.0073681, 0.0015794, -0.0069338, 0.0065858
9: -0.0041346, -0.0033595, -0.0041460, -0.0033741, -0.0005682, 0.0005982

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014007, upper bound: 0.0014529
time: 2.18 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014081, upper bound: 0.0014529
time: 2.04 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0090767, -0.0012231, -0.0086889, -0.0009942, -0.0063388, 0.0056330
1: -0.0054977, -0.0032835, -0.0053884, -0.0032190, -0.0017871, 0.0015882
2: -0.0020036, 0.0143337, -0.0011968, 0.0148096, -0.0131859, 0.0117178
3: 0.0013621, 0.0035241, 0.0014689, 0.0035871, -0.0017449, 0.0015507
4: -0.0046202, 0.0075892, -0.0049759, 0.0069863, -0.0087572, 0.0098543
5: 0.9942226, 0.9976147, 0.9941238, 0.9974472, -0.0024330, 0.0027378
6: 0.0026395, 0.0057186, 0.0025498, 0.0055665, -0.0022084, 0.0024851
7: -0.0135313, -0.0020408, -0.0138660, -0.0026083, -0.0082415, 0.0092740
8: -0.0076045, 0.0013385, -0.0071628, 0.0015991, -0.0072180, 0.0064144
9: -0.0041252, -0.0033537, -0.0041477, -0.0033918, -0.0005534, 0.0006227

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014117, upper bound: 0.0014459
time: 1.54 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014175, upper bound: 0.0014459
time: 2.01 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0090767, -0.0012231, -0.0090038, -0.0010693, -0.0060861, 0.0057775
1: -0.0054977, -0.0032835, -0.0054772, -0.0032401, -0.0017159, 0.0016289
2: -0.0020036, 0.0143337, -0.0018519, 0.0146534, -0.0126602, 0.0120184
3: 0.0013621, 0.0035241, 0.0013822, 0.0035664, -0.0016754, 0.0015904
4: -0.0046202, 0.0075892, -0.0048592, 0.0074759, -0.0089818, 0.0094615
5: 0.9942226, 0.9976147, 0.9941562, 0.9975833, -0.0024954, 0.0026287
6: 0.0026395, 0.0057186, 0.0025792, 0.0056900, -0.0022651, 0.0023860
7: -0.0135313, -0.0020408, -0.0137562, -0.0021475, -0.0084529, 0.0089043
8: -0.0076045, 0.0013385, -0.0075215, 0.0015136, -0.0069302, 0.0065789
9: -0.0041252, -0.0033537, -0.0041403, -0.0033608, -0.0005676, 0.0005979

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014117, upper bound: 0.0014531
time: 2.19 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014175, upper bound: 0.0014530
time: 2.34 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0090876, -0.0011104, -0.0086892, -0.0009512, -0.0064283, 0.0056749
1: -0.0055008, -0.0032517, -0.0053885, -0.0032068, -0.0018124, 0.0016000
2: -0.0020262, 0.0145679, -0.0011975, 0.0148991, -0.0133723, 0.0118048
3: 0.0013592, 0.0035551, 0.0014688, 0.0035990, -0.0017696, 0.0015622
4: -0.0047953, 0.0076062, -0.0050428, 0.0069868, -0.0088222, 0.0099936
5: 0.9941739, 0.9976195, 0.9941052, 0.9974474, -0.0024511, 0.0027765
6: 0.0025954, 0.0057228, 0.0025329, 0.0055666, -0.0022248, 0.0025202
7: -0.0136960, -0.0020249, -0.0139290, -0.0026078, -0.0083027, 0.0094051
8: -0.0076169, 0.0014668, -0.0071632, 0.0016481, -0.0073200, 0.0064620
9: -0.0041363, -0.0033526, -0.0041519, -0.0033917, -0.0005575, 0.0006315

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014195, upper bound: 0.0014459
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014244, upper bound: 0.0014459
time: 2.10 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0090876, -0.0011104, -0.0090042, -0.0010273, -0.0061921, 0.0058266
1: -0.0055008, -0.0032517, -0.0054773, -0.0032283, -0.0017458, 0.0016427
2: -0.0020262, 0.0145679, -0.0018526, 0.0147409, -0.0128808, 0.0121204
3: 0.0013592, 0.0035551, 0.0013821, 0.0035780, -0.0017046, 0.0016039
4: -0.0047953, 0.0076062, -0.0049246, 0.0074764, -0.0090581, 0.0096263
5: 0.9941739, 0.9976195, 0.9941381, 0.9975834, -0.0025166, 0.0026745
6: 0.0025954, 0.0057228, 0.0025628, 0.0056901, -0.0022843, 0.0024276
7: -0.0136960, -0.0020249, -0.0138177, -0.0021470, -0.0085246, 0.0090595
8: -0.0076169, 0.0014668, -0.0075219, 0.0015615, -0.0070510, 0.0066347
9: -0.0041363, -0.0033526, -0.0041445, -0.0033608, -0.0005724, 0.0006083

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014195, upper bound: 0.0014530
time: 2.08 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014244, upper bound: 0.0014530
time: 2.12 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 5.79 seconds
NS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0013925, upper bound: 0.0014442
NS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0013925, upper bound: 0.0014442
NS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0013925, upper bound: 0.0014529
NS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0013974, upper bound: 0.0014529
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014007, upper bound: 0.0014442
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014080, upper bound: 0.0014442
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014007, upper bound: 0.0014529
NS_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014081, upper bound: 0.0014529
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014117, upper bound: 0.0014459
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014175, upper bound: 0.0014459
NS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014117, upper bound: 0.0014531
NS_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014175, upper bound: 0.0014530
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014195, upper bound: 0.0014459
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014244, upper bound: 0.0014459
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014195, upper bound: 0.0014530
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 5.79
Output dim: 5, lower bound: -0.0014244, upper bound: 0.0014530

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.32 + 117.91 = 122.23 seconds
