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
execution time: IAR + RelationalAnalysis = 2.01 + 2.90 = 4.91 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0015904, upper bound: 0.0015903

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015478, upper bound: 0.0015255
time: 1.87 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015525
time: 1.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.01 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.01
Output dim: 5, lower bound: -0.0015478, upper bound: 0.0015255
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.01
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015525

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

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015186, upper bound: 0.0014820
time: 1.83 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015187, upper bound: 0.0014973
time: 1.86 seconds

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

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015235, upper bound: 0.0015096
time: 1.95 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015235, upper bound: 0.0015235
time: 1.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.34 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 5, lower bound: -0.0015186, upper bound: 0.0014820
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 5, lower bound: -0.0015187, upper bound: 0.0014973
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 5, lower bound: -0.0015235, upper bound: 0.0015096
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.34
Output dim: 5, lower bound: -0.0015235, upper bound: 0.0015235

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -0.0085547, -0.0008703, -0.0089602, -0.0009512, -0.0058574, 0.0063275
1: -0.0053506, -0.0031840, -0.0054649, -0.0032068, -0.0016514, 0.0017840
2: -0.0009177, 0.0150674, -0.0017613, 0.0148992, -0.0121845, 0.0131625
3: 0.0015058, 0.0036212, 0.0013942, 0.0035990, -0.0016124, 0.0017419
4: -0.0051686, 0.0067777, -0.0050429, 0.0074081, -0.0098369, 0.0091060
5: 0.9940703, 0.9973893, 0.9941052, 0.9975644, -0.0027330, 0.0025299
6: 0.0025012, 0.0055139, 0.0025329, 0.0056729, -0.0024807, 0.0022964
7: -0.0140473, -0.0028046, -0.0139291, -0.0022113, -0.0092576, 0.0085697
8: -0.0070101, 0.0017402, -0.0074718, 0.0016482, -0.0066698, 0.0072052
9: -0.0041599, -0.0034049, -0.0041519, -0.0033651, -0.0006216, 0.0005754

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014857, upper bound: 0.0014575
time: 1.91 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014917, upper bound: 0.0014575
time: 1.51 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -0.0086900, -0.0008827, -0.0090300, -0.0009343, -0.0059312, 0.0064406
1: -0.0053887, -0.0031875, -0.0054846, -0.0032021, -0.0016722, 0.0018158
2: -0.0011991, 0.0150417, -0.0019065, 0.0149343, -0.0123380, 0.0133977
3: 0.0014686, 0.0036178, 0.0013750, 0.0036036, -0.0016327, 0.0017730
4: -0.0051493, 0.0069880, -0.0050691, 0.0075166, -0.0100126, 0.0092207
5: 0.9940756, 0.9974477, 0.9940979, 0.9975946, -0.0027818, 0.0025618
6: 0.0025061, 0.0055669, 0.0025263, 0.0057003, -0.0025250, 0.0023253
7: -0.0140292, -0.0026066, -0.0139537, -0.0021092, -0.0094230, 0.0086777
8: -0.0071641, 0.0017261, -0.0075513, 0.0016674, -0.0067539, 0.0073339
9: -0.0041587, -0.0033917, -0.0041536, -0.0033582, -0.0006327, 0.0005827

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014859, upper bound: 0.0014713
time: 1.90 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014918, upper bound: 0.0014713
time: 1.77 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0088698, -0.0009451, -0.0091173, -0.0009361, -0.0060504, 0.0063578
1: -0.0054394, -0.0032051, -0.0055092, -0.0032026, -0.0017058, 0.0017925
2: -0.0015732, 0.0149119, -0.0020881, 0.0149306, -0.0125861, 0.0132255
3: 0.0014191, 0.0036006, 0.0013510, 0.0036031, -0.0016656, 0.0017502
4: -0.0050523, 0.0072676, -0.0050663, 0.0076524, -0.0098839, 0.0094061
5: 0.9941025, 0.9975254, 0.9940987, 0.9976323, -0.0027460, 0.0026133
6: 0.0025305, 0.0056375, 0.0025270, 0.0057345, -0.0024926, 0.0023721
7: -0.0139380, -0.0023435, -0.0139511, -0.0019814, -0.0093019, 0.0088522
8: -0.0073689, 0.0016551, -0.0076507, 0.0016653, -0.0068897, 0.0072397
9: -0.0041525, -0.0033740, -0.0041534, -0.0033497, -0.0006246, 0.0005944

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A1_A1

### Relational analysis result of NS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014907, upper bound: 0.0014850
time: 1.44 seconds

## Relational analysis of NS_A2_A1_A2

### Relational analysis result of NS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014967, upper bound: 0.0014850
time: 1.82 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0090049, -0.0009574, -0.0091873, -0.0009192, -0.0061030, 0.0064368
1: -0.0054775, -0.0032086, -0.0055289, -0.0031978, -0.0017207, 0.0018148
2: -0.0018543, 0.0148862, -0.0022337, 0.0149657, -0.0126955, 0.0133899
3: 0.0013819, 0.0035972, 0.0013317, 0.0036078, -0.0016800, 0.0017719
4: -0.0050332, 0.0074776, -0.0050925, 0.0077612, -0.0100068, 0.0094878
5: 0.9941078, 0.9975838, 0.9940913, 0.9976625, -0.0027802, 0.0026360
6: 0.0025354, 0.0056904, 0.0025204, 0.0057619, -0.0025236, 0.0023927
7: -0.0139199, -0.0021458, -0.0139758, -0.0018790, -0.0094175, 0.0089291
8: -0.0075227, 0.0016410, -0.0077304, 0.0016845, -0.0069495, 0.0073297
9: -0.0041513, -0.0033607, -0.0041551, -0.0033428, -0.0006324, 0.0005996

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_A2_A1

### Relational analysis result of NS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014907, upper bound: 0.0014967
time: 1.37 seconds

## Relational analysis of NS_A2_A2_A2

### Relational analysis result of NS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014967, upper bound: 0.0014967
time: 2.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.94 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014857, upper bound: 0.0014575
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014917, upper bound: 0.0014575
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014859, upper bound: 0.0014713
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014918, upper bound: 0.0014713
NS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014907, upper bound: 0.0014850
NS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014967, upper bound: 0.0014850
NS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014907, upper bound: 0.0014967
NS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 4.94
Output dim: 5, lower bound: -0.0014967, upper bound: 0.0014967

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0085310, -0.0011662, -0.0089591, -0.0010631, -0.0056893, 0.0060088
1: -0.0053439, -0.0032674, -0.0054646, -0.0032384, -0.0016040, 0.0016941
2: -0.0008684, 0.0144520, -0.0017590, 0.0146664, -0.0118349, 0.0124996
3: 0.0015124, 0.0035398, 0.0013945, 0.0035682, -0.0015662, 0.0016541
4: -0.0047086, 0.0067408, -0.0048689, 0.0074064, -0.0093414, 0.0088447
5: 0.9941981, 0.9973790, 0.9941536, 0.9975638, -0.0025953, 0.0024573
6: 0.0026172, 0.0055046, 0.0025768, 0.0056725, -0.0023558, 0.0022305
7: -0.0136145, -0.0028393, -0.0137653, -0.0022129, -0.0087913, 0.0083238
8: -0.0069831, 0.0014033, -0.0074706, 0.0015207, -0.0064784, 0.0068423
9: -0.0041308, -0.0034073, -0.0041409, -0.0033652, -0.0005903, 0.0005589

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014377, upper bound: 0.0014007
time: 1.51 seconds

## Relational analysis of NS_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014377, upper bound: 0.0014080
time: 1.84 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0085528, -0.0010400, -0.0089594, -0.0010209, -0.0057932, 0.0060678
1: -0.0053500, -0.0032319, -0.0054647, -0.0032265, -0.0016333, 0.0017107
2: -0.0009137, 0.0147145, -0.0017596, 0.0147542, -0.0120511, 0.0126222
3: 0.0015064, 0.0035745, 0.0013944, 0.0035798, -0.0015948, 0.0016703
4: -0.0049049, 0.0067747, -0.0049345, 0.0074069, -0.0094330, 0.0090062
5: 0.9941435, 0.9973885, 0.9941353, 0.9975641, -0.0026208, 0.0025022
6: 0.0025677, 0.0055131, 0.0025603, 0.0056726, -0.0023789, 0.0022712
7: -0.0137991, -0.0028074, -0.0138270, -0.0022124, -0.0088775, 0.0084759
8: -0.0070079, 0.0015470, -0.0074709, 0.0015687, -0.0065968, 0.0069094
9: -0.0041432, -0.0034051, -0.0041451, -0.0033652, -0.0005961, 0.0005691

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A1_A1_A2_A1

### Relational analysis result of NS_A1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014401, upper bound: 0.0014080
time: 1.54 seconds

## Relational analysis of NS_A1_A1_A2_A2

### Relational analysis result of NS_A1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014442, upper bound: 0.0014081
time: 1.94 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0086731, -0.0011713, -0.0090288, -0.0010462, -0.0057633, 0.0061334
1: -0.0053839, -0.0032689, -0.0054842, -0.0032336, -0.0016249, 0.0017292
2: -0.0011639, 0.0144413, -0.0019040, 0.0147016, -0.0119888, 0.0127587
3: 0.0014733, 0.0035384, 0.0013753, 0.0035728, -0.0015865, 0.0016884
4: -0.0047007, 0.0069617, -0.0048952, 0.0075148, -0.0095351, 0.0089597
5: 0.9942002, 0.9974405, 0.9941462, 0.9975941, -0.0026491, 0.0024893
6: 0.0026192, 0.0055603, 0.0025702, 0.0056998, -0.0024046, 0.0022595
7: -0.0136070, -0.0026314, -0.0137900, -0.0021109, -0.0089735, 0.0084321
8: -0.0071448, 0.0013975, -0.0075500, 0.0015400, -0.0065627, 0.0069841
9: -0.0041303, -0.0033933, -0.0041426, -0.0033584, -0.0006026, 0.0005662

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A1_A2_A1_B1

### Relational analysis result of NS_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014196
time: 1.52 seconds

## Relational analysis of NS_A1_A2_A1_B2

### Relational analysis result of NS_A1_A2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014244
time: 2.07 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0086880, -0.0010563, -0.0090292, -0.0010039, -0.0058671, 0.0061829
1: -0.0053881, -0.0032365, -0.0054843, -0.0032217, -0.0016542, 0.0017432
2: -0.0011949, 0.0146806, -0.0019048, 0.0147896, -0.0122048, 0.0128618
3: 0.0014692, 0.0035700, 0.0013752, 0.0035845, -0.0016151, 0.0017020
4: -0.0048795, 0.0069848, -0.0049609, 0.0075154, -0.0096121, 0.0091211
5: 0.9941505, 0.9974468, 0.9941279, 0.9975943, -0.0026705, 0.0025341
6: 0.0025741, 0.0055661, 0.0025536, 0.0056999, -0.0024240, 0.0023002
7: -0.0137753, -0.0026096, -0.0138519, -0.0021104, -0.0090460, 0.0085840
8: -0.0071618, 0.0015285, -0.0075504, 0.0015881, -0.0066809, 0.0070405
9: -0.0041416, -0.0033919, -0.0041468, -0.0033583, -0.0006074, 0.0005764

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A1_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014195
time: 1.93 seconds

## Relational analysis of NS_A1_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014244
time: 1.92 seconds

## BFS NS instance: NS_A2_A1_A1

### Backsubstitution after applying NS history:
0: -0.0088500, -0.0012453, -0.0091162, -0.0010480, -0.0058894, 0.0060528
1: -0.0054338, -0.0032898, -0.0055089, -0.0032341, -0.0016604, 0.0017065
2: -0.0015319, 0.0142873, -0.0020858, 0.0146979, -0.0122511, 0.0125911
3: 0.0014246, 0.0035180, 0.0013513, 0.0035723, -0.0016212, 0.0016662
4: -0.0045856, 0.0072367, -0.0048924, 0.0076506, -0.0094098, 0.0091557
5: 0.9942322, 0.9975168, 0.9941469, 0.9976318, -0.0026143, 0.0025437
6: 0.0026483, 0.0056297, 0.0025709, 0.0057340, -0.0023730, 0.0023089
7: -0.0134987, -0.0023726, -0.0137874, -0.0019830, -0.0088556, 0.0086166
8: -0.0073463, 0.0013132, -0.0076495, 0.0015379, -0.0067063, 0.0068924
9: -0.0041230, -0.0033759, -0.0041424, -0.0033498, -0.0005946, 0.0005786

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A2_A1_A1_B1

### Relational analysis result of NS_A2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014343
time: 1.75 seconds

## Relational analysis of NS_A2_A1_A1_B2

### Relational analysis result of NS_A2_A1_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014416
time: 1.80 seconds

## BFS NS instance: NS_A2_A1_A2

### Backsubstitution after applying NS history:
0: -0.0088679, -0.0011145, -0.0091166, -0.0010059, -0.0059872, 0.0061170
1: -0.0054389, -0.0032529, -0.0055090, -0.0032223, -0.0016880, 0.0017246
2: -0.0015693, 0.0145594, -0.0020864, 0.0147854, -0.0124545, 0.0127247
3: 0.0014196, 0.0035540, 0.0013512, 0.0035839, -0.0016482, 0.0016839
4: -0.0047889, 0.0072646, -0.0049578, 0.0076511, -0.0095096, 0.0093077
5: 0.9941757, 0.9975245, 0.9941288, 0.9976320, -0.0026421, 0.0025860
6: 0.0025970, 0.0056367, 0.0025544, 0.0057342, -0.0023982, 0.0023473
7: -0.0136900, -0.0023463, -0.0138490, -0.0019826, -0.0089496, 0.0087596
8: -0.0073667, 0.0014621, -0.0076498, 0.0015858, -0.0068176, 0.0069655
9: -0.0041359, -0.0033742, -0.0041466, -0.0033497, -0.0006010, 0.0005882

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 90
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 90

## Relational analysis of NS_A2_A1_A2_A1

### Relational analysis result of NS_A2_A1_A2_A1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014479, upper bound: 0.0014416
time: 2.05 seconds

## Relational analysis of NS_A2_A1_A2_A2

### Relational analysis result of NS_A2_A1_A2_A2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014529, upper bound: 0.0014416
time: 1.77 seconds

## BFS NS instance: NS_A2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0089950, -0.0012474, -0.0091862, -0.0010310, -0.0059379, 0.0061383
1: -0.0054747, -0.0032904, -0.0055286, -0.0032293, -0.0016741, 0.0017306
2: -0.0018335, 0.0142829, -0.0022312, 0.0147331, -0.0123521, 0.0127689
3: 0.0013847, 0.0035174, 0.0013320, 0.0035770, -0.0016346, 0.0016898
4: -0.0045823, 0.0074621, -0.0049187, 0.0077593, -0.0095427, 0.0092312
5: 0.9942331, 0.9975795, 0.9941397, 0.9976620, -0.0026512, 0.0025647
6: 0.0026491, 0.0056865, 0.0025642, 0.0057615, -0.0024065, 0.0023280
7: -0.0134956, -0.0021604, -0.0138122, -0.0018807, -0.0089807, 0.0086876
8: -0.0075114, 0.0013108, -0.0077291, 0.0015572, -0.0067616, 0.0069897
9: -0.0041228, -0.0033617, -0.0041441, -0.0033429, -0.0006030, 0.0005834

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A2_A2_A1_B1

### Relational analysis result of NS_A2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014480
time: 2.06 seconds

## Relational analysis of NS_A2_A2_A1_B2

### Relational analysis result of NS_A2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014530
time: 1.98 seconds

## BFS NS instance: NS_A2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0090029, -0.0011345, -0.0091865, -0.0009889, -0.0060426, 0.0061960
1: -0.0054769, -0.0032585, -0.0055287, -0.0032175, -0.0017036, 0.0017469
2: -0.0018500, 0.0145178, -0.0022320, 0.0148208, -0.0125698, 0.0128889
3: 0.0013825, 0.0035485, 0.0013319, 0.0035886, -0.0016634, 0.0017056
4: -0.0047578, 0.0074745, -0.0049843, 0.0077599, -0.0096324, 0.0093939
5: 0.9941844, 0.9975829, 0.9941214, 0.9976622, -0.0026762, 0.0026099
6: 0.0026048, 0.0056896, 0.0025477, 0.0057616, -0.0024291, 0.0023690
7: -0.0136608, -0.0021488, -0.0138739, -0.0018802, -0.0090651, 0.0088407
8: -0.0075204, 0.0014393, -0.0077295, 0.0016052, -0.0068807, 0.0070554
9: -0.0041339, -0.0033609, -0.0041482, -0.0033429, -0.0006087, 0.0005936

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 90
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 207
type: A, layer: 1, pos: 207
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 235
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 177
type: B, layer: 1, pos: 177
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 90

## Relational analysis of NS_A2_A2_A2_B1

### Relational analysis result of NS_A2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014480
time: 1.97 seconds

## Relational analysis of NS_A2_A2_A2_B2

### Relational analysis result of NS_A2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014530
time: 1.97 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.00 seconds
NS_A1_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014377, upper bound: 0.0014007
NS_A1_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014377, upper bound: 0.0014080
NS_A1_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014401, upper bound: 0.0014080
NS_A1_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014442, upper bound: 0.0014081
NS_A1_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014196
NS_A1_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014244
NS_A1_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014195
NS_A1_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014459, upper bound: 0.0014244
NS_A2_A1_A1_B1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014343
NS_A2_A1_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014416
NS_A2_A1_A2_A1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014479, upper bound: 0.0014416
NS_A2_A1_A2_A2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014529, upper bound: 0.0014416
NS_A2_A2_A1_B1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014480
NS_A2_A2_A1_B2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014460, upper bound: 0.0014530
NS_A2_A2_A2_B1, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014480
NS_A2_A2_A2_B2, status: Status.VERIFIED, split count: 4, time: 6.00
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014530

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 4.91 + 78.63 = 83.54 seconds
