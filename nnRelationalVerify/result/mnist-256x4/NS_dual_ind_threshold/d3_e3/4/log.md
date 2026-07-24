## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.51975288


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.4363009, 1.0305700, 0.4363009, 1.0305700, -0.5942692, 0.5942692)
1: (-0.0946363, 0.1140718, -0.0946363, 0.1140718, -0.2087081, 0.2087081)
2: (-0.0521084, 0.1664880, -0.0521084, 0.1664880, -0.2185965, 0.2185965)
3: (-0.0998731, 0.1264820, -0.0998731, 0.1264820, -0.2263550, 0.2263550)
4: (-0.1233485, 0.0905924, -0.1233485, 0.0905924, -0.2139409, 0.2139409)
5: (-0.1413220, 0.2247675, -0.1413220, 0.2247675, -0.3660894, 0.3660894)
6: (-0.0816977, 0.1669606, -0.0816977, 0.1669606, -0.2486582, 0.2486582)
7: (-0.1244795, 0.2396419, -0.1244795, 0.2396419, -0.3641213, 0.3641213)
8: (-0.1047466, 0.1465420, -0.1047466, 0.1465420, -0.2512886, 0.2512886)
9: (-0.1048256, 0.1604646, -0.1048256, 0.1604646, -0.2652901, 0.2652901)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.20 + 3.26 = 5.46 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5775032, upper bound: 0.5775021

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5579421, upper bound: 0.5630132
time: 2.14 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5630417, upper bound: 0.5630407
time: 2.07 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.41
Output dim: 0, lower bound: -0.5579421, upper bound: 0.5630132
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.41
Output dim: 0, lower bound: -0.5630417, upper bound: 0.5630407

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: 0.7872287, 1.0059248, 0.4722158, 1.0283251, -0.2410964, 0.5337090
1: -0.0444004, 0.0352892, -0.0875494, 0.1060424, -0.1504428, 0.1228386
2: -0.0110953, 0.0830213, -0.0468349, 0.1566593, -0.1677546, 0.1298561
3: -0.0323186, 0.0679883, -0.0927515, 0.1199680, -0.1522866, 0.1607398
4: -0.0618360, 0.0242187, -0.1163393, 0.0829427, -0.1447787, 0.1405579
5: -0.0354133, 0.1518373, -0.1295023, 0.2171720, -0.2525853, 0.2813396
6: -0.0413472, 0.0512399, -0.0772902, 0.1535628, -0.1949099, 0.1285302
7: -0.0700250, 0.0528942, -0.1183179, 0.2197753, -0.2898003, 0.1712120
8: -0.0445021, 0.0713721, -0.0970791, 0.1380233, -0.1825253, 0.1684512
9: -0.0472585, 0.0594928, -0.0981368, 0.1493395, -0.1965980, 0.1576296

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5574165, upper bound: 0.5574165
time: 2.25 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5574165, upper bound: 0.5630121
time: 2.01 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: 0.7796468, 1.0071807, 0.5822107, 1.0208312, -0.2411845, 0.4249700
1: -0.0419093, 0.0329257, -0.0648279, 0.0769708, -0.1188801, 0.0977536
2: -0.0121522, 0.0791336, -0.0289400, 0.1222959, -0.1344480, 0.1080736
3: -0.0309005, 0.0660053, -0.0698524, 0.0956648, -0.1265653, 0.1358576
4: -0.0578032, 0.0214984, -0.0933135, 0.0556732, -0.1134764, 0.1148119
5: -0.0322095, 0.1489199, -0.0896282, 0.1917535, -0.2239630, 0.2385481
6: -0.0386778, 0.0517899, -0.0608610, 0.1136293, -0.1523071, 0.1126509
7: -0.0652398, 0.0562240, -0.0950978, 0.1565235, -0.2217633, 0.1513218
8: -0.0414073, 0.0686763, -0.0692926, 0.1121410, -0.1535482, 0.1379689
9: -0.0446797, 0.0532376, -0.0746276, 0.1102251, -0.1549048, 0.1278652

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5529614, upper bound: 0.5444608
time: 2.24 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5532784, upper bound: 0.5532784
time: 2.01 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 6.94 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 0, lower bound: -0.5574165, upper bound: 0.5574165
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 0, lower bound: -0.5574165, upper bound: 0.5630121
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 0, lower bound: -0.5529614, upper bound: 0.5444608
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 0, lower bound: -0.5532784, upper bound: 0.5532784

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: 0.7872287, 1.0059248, 0.7872287, 1.0059248, -0.2186961, 0.2186961
1: -0.0444004, 0.0352892, -0.0444004, 0.0352892, -0.0796896, 0.0796896
2: -0.0110953, 0.0830213, -0.0110953, 0.0830213, -0.0941166, 0.0941166
3: -0.0323186, 0.0679883, -0.0323186, 0.0679883, -0.1003069, 0.1003069
4: -0.0618360, 0.0242187, -0.0618360, 0.0242187, -0.0860547, 0.0860547
5: -0.0354133, 0.1518373, -0.0354133, 0.1518373, -0.1872506, 0.1872506
6: -0.0413472, 0.0512399, -0.0413472, 0.0512399, -0.0925871, 0.0925871
7: -0.0700250, 0.0528942, -0.0700250, 0.0528942, -0.1229191, 0.1229191
8: -0.0445021, 0.0713721, -0.0445021, 0.0713721, -0.1158741, 0.1158741
9: -0.0472585, 0.0594928, -0.0472585, 0.0594928, -0.1067513, 0.1067513

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5576176, upper bound: 0.5567710
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5570172, upper bound: 0.5568135
time: 2.49 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: 0.7872287, 1.0059248, 0.7796468, 1.0071807, -0.2199520, 0.2262781
1: -0.0444004, 0.0352892, -0.0419093, 0.0329257, -0.0773261, 0.0771985
2: -0.0110953, 0.0830213, -0.0121522, 0.0791336, -0.0902289, 0.0951734
3: -0.0323186, 0.0679883, -0.0309005, 0.0660053, -0.0983239, 0.0988888
4: -0.0618360, 0.0242187, -0.0578032, 0.0214984, -0.0833344, 0.0820219
5: -0.0354133, 0.1518373, -0.0322095, 0.1489199, -0.1843332, 0.1840468
6: -0.0413472, 0.0512399, -0.0386778, 0.0517899, -0.0931371, 0.0899177
7: -0.0700250, 0.0528942, -0.0652398, 0.0562240, -0.1262490, 0.1181340
8: -0.0445021, 0.0713721, -0.0414073, 0.0686763, -0.1131784, 0.1127793
9: -0.0472585, 0.0594928, -0.0446797, 0.0532376, -0.1004961, 0.1041726

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5576176, upper bound: 0.5627411
time: 2.39 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5570172, upper bound: 0.5627454
time: 2.12 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: 0.8394343, 1.0061541, 0.8318627, 1.0056251, -0.1661908, 0.1742914
1: -0.0384367, 0.0194921, -0.0405879, 0.0247676, -0.0632043, 0.0600800
2: -0.0114821, 0.0662486, -0.0107836, 0.0718399, -0.0833220, 0.0770323
3: -0.0204285, 0.0586012, -0.0236310, 0.0620133, -0.0824418, 0.0822322
4: -0.0479844, 0.0164969, -0.0532674, 0.0186319, -0.0666164, 0.0697643
5: -0.0179585, 0.1365673, -0.0226503, 0.1419868, -0.1599452, 0.1592176
6: -0.0338823, 0.0352491, -0.0366453, 0.0387052, -0.0725874, 0.0718944
7: -0.0575014, 0.0314422, -0.0630385, 0.0345770, -0.0920784, 0.0944806
8: -0.0358113, 0.0555801, -0.0392644, 0.0603712, -0.0961825, 0.0948445
9: -0.0375096, 0.0417242, -0.0413611, 0.0475916, -0.0851011, 0.0830853

Time for backsubstitution: 2.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 11

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526867, upper bound: 0.5441826
time: 2.28 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526867, upper bound: 0.5442188
time: 2.33 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: 0.8380226, 1.0061599, 0.7724793, 1.0062048, -0.1681822, 0.2336806
1: -0.0385854, 0.0200403, -0.0460297, 0.0391992, -0.0777846, 0.0660700
2: -0.0114887, 0.0667626, -0.0119574, 0.0872734, -0.0987621, 0.0787200
3: -0.0207424, 0.0589393, -0.0355494, 0.0702524, -0.0909949, 0.0944887
4: -0.0483924, 0.0166777, -0.0652560, 0.0264901, -0.0748825, 0.0819337
5: -0.0184403, 0.1371136, -0.0400948, 0.1555204, -0.1739607, 0.1772085
6: -0.0341191, 0.0356811, -0.0432932, 0.0556060, -0.0897251, 0.0789743
7: -0.0579197, 0.0320181, -0.0730441, 0.0589024, -0.1168220, 0.1050622
8: -0.0360240, 0.0560851, -0.0466886, 0.0754164, -0.1114404, 0.1027738
9: -0.0378698, 0.0420696, -0.0495793, 0.0643504, -0.1022202, 0.0916489

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5530215, upper bound: 0.5529927
time: 2.00 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5530213, upper bound: 0.5530213
time: 2.88 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 7.34 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5576176, upper bound: 0.5567710
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5570172, upper bound: 0.5568135
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5576176, upper bound: 0.5627411
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5570172, upper bound: 0.5627454
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5526867, upper bound: 0.5441826
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5526867, upper bound: 0.5442188
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5530215, upper bound: 0.5529927
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 7.34
Output dim: 0, lower bound: -0.5530213, upper bound: 0.5530213

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8312534, 1.0050144, 0.7984951, 1.0056951, -0.1744417, 0.2065194
1: -0.0362795, 0.0151266, -0.0416288, 0.0302396, -0.0665191, 0.0567554
2: -0.0099637, 0.0616470, -0.0108155, 0.0768992, -0.0868630, 0.0724626
3: -0.0191522, 0.0552621, -0.0284125, 0.0647817, -0.0839338, 0.0836746
4: -0.0433445, 0.0146288, -0.0565338, 0.0206489, -0.0639933, 0.0711626
5: -0.0153581, 0.1314663, -0.0291745, 0.1467146, -0.1620726, 0.1606408
6: -0.0307498, 0.0356030, -0.0381675, 0.0471154, -0.0778652, 0.0737705
7: -0.0511452, 0.0349576, -0.0648156, 0.0484972, -0.0996423, 0.0997732
8: -0.0329977, 0.0519618, -0.0408883, 0.0659999, -0.0989977, 0.0928501
9: -0.0332891, 0.0380912, -0.0437258, 0.0517410, -0.0850301, 0.0818171

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394599, upper bound: 0.5505533
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394016, upper bound: 0.5386521
time: 1.97 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.7777218, 1.0053495, 0.8010582, 1.0056857, -0.2279639, 0.2042913
1: -0.0501968, 0.0441914, -0.0408103, 0.0288405, -0.0790372, 0.0850017
2: -0.0144838, 0.0945748, -0.0108089, 0.0751572, -0.0896409, 0.1053838
3: -0.0389163, 0.0739971, -0.0273454, 0.0638718, -0.1027881, 0.1013425
4: -0.0723479, 0.0313925, -0.0549983, 0.0196109, -0.0919589, 0.0863909
5: -0.0465975, 0.1612530, -0.0274311, 0.1452706, -0.1918681, 0.1886841
6: -0.0478470, 0.0564767, -0.0372363, 0.0460805, -0.0939275, 0.0937130
7: -0.0809997, 0.0559793, -0.0632729, 0.0475296, -0.1285293, 0.1192522
8: -0.0519373, 0.0809777, -0.0398275, 0.0645002, -0.1164374, 0.1208052
9: -0.0541734, 0.0751528, -0.0427074, 0.0494811, -0.1036545, 0.1178602

Time for backsubstitution: 2.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5387579, upper bound: 0.5504704
time: 1.98 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5387002, upper bound: 0.5386991
time: 2.53 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8312534, 1.0050144, 0.7916514, 1.0064533, -0.1751999, 0.2133631
1: -0.0362795, 0.0151266, -0.0393693, 0.0275422, -0.0638217, 0.0544959
2: -0.0099637, 0.0616470, -0.0116194, 0.0729656, -0.0829294, 0.0732664
3: -0.0191522, 0.0552621, -0.0269226, 0.0625750, -0.0817271, 0.0821847
4: -0.0433445, 0.0146288, -0.0521529, 0.0185652, -0.0619096, 0.0667817
5: -0.0153581, 0.1314663, -0.0263323, 0.1434162, -0.1587743, 0.1577986
6: -0.0307498, 0.0356030, -0.0356873, 0.0472589, -0.0780087, 0.0712903
7: -0.0511452, 0.0349576, -0.0597337, 0.0514293, -0.1025745, 0.0946913
8: -0.0329977, 0.0519618, -0.0375649, 0.0633151, -0.0963129, 0.0895266
9: -0.0332891, 0.0380912, -0.0409135, 0.0457703, -0.0790594, 0.0790047

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5393556, upper bound: 0.5563066
time: 2.12 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5392998, upper bound: 0.5439052
time: 2.64 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.7777218, 1.0053495, 0.7933407, 1.0063710, -0.2286493, 0.2120088
1: -0.0501968, 0.0441914, -0.0389913, 0.0265052, -0.0767019, 0.0831827
2: -0.0144838, 0.0945748, -0.0115559, 0.0719061, -0.0863899, 0.1061307
3: -0.0389163, 0.0739971, -0.0263240, 0.0618923, -0.1008086, 0.1003212
4: -0.0723479, 0.0313925, -0.0511726, 0.0181962, -0.0905441, 0.0825652
5: -0.0465975, 0.1612530, -0.0254479, 0.1423319, -0.1889294, 0.1867009
6: -0.0478470, 0.0564767, -0.0351638, 0.0465700, -0.0944170, 0.0916405
7: -0.0809997, 0.0559793, -0.0586876, 0.0507395, -0.1317392, 0.1146669
8: -0.0519373, 0.0809777, -0.0369596, 0.0623765, -0.1143138, 0.1179373
9: -0.0541734, 0.0751528, -0.0401529, 0.0448003, -0.0989738, 0.1153058

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5387040, upper bound: 0.5561176
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5386495, upper bound: 0.5439192
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8670220, 1.0054381, 0.8386656, 1.0054388, -0.1384168, 0.1667725
1: -0.0320685, 0.0066773, -0.0386142, 0.0200355, -0.0521040, 0.0452914
2: -0.0105367, 0.0476012, -0.0105377, 0.0667681, -0.0773048, 0.0581389
3: -0.0137826, 0.0459845, -0.0207006, 0.0589527, -0.0727354, 0.0666851
4: -0.0343578, 0.0096255, -0.0484286, 0.0166884, -0.0510462, 0.0580541
5: -0.0142930, 0.1165282, -0.0183878, 0.1371256, -0.1514186, 0.1349160
6: -0.0253185, 0.0240102, -0.0341547, 0.0355623, -0.0608808, 0.0581649
7: -0.0418344, 0.0198512, -0.0580100, 0.0317651, -0.0735995, 0.0778612
8: -0.0270298, 0.0416590, -0.0360560, 0.0560613, -0.0830910, 0.0777150
9: -0.0232682, 0.0327115, -0.0379063, 0.0420907, -0.0653589, 0.0706178

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526854, upper bound: 0.5441801
time: 2.19 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5526854, upper bound: 0.5441801
time: 2.15 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.8325512, 1.0055453, 0.8403268, 1.0054393, -0.1728881, 0.1652185
1: -0.0400256, 0.0243589, -0.0382306, 0.0188609, -0.0588864, 0.0625895
2: -0.0106782, 0.0709009, -0.0105381, 0.0656453, -0.0763235, 0.0814390
3: -0.0228656, 0.0617490, -0.0201124, 0.0581929, -0.0810585, 0.0818614
4: -0.0519623, 0.0182113, -0.0474685, 0.0162746, -0.0682369, 0.0656798
5: -0.0218038, 0.1415671, -0.0174597, 0.1359189, -0.1577227, 0.1590268
6: -0.0363222, 0.0380532, -0.0335658, 0.0348854, -0.0712077, 0.0716190
7: -0.0620562, 0.0343340, -0.0569107, 0.0310672, -0.0931234, 0.0912447
8: -0.0380023, 0.0598886, -0.0355272, 0.0550213, -0.0930236, 0.0954158
9: -0.0410627, 0.0450062, -0.0370488, 0.0412985, -0.0823612, 0.0820550

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 11

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5523068, upper bound: 0.5440410
time: 1.99 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5523519, upper bound: 0.5440343
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.8664274, 1.0054400, 0.7840911, 1.0057667, -0.1393394, 0.2213489
1: -0.0322058, 0.0068628, -0.0432165, 0.0340840, -0.0662898, 0.0500793
2: -0.0105393, 0.0480032, -0.0108789, 0.0810638, -0.0916031, 0.0588821
3: -0.0138836, 0.0462565, -0.0315675, 0.0669892, -0.0808727, 0.0778240
4: -0.0346202, 0.0097736, -0.0598724, 0.0228597, -0.0574799, 0.0696459
5: -0.0142963, 0.1169602, -0.0337394, 0.1503137, -0.1646100, 0.1506996
6: -0.0254867, 0.0242525, -0.0400663, 0.0513729, -0.0768596, 0.0643188
7: -0.0421370, 0.0201010, -0.0677573, 0.0543797, -0.0965167, 0.0878583
8: -0.0272191, 0.0419137, -0.0430212, 0.0699542, -0.0971733, 0.0849349
9: -0.0235752, 0.0328496, -0.0459940, 0.0564752, -0.0800504, 0.0788436

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5314149, upper bound: 0.5456924
time: 1.96 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5313374, upper bound: 0.5313019
time: 2.00 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.8319507, 1.0055476, 0.7862376, 1.0057554, -0.1738047, 0.2193100
1: -0.0401642, 0.0247834, -0.0425110, 0.0328930, -0.0730572, 0.0672945
2: -0.0106812, 0.0713069, -0.0108705, 0.0795719, -0.0902531, 0.0821774
3: -0.0230782, 0.0620237, -0.0306599, 0.0662086, -0.0892868, 0.0926835
4: -0.0523094, 0.0183609, -0.0585525, 0.0219698, -0.0742792, 0.0769134
5: -0.0221393, 0.1420032, -0.0322464, 0.1490778, -0.1712171, 0.1742496
6: -0.0365351, 0.0382978, -0.0392648, 0.0505012, -0.0870363, 0.0775626
7: -0.0624535, 0.0345864, -0.0664273, 0.0535753, -0.1160288, 0.1010137
8: -0.0381934, 0.0602645, -0.0421078, 0.0686723, -0.1068657, 0.1023722
9: -0.0413727, 0.0452925, -0.0451194, 0.0545303, -0.0959030, 0.0904119

Time for backsubstitution: 2.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5314149, upper bound: 0.5456147
time: 2.29 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5313374, upper bound: 0.5313374
time: 2.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.50 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5394599, upper bound: 0.5505533
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5394016, upper bound: 0.5386521
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5387579, upper bound: 0.5504704
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5387002, upper bound: 0.5386991
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5393556, upper bound: 0.5563066
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5392998, upper bound: 0.5439052
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5387040, upper bound: 0.5561176
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5386495, upper bound: 0.5439192
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5526854, upper bound: 0.5441801
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5526854, upper bound: 0.5441801
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5523068, upper bound: 0.5440410
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5523519, upper bound: 0.5440343
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5314149, upper bound: 0.5456924
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5313374, upper bound: 0.5313019
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5314149, upper bound: 0.5456147
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 6.50
Output dim: 0, lower bound: -0.5313374, upper bound: 0.5313374

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.8312534, 1.0050144, 0.8157468, 1.0051644, -0.1739110, 0.1892676
1: -0.0362795, 0.0151266, -0.0389429, 0.0236425, -0.0599220, 0.0540695
2: -0.0099637, 0.0616470, -0.0101435, 0.0697602, -0.0797239, 0.0717905
3: -0.0191522, 0.0552621, -0.0236597, 0.0607227, -0.0798749, 0.0789218
4: -0.0433445, 0.0146288, -0.0501662, 0.0175735, -0.0609179, 0.0647950
5: -0.0153581, 0.1314663, -0.0223202, 0.1401681, -0.1555261, 0.1537865
6: -0.0307498, 0.0356030, -0.0348772, 0.0411934, -0.0719432, 0.0704802
7: -0.0511452, 0.0349576, -0.0587467, 0.0414129, -0.0925580, 0.0937043
8: -0.0329977, 0.0519618, -0.0367036, 0.0596123, -0.0926100, 0.0886653
9: -0.0332891, 0.0380912, -0.0393500, 0.0437550, -0.0770440, 0.0774413

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394016, upper bound: 0.5386521
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394016, upper bound: 0.5386521
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.8436391, 1.0046573, 0.5612040, 1.0170006, -0.1733615, 0.4434533
1: -0.0346415, 0.0118255, -0.0712627, 0.0949628, -0.1296043, 0.0830882
2: -0.0095050, 0.0563716, -0.0299286, 0.1502061, -0.1597112, 0.0863002
3: -0.0171246, 0.0517517, -0.0812399, 0.1040333, -0.1211578, 0.1329916
4: -0.0398449, 0.0127376, -0.1171794, 0.0598124, -0.0996573, 0.1299171
5: -0.0129715, 0.1258297, -0.1091665, 0.2099274, -0.2228989, 0.2349963
6: -0.0286120, 0.0314641, -0.0731646, 0.1165982, -0.1452102, 0.1046287
7: -0.0474114, 0.0297722, -0.1199478, 0.1438604, -0.1912718, 0.1497200
8: -0.0306932, 0.0480643, -0.0803338, 0.1354811, -0.1661744, 0.1283980
9: -0.0294629, 0.0359353, -0.0846772, 0.1385620, -0.1680248, 0.1206125

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5391884, upper bound: 0.5383465
time: 1.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5390143, upper bound: 0.5384181
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.7777218, 1.0053495, 0.8181052, 1.0051500, -0.2274282, 0.1872444
1: -0.0501968, 0.0441914, -0.0384902, 0.0222300, -0.0724268, 0.0826816
2: -0.0144838, 0.0945748, -0.0101275, 0.0684056, -0.0828894, 0.1047024
3: -0.0389163, 0.0739971, -0.0229272, 0.0598074, -0.0987237, 0.0969243
4: -0.0723479, 0.0313925, -0.0490172, 0.0170797, -0.0894276, 0.0804098
5: -0.0465975, 0.1612530, -0.0211726, 0.1387129, -0.1853103, 0.1824256
6: -0.0478470, 0.0564767, -0.0341782, 0.0403130, -0.0881600, 0.0906548
7: -0.0809997, 0.0559793, -0.0574523, 0.0404316, -0.1214313, 0.1134315
8: -0.0519373, 0.0809777, -0.0360759, 0.0583427, -0.1102800, 0.1170536
9: -0.0541734, 0.0751528, -0.0383255, 0.0428035, -0.0969769, 0.1134783

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5387002, upper bound: 0.5386991
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5387002, upper bound: 0.5386991
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.7911049, 1.0049680, 0.5637264, 1.0165063, -0.2254014, 0.4412416
1: -0.0472294, 0.0386205, -0.0704899, 0.0936942, -0.1409235, 0.1091103
2: -0.0124789, 0.0878950, -0.0294215, 0.1486143, -0.1610932, 0.1173165
3: -0.0345806, 0.0704903, -0.0802658, 0.1031768, -0.1377574, 0.1507561
4: -0.0666104, 0.0275428, -0.1157370, 0.0588700, -0.1254804, 0.1432799
5: -0.0397206, 0.1556377, -0.1075276, 0.2085956, -0.2483163, 0.2631653
6: -0.0444269, 0.0517726, -0.0722883, 0.1156785, -0.1601055, 0.1240609
7: -0.0754307, 0.0507763, -0.1184908, 0.1429038, -0.2183345, 0.1692671
8: -0.0480555, 0.0750589, -0.0793345, 0.1341029, -0.1821584, 0.1543934
9: -0.0503416, 0.0667949, -0.0837231, 0.1364301, -0.1867718, 0.1505180

Time for backsubstitution: 2.02 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5385075, upper bound: 0.5383833
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384655, upper bound: 0.5384655
time: 1.86 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.8312534, 1.0050144, 0.8101797, 1.0056833, -0.1744299, 0.1948347
1: -0.0362795, 0.0151266, -0.0374138, 0.0205663, -0.0568458, 0.0525404
2: -0.0099637, 0.0616470, -0.0108307, 0.0664197, -0.0763834, 0.0724777
3: -0.0191522, 0.0552621, -0.0227687, 0.0582890, -0.0774411, 0.0780308
4: -0.0433445, 0.0146288, -0.0468290, 0.0162516, -0.0595961, 0.0614578
5: -0.0153581, 0.1314663, -0.0202895, 0.1364836, -0.1518417, 0.1517558
6: -0.0307498, 0.0356030, -0.0326467, 0.0414243, -0.0721741, 0.0682497
7: -0.0511452, 0.0349576, -0.0542534, 0.0436967, -0.0948418, 0.0892110
8: -0.0329977, 0.0519618, -0.0347001, 0.0569361, -0.0899339, 0.0866618
9: -0.0332891, 0.0380912, -0.0363343, 0.0411208, -0.0744099, 0.0744255

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5392998, upper bound: 0.5439052
time: 1.77 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5392998, upper bound: 0.5439064
time: 1.69 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.8436391, 1.0046573, 0.5739431, 1.0141268, -0.1704877, 0.4307141
1: -0.0346415, 0.0118255, -0.0589999, 0.0887383, -0.1233798, 0.0708254
2: -0.0095050, 0.0563716, -0.0271116, 0.1350226, -0.1445276, 0.0834831
3: -0.0171246, 0.0517517, -0.0744398, 0.0994863, -0.1166108, 0.1261915
4: -0.0398449, 0.0127376, -0.1092277, 0.0404145, -0.0802594, 0.1219654
5: -0.0129715, 0.1258297, -0.0851362, 0.2044876, -0.2174591, 0.2109660
6: -0.0286120, 0.0314641, -0.0606720, 0.1178695, -0.1464814, 0.0921361
7: -0.0474114, 0.0297722, -0.1109458, 0.1412103, -0.1886218, 0.1407180
8: -0.0306932, 0.0480643, -0.0746371, 0.1212246, -0.1519178, 0.1227014
9: -0.0294629, 0.0359353, -0.0795086, 0.1113928, -0.1408557, 0.1154439

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5390716, upper bound: 0.5436457
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5389153, upper bound: 0.5437163
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.7777218, 1.0053495, 0.8118653, 1.0056447, -0.2279229, 0.1934842
1: -0.0501968, 0.0441914, -0.0370701, 0.0195176, -0.0697144, 0.0812615
2: -0.0144838, 0.0945748, -0.0107825, 0.0654024, -0.0798862, 0.1053573
3: -0.0389163, 0.0739971, -0.0222278, 0.0575993, -0.0965156, 0.0962249
4: -0.0723479, 0.0313925, -0.0459609, 0.0158819, -0.0882298, 0.0773534
5: -0.0465975, 0.1612530, -0.0194329, 0.1353886, -0.1819860, 0.1806859
6: -0.0478470, 0.0564767, -0.0321170, 0.0407853, -0.0886323, 0.0885937
7: -0.0809997, 0.0559793, -0.0532697, 0.0430051, -0.1240048, 0.1092490
8: -0.0519373, 0.0809777, -0.0342245, 0.0559868, -0.1079241, 0.1152022
9: -0.0541734, 0.0751528, -0.0355647, 0.0404030, -0.0945764, 0.1107175

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5386495, upper bound: 0.5439192
time: 1.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5386495, upper bound: 0.5439181
time: 2.14 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.7911049, 1.0049680, 0.5755749, 1.0135478, -0.2224429, 0.4293932
1: -0.0472294, 0.0386205, -0.0544827, 0.0878236, -0.1350529, 0.0931032
2: -0.0124789, 0.0878950, -0.0267418, 0.1283629, -0.1408418, 0.1146367
3: -0.0345806, 0.0704903, -0.0679742, 0.0988633, -0.1334439, 0.1384645
4: -0.0666104, 0.0275428, -0.0951350, 0.0400887, -0.1066992, 0.1226778
5: -0.0397206, 0.1556377, -0.0805493, 0.2035101, -0.2432308, 0.2361870
6: -0.0444269, 0.0517726, -0.0601960, 0.1126141, -0.1570411, 0.1119686
7: -0.0754307, 0.0507763, -0.1019429, 0.1405361, -0.2159669, 0.1527192
8: -0.0480555, 0.0750589, -0.0594277, 0.1203854, -0.1684408, 0.1344866
9: -0.0503416, 0.0667949, -0.0788132, 0.0828922, -0.1332338, 0.1456081

Time for backsubstitution: 2.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384521, upper bound: 0.5436563
time: 2.24 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384163, upper bound: 0.5437301
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.8670220, 1.0054381, 0.8584474, 1.0048623, -0.1378403, 0.1469907
1: -0.0320685, 0.0066773, -0.0340479, 0.0093517, -0.0414202, 0.0407251
2: -0.0105367, 0.0476012, -0.0097762, 0.0533970, -0.0639337, 0.0573774
3: -0.0137826, 0.0459845, -0.0152376, 0.0499060, -0.0636886, 0.0612222
4: -0.0343578, 0.0096255, -0.0381400, 0.0117611, -0.0461190, 0.0477655
5: -0.0142930, 0.1165282, -0.0133364, 0.1227566, -0.1370496, 0.1298646
6: -0.0253185, 0.0240102, -0.0277425, 0.0275034, -0.0528219, 0.0517527
7: -0.0418344, 0.0198512, -0.0461976, 0.0234538, -0.0652881, 0.0660488
8: -0.0270298, 0.0416590, -0.0297592, 0.0453313, -0.0723611, 0.0714182
9: -0.0232682, 0.0327115, -0.0276945, 0.0347028, -0.0579710, 0.0604060

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524572, upper bound: 0.5439109
time: 3.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5523488, upper bound: 0.5440009
time: 2.53 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.8670220, 1.0054381, 0.8225403, 1.0050406, -0.1380187, 0.1828977
1: -0.0320685, 0.0066773, -0.0423364, 0.0314374, -0.0635059, 0.0490137
2: -0.0105367, 0.0476012, -0.0104304, 0.0776677, -0.0882044, 0.0580317
3: -0.0137826, 0.0459845, -0.0264103, 0.0663273, -0.0801099, 0.0723948
4: -0.0343578, 0.0096255, -0.0577481, 0.0207047, -0.0550626, 0.0673736
5: -0.0142930, 0.1165282, -0.0273968, 0.1488387, -0.1631316, 0.1439250
6: -0.0253185, 0.0240102, -0.0398711, 0.0421314, -0.0674499, 0.0638813
7: -0.0418344, 0.0198512, -0.0686807, 0.0385402, -0.0803746, 0.0885319
8: -0.0270298, 0.0416590, -0.0411889, 0.0661549, -0.0931847, 0.0828479
9: -0.0232682, 0.0327115, -0.0462305, 0.0497797, -0.0730478, 0.0789420

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5524572, upper bound: 0.5439098
time: 2.51 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5523488, upper bound: 0.5440009
time: 2.12 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.8382477, 1.0052836, 0.8687410, 1.0046151, -0.1663674, 0.1365426
1: -0.0387106, 0.0203311, -0.0316717, 0.0061411, -0.0448517, 0.0520028
2: -0.0103326, 0.0670507, -0.0094498, 0.0464393, -0.0567719, 0.0765004
3: -0.0208486, 0.0591438, -0.0134909, 0.0451984, -0.0660470, 0.0726348
4: -0.0486702, 0.0167925, -0.0335997, 0.0091972, -0.0578674, 0.0503921
5: -0.0186214, 0.1374292, -0.0129259, 0.1152796, -0.1339009, 0.1503551
6: -0.0343029, 0.0357325, -0.0248327, 0.0233099, -0.0576128, 0.0605651
7: -0.0582866, 0.0319407, -0.0409596, 0.0191289, -0.0774156, 0.0729004
8: -0.0361890, 0.0563229, -0.0264826, 0.0409228, -0.0771119, 0.0828055
9: -0.0381221, 0.0422900, -0.0223807, 0.0323123, -0.0704344, 0.0646707

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366736, upper bound: 0.5168901
time: 1.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5219611, upper bound: 0.5168025
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.8357103, 1.0054331, 0.8516055, 1.0050014, -0.1692911, 0.1538276
1: -0.0392963, 0.0221250, -0.0356271, 0.0114858, -0.0507821, 0.0577521
2: -0.0105301, 0.0687656, -0.0099601, 0.0580217, -0.0685518, 0.0787256
3: -0.0217469, 0.0603041, -0.0163986, 0.0530349, -0.0747818, 0.0767028
4: -0.0501365, 0.0174244, -0.0411578, 0.0134654, -0.0636018, 0.0585822
5: -0.0200388, 0.1392722, -0.0135677, 0.1277263, -0.1477651, 0.1528399
6: -0.0352023, 0.0367661, -0.0296766, 0.0302907, -0.0654930, 0.0664427
7: -0.0599655, 0.0330067, -0.0496792, 0.0263285, -0.0862940, 0.0826859
8: -0.0369966, 0.0579110, -0.0319371, 0.0482615, -0.0852581, 0.0898481
9: -0.0394318, 0.0434997, -0.0312266, 0.0362918, -0.0757236, 0.0747263

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366877, upper bound: 0.5158620
time: 2.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5219841, upper bound: 0.5157833
time: 1.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.8664274, 1.0054400, 0.8020796, 1.0052152, -0.1387878, 0.2033604
1: -0.0322058, 0.0068628, -0.0400006, 0.0274659, -0.0596717, 0.0468633
2: -0.0105393, 0.0480032, -0.0101879, 0.0734017, -0.0839411, 0.0581911
3: -0.0138836, 0.0462565, -0.0264262, 0.0629036, -0.0767872, 0.0726827
4: -0.0346202, 0.0097736, -0.0533094, 0.0187335, -0.0533537, 0.0630829
5: -0.0142963, 0.1169602, -0.0258637, 0.1437354, -0.1580317, 0.1428239
6: -0.0254867, 0.0242525, -0.0363088, 0.0453397, -0.0708264, 0.0605612
7: -0.0421370, 0.0201010, -0.0615233, 0.0471603, -0.0892974, 0.0816243
8: -0.0272191, 0.0419137, -0.0386352, 0.0630718, -0.0902909, 0.0805489
9: -0.0235752, 0.0328496, -0.0415946, 0.0471995, -0.0707747, 0.0744442

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5454485
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5454546
time: 1.95 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.8736324, 1.0051091, 0.5421206, 1.0179931, -0.1443607, 0.4629885
1: -0.0305426, 0.0046155, -0.0728287, 0.0988154, -0.1293580, 0.0774442
2: -0.0101020, 0.0431330, -0.0310920, 0.1543489, -0.1644509, 0.0742250
3: -0.0126609, 0.0429614, -0.0846072, 0.1062744, -0.1189353, 0.1275686
4: -0.0314421, 0.0079789, -0.1205183, 0.0618833, -0.0933254, 0.1284972
5: -0.0137463, 0.1117266, -0.1138358, 0.2136508, -0.2273970, 0.2255624
6: -0.0234499, 0.0213172, -0.0750471, 0.1216576, -0.1451075, 0.0963642
7: -0.0384705, 0.0170738, -0.1228407, 0.1512467, -0.1897172, 0.1399146
8: -0.0249256, 0.0388280, -0.0824477, 0.1396631, -0.1645887, 0.1212756
9: -0.0198557, 0.0311763, -0.0869564, 0.1432792, -0.1631349, 0.1181327

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5310490
time: 1.95 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5311269
time: 2.32 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.8319507, 1.0055476, 0.8040784, 1.0052030, -0.1732523, 0.2014692
1: -0.0401642, 0.0247834, -0.0394955, 0.0262470, -0.0664112, 0.0642789
2: -0.0106812, 0.0713069, -0.0101768, 0.0720723, -0.0827534, 0.0814838
3: -0.0230782, 0.0620237, -0.0256274, 0.0621073, -0.0851855, 0.0876511
4: -0.0523094, 0.0183609, -0.0519624, 0.0183059, -0.0706153, 0.0703233
5: -0.0221393, 0.1420032, -0.0247723, 0.1424723, -0.1646116, 0.1667755
6: -0.0365351, 0.0382978, -0.0356986, 0.0444470, -0.0809821, 0.0739964
7: -0.0624535, 0.0345864, -0.0601801, 0.0463403, -0.1087938, 0.0947665
8: -0.0381934, 0.0602645, -0.0377033, 0.0619730, -0.1001664, 0.0979677
9: -0.0413727, 0.0452925, -0.0407017, 0.0456327, -0.0870055, 0.0859943

Time for backsubstitution: 2.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5453703
time: 2.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5453916
time: 2.13 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.8391970, 1.0052139, 0.5440550, 1.0176034, -0.1784064, 0.4611588
1: -0.0384915, 0.0196597, -0.0722190, 0.0978262, -0.1363177, 0.0918787
2: -0.0102405, 0.0664090, -0.0306925, 0.1530969, -0.1633374, 0.0971015
3: -0.0205124, 0.0587097, -0.0838477, 0.1056017, -0.1261141, 0.1425574
4: -0.0481214, 0.0165561, -0.1193818, 0.0611423, -0.1092637, 0.1359378
5: -0.0180910, 0.1367396, -0.1125506, 0.2125984, -0.2306894, 0.2492902
6: -0.0339663, 0.0353458, -0.0743559, 0.1209460, -0.1549123, 0.1097017
7: -0.0576584, 0.0315418, -0.1216910, 0.1505107, -0.2081691, 0.1532328
8: -0.0358868, 0.0557286, -0.0816596, 0.1385818, -0.1744686, 0.1373882
9: -0.0376320, 0.0418372, -0.0862045, 0.1415997, -0.1792317, 0.1280418

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5310768
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5311610
time: 1.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 6.38 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5394016, upper bound: 0.5386521
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5394016, upper bound: 0.5386521
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5391884, upper bound: 0.5383465
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5390143, upper bound: 0.5384181
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5387002, upper bound: 0.5386991
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5387002, upper bound: 0.5386991
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5385075, upper bound: 0.5383833
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5384655, upper bound: 0.5384655
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5392998, upper bound: 0.5439052
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5392998, upper bound: 0.5439064
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5390716, upper bound: 0.5436457
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5389153, upper bound: 0.5437163
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5386495, upper bound: 0.5439192
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5386495, upper bound: 0.5439181
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5384521, upper bound: 0.5436563
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5384163, upper bound: 0.5437301
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5524572, upper bound: 0.5439109
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5523488, upper bound: 0.5440009
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5524572, upper bound: 0.5439098
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5523488, upper bound: 0.5440009
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5366736, upper bound: 0.5168901
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5219611, upper bound: 0.5168025
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5366877, upper bound: 0.5158620
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5219841, upper bound: 0.5157833
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5454485
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5454546
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5310490
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5311269
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5453703
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5312454, upper bound: 0.5453916
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5310768
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 6.38
Output dim: 0, lower bound: -0.5311621, upper bound: 0.5311610

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8475226, 1.0045170, 0.8157468, 1.0051644, -0.1576418, 0.1887702
1: -0.0344528, 0.0111502, -0.0389429, 0.0236425, -0.0580954, 0.0500931
2: -0.0093207, 0.0555231, -0.0101435, 0.0697602, -0.0790809, 0.0656666
3: -0.0166171, 0.0512180, -0.0236597, 0.0607227, -0.0773399, 0.0748776
4: -0.0393484, 0.0124539, -0.0501662, 0.0175735, -0.0569219, 0.0626200
5: -0.0127443, 0.1249436, -0.0223202, 0.1401681, -0.1529123, 0.1472638
6: -0.0283471, 0.0303835, -0.0348772, 0.0411934, -0.0695404, 0.0652607
7: -0.0470183, 0.0281106, -0.0587467, 0.0414129, -0.0884311, 0.0868573
8: -0.0304056, 0.0473210, -0.0367036, 0.0596123, -0.0900179, 0.0840246
9: -0.0289359, 0.0355937, -0.0393500, 0.0437550, -0.0726909, 0.0749438

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505543
time: 1.92 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505533
time: 1.90 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.6050580, 1.0060393, 0.8157468, 1.0051644, -0.4001064, 0.1902925
1: -0.0521063, 0.0787459, -0.0389429, 0.0236425, -0.0757489, 0.1176888
2: -0.0237023, 0.1198277, -0.0101435, 0.0697602, -0.0934625, 0.1299713
3: -0.0600483, 0.0933216, -0.0236597, 0.0607227, -0.1207710, 0.1169813
4: -0.0884303, 0.0344361, -0.0501662, 0.0175735, -0.1060038, 0.0846023
5: -0.0724631, 0.1930384, -0.0223202, 0.1401681, -0.2126311, 0.2153586
6: -0.0563515, 0.0994952, -0.0348772, 0.0411934, -0.0975448, 0.1343724
7: -0.0952857, 0.1288399, -0.0587467, 0.0414129, -0.1366986, 0.1875866
8: -0.0559764, 0.1118198, -0.0367036, 0.0596123, -0.1155887, 0.1485234
9: -0.0728827, 0.0766899, -0.0393500, 0.0437550, -0.1166377, 0.1160399

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505533
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505533
time: 2.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8830812, 1.0038277, 0.5875692, 1.0155369, -0.1324557, 0.4162585
1: -0.0279403, 0.0017898, -0.0689482, 0.0893146, -0.1172549, 0.0707381
2: -0.0084101, 0.0366638, -0.0282133, 0.1440540, -0.1524641, 0.0648771
3: -0.0115233, 0.0378107, -0.0764720, 0.1007089, -0.1122322, 0.1142827
4: -0.0264696, 0.0074877, -0.1122374, 0.0567288, -0.0831984, 0.1197251
5: -0.0116182, 0.1035378, -0.1022272, 0.2044471, -0.2160652, 0.2057650
6: -0.0204868, 0.0190308, -0.0703794, 0.1094604, -0.1299472, 0.0894102
7: -0.0346523, 0.0123372, -0.1156715, 0.1332350, -0.1678873, 0.1280087
8: -0.0216055, 0.0353422, -0.0772110, 0.1293185, -0.1509240, 0.1125532
9: -0.0155588, 0.0285582, -0.0813087, 0.1315762, -0.1471349, 0.1098668

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5260687, upper bound: 0.5216633
time: 1.93 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5261368, upper bound: 0.5251925
time: 1.90 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.8679925, 1.0041752, 0.5795478, 1.0162206, -0.1482280, 0.4246274
1: -0.0318445, 0.0063746, -0.0700241, 0.0915420, -0.1233865, 0.0763987
2: -0.0088690, 0.0469453, -0.0289687, 0.1466221, -0.1554911, 0.0759140
3: -0.0136180, 0.0455407, -0.0782824, 0.1020909, -0.1157089, 0.1238231
4: -0.0339298, 0.0093837, -0.1144015, 0.0581128, -0.0920426, 0.1237852
5: -0.0121955, 0.1158233, -0.1050163, 0.2066798, -0.2188753, 0.2208397
6: -0.0250442, 0.0236148, -0.0716398, 0.1118062, -0.1368504, 0.0952546
7: -0.0413405, 0.0194435, -0.1176779, 0.1364426, -0.1777831, 0.1371214
8: -0.0267209, 0.0412434, -0.0786346, 0.1317392, -0.1584601, 0.1198780
9: -0.0227672, 0.0324861, -0.0827653, 0.1346941, -0.1574613, 0.1152513

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5257938, upper bound: 0.5217408
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5258640, upper bound: 0.5252633
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.7955102, 1.0047996, 0.8181052, 1.0051500, -0.2096398, 0.1866944
1: -0.0468281, 0.0375616, -0.0384902, 0.0222300, -0.0690581, 0.0760519
2: -0.0121755, 0.0867606, -0.0101275, 0.0684056, -0.0805811, 0.0968881
3: -0.0337062, 0.0698839, -0.0229272, 0.0598074, -0.0935136, 0.0928110
4: -0.0657257, 0.0269691, -0.0490172, 0.0170797, -0.0828054, 0.0759863
5: -0.0384374, 0.1546409, -0.0211726, 0.1387129, -0.1771503, 0.1758135
6: -0.0439362, 0.0505414, -0.0341782, 0.0403130, -0.0842492, 0.0847196
7: -0.0746928, 0.0490165, -0.0574523, 0.0404316, -0.1151245, 0.1064688
8: -0.0475077, 0.0739535, -0.0360759, 0.0583427, -0.1058504, 0.1100294
9: -0.0497345, 0.0655603, -0.0383255, 0.0428035, -0.0925380, 0.1038857

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384525, upper bound: 0.5501970
time: 2.10 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5385368, upper bound: 0.5500629
time: 2.01 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.5454589, 1.0222929, 0.8181052, 1.0051500, -0.4596910, 0.2041877
1: -0.0795269, 0.1076592, -0.0384902, 0.0222300, -0.1017568, 0.1461494
2: -0.0352715, 0.1666846, -0.0101275, 0.0684056, -0.1036771, 0.1768121
3: -0.0907259, 0.1127761, -0.0229272, 0.0598074, -0.1505333, 0.1357033
4: -0.1323792, 0.0697831, -0.0490172, 0.0170797, -0.1494589, 0.1188003
5: -0.1255808, 0.2235657, -0.0211726, 0.1387129, -0.2642937, 0.2447383
6: -0.0825002, 0.1237503, -0.0341782, 0.0403130, -0.1228133, 0.1579285
7: -0.1355744, 0.1498369, -0.0574523, 0.0404316, -0.1760060, 0.2072892
8: -0.0909913, 0.1492868, -0.0360759, 0.0583427, -0.1493340, 0.1853628
9: -0.0947299, 0.1610273, -0.0383255, 0.0428035, -0.1375334, 0.1993528

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384525, upper bound: 0.5501960
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5385368, upper bound: 0.5500639
time: 1.99 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.8453747, 1.0039610, 0.5901104, 1.0150445, -0.1696697, 0.4138505
1: -0.0370654, 0.0152915, -0.0681782, 0.0880423, -0.1251077, 0.0834697
2: -0.0085860, 0.0622332, -0.0277077, 0.1424588, -0.1510448, 0.0899409
3: -0.0183249, 0.0558844, -0.0754946, 0.0998538, -0.1181787, 0.1313790
4: -0.0445510, 0.0150173, -0.1107982, 0.0557858, -0.1003369, 0.1258155
5: -0.0146395, 0.1322522, -0.1005897, 0.2031153, -0.2177548, 0.2328420
6: -0.0317763, 0.0328290, -0.0695058, 0.1085196, -0.1402960, 0.1023348
7: -0.0535702, 0.0289462, -0.1142200, 0.1322717, -0.1858419, 0.1431662
8: -0.0339204, 0.0518615, -0.0762148, 0.1279369, -0.1618572, 0.1280764
9: -0.0344429, 0.0388915, -0.0803563, 0.1294508, -0.1638937, 0.1192477

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5253942, upper bound: 0.5217003
time: 2.11 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5254547, upper bound: 0.5252296
time: 2.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.8283511, 1.0043559, 0.5820538, 1.0157275, -0.1873764, 0.4223021
1: -0.0419131, 0.0269977, -0.0692529, 0.0902729, -0.1321860, 0.0962506
2: -0.0091076, 0.0746209, -0.0284629, 0.1450303, -0.1541379, 0.1030837
3: -0.0253107, 0.0634558, -0.0773081, 0.1012369, -0.1265476, 0.1407639
4: -0.0557280, 0.0203141, -0.1129622, 0.0571703, -0.1128983, 0.1332763
5: -0.0253908, 0.1442779, -0.1033825, 0.2053505, -0.2307412, 0.2476604
6: -0.0381463, 0.0402637, -0.0707653, 0.1108755, -0.1490218, 0.1110290
7: -0.0655396, 0.0359021, -0.1162240, 0.1354939, -0.2010335, 0.1521261
8: -0.0409765, 0.0627370, -0.0776375, 0.1303606, -0.1713371, 0.1403745
9: -0.0429893, 0.0512261, -0.0818131, 0.1325674, -0.1755567, 0.1330391

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5252426, upper bound: 0.5217861
time: 2.05 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5253122, upper bound: 0.5253122
time: 2.21 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8475226, 1.0045170, 0.8101797, 1.0056833, -0.1581607, 0.1943372
1: -0.0344528, 0.0111502, -0.0374138, 0.0205663, -0.0550192, 0.0485640
2: -0.0093207, 0.0555231, -0.0108307, 0.0664197, -0.0757404, 0.0663538
3: -0.0166171, 0.0512180, -0.0227687, 0.0582890, -0.0749061, 0.0739867
4: -0.0393484, 0.0124539, -0.0468290, 0.0162516, -0.0556000, 0.0592828
5: -0.0127443, 0.1249436, -0.0202895, 0.1364836, -0.1492279, 0.1452331
6: -0.0283471, 0.0303835, -0.0326467, 0.0414243, -0.0697714, 0.0630302
7: -0.0470183, 0.0281106, -0.0542534, 0.0436967, -0.0907149, 0.0823641
8: -0.0304056, 0.0473210, -0.0347001, 0.0569361, -0.0873417, 0.0820211
9: -0.0289359, 0.0355937, -0.0363343, 0.0411208, -0.0700567, 0.0719280

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563066
time: 2.00 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563055
time: 1.98 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.6050580, 1.0060393, 0.8101797, 1.0056833, -0.4006253, 0.1958596
1: -0.0521063, 0.0787459, -0.0374138, 0.0205663, -0.0726727, 0.1161597
2: -0.0237023, 0.1198277, -0.0108307, 0.0664197, -0.0901220, 0.1306584
3: -0.0600483, 0.0933216, -0.0227687, 0.0582890, -0.1183372, 0.1160903
4: -0.0884303, 0.0344361, -0.0468290, 0.0162516, -0.1046820, 0.0812651
5: -0.0724631, 0.1930384, -0.0202895, 0.1364836, -0.2089467, 0.2133278
6: -0.0563515, 0.0994952, -0.0326467, 0.0414243, -0.0977758, 0.1321419
7: -0.0952857, 0.1288399, -0.0542534, 0.0436967, -0.1389824, 0.1830933
8: -0.0559764, 0.1118198, -0.0347001, 0.0569361, -0.1129125, 0.1465199
9: -0.0728827, 0.0766899, -0.0363343, 0.0411208, -0.1140035, 0.1130241

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563066
time: 2.08 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563066
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8830812, 1.0038277, 0.6004568, 1.0126040, -0.1295228, 0.4033709
1: -0.0279403, 0.0017898, -0.0572961, 0.0828254, -0.1107657, 0.0590859
2: -0.0084101, 0.0366638, -0.0253309, 0.1291856, -0.1375956, 0.0619947
3: -0.0115233, 0.0378107, -0.0692302, 0.0960304, -0.1075537, 0.1070409
4: -0.0264696, 0.0074877, -0.1040817, 0.0379388, -0.0644084, 0.1115694
5: -0.0116182, 0.1035378, -0.0793393, 0.1984689, -0.2100871, 0.1828771
6: -0.0204868, 0.0190308, -0.0584533, 0.1092419, -0.1297286, 0.0774842
7: -0.0346523, 0.0123372, -0.1065429, 0.1303730, -0.1650253, 0.1188801
8: -0.0216055, 0.0353422, -0.0713864, 0.1154507, -0.1370561, 0.1067287
9: -0.0155588, 0.0285582, -0.0759994, 0.1054972, -0.1210559, 0.1045575

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5259109, upper bound: 0.5276695
time: 2.10 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5260228, upper bound: 0.5310478
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.8679925, 1.0041752, 0.5925992, 1.0133096, -0.1453171, 0.4115760
1: -0.0318445, 0.0063746, -0.0580787, 0.0851231, -0.1169676, 0.0644533
2: -0.0088690, 0.0469453, -0.0261098, 0.1315530, -0.1404221, 0.0730551
3: -0.0136180, 0.0455407, -0.0710668, 0.0974539, -0.1110719, 0.1166075
4: -0.0339298, 0.0093837, -0.1063143, 0.0388552, -0.0727851, 0.1156980
5: -0.0121955, 0.1158233, -0.0815519, 0.2008564, -0.2130518, 0.1973752
6: -0.0250442, 0.0236148, -0.0594370, 0.1119716, -0.1370159, 0.0830518
7: -0.0413405, 0.0194435, -0.1085925, 0.1336132, -0.1749537, 0.1280359
8: -0.0267209, 0.0412434, -0.0728555, 0.1176385, -0.1443594, 0.1140989
9: -0.0227672, 0.0324861, -0.0775026, 0.1080752, -0.1308424, 0.1099887

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5256471, upper bound: 0.5277422
time: 2.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5257506, upper bound: 0.5311229
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.7955102, 1.0047996, 0.8118653, 1.0056447, -0.2101345, 0.1929343
1: -0.0468281, 0.0375616, -0.0370701, 0.0195176, -0.0663457, 0.0746318
2: -0.0121755, 0.0867606, -0.0107825, 0.0654024, -0.0775779, 0.0975431
3: -0.0337062, 0.0698839, -0.0222278, 0.0575993, -0.0913055, 0.0921116
4: -0.0657257, 0.0269691, -0.0459609, 0.0158819, -0.0816076, 0.0729299
5: -0.0384374, 0.1546409, -0.0194329, 0.1353886, -0.1738259, 0.1740737
6: -0.0439362, 0.0505414, -0.0321170, 0.0407853, -0.0847215, 0.0826584
7: -0.0746928, 0.0490165, -0.0532697, 0.0430051, -0.1176979, 0.1022862
8: -0.0475077, 0.0739535, -0.0342245, 0.0559868, -0.1034945, 0.1081780
9: -0.0497345, 0.0655603, -0.0355647, 0.0404030, -0.0901375, 0.1011250

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384086, upper bound: 0.5559014
time: 1.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384834, upper bound: 0.5558897
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.5454589, 1.0222929, 0.8118653, 1.0056447, -0.4601858, 0.2104275
1: -0.0795269, 0.1076592, -0.0370701, 0.0195176, -0.0990444, 0.1447293
2: -0.0352715, 0.1666846, -0.0107825, 0.0654024, -0.1006739, 0.1774671
3: -0.0907259, 0.1127761, -0.0222278, 0.0575993, -0.1483252, 0.1350038
4: -0.1323792, 0.0697831, -0.0459609, 0.0158819, -0.1482611, 0.1157440
5: -0.1255808, 0.2235657, -0.0194329, 0.1353886, -0.2609694, 0.2429986
6: -0.0825002, 0.1237503, -0.0321170, 0.0407853, -0.1232856, 0.1558673
7: -0.1355744, 0.1498369, -0.0532697, 0.0430051, -0.1785795, 0.2031066
8: -0.0909913, 0.1492868, -0.0342245, 0.0559868, -0.1469781, 0.1835114
9: -0.0947299, 0.1610273, -0.0355647, 0.0404030, -0.1351329, 0.1965920

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384086, upper bound: 0.5559024
time: 2.11 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384834, upper bound: 0.5558907
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.8453747, 1.0039610, 0.6020847, 1.0118539, -0.1664792, 0.4018763
1: -0.0370654, 0.0152915, -0.0531378, 0.0819078, -0.1189732, 0.0684293
2: -0.0085860, 0.0622332, -0.0249616, 0.1229976, -0.1315836, 0.0871948
3: -0.0183249, 0.0558844, -0.0632209, 0.0954079, -0.1137329, 0.1191053
4: -0.0445510, 0.0150173, -0.0911242, 0.0376094, -0.0821604, 0.1061414
5: -0.0146395, 0.1322522, -0.0750769, 0.1974905, -0.2121300, 0.2073291
6: -0.0317763, 0.0328290, -0.0579780, 0.1042777, -0.1360540, 0.0908071
7: -0.0535702, 0.0289462, -0.0982329, 0.1297024, -0.1832727, 0.1271792
8: -0.0339204, 0.0518615, -0.0574370, 0.1146086, -0.1485289, 0.1092985
9: -0.0344429, 0.0388915, -0.0753050, 0.0789749, -0.1134178, 0.1141964

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5252866, upper bound: 0.5276763
time: 1.97 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5253692, upper bound: 0.5310632
time: 2.25 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.8283511, 1.0043559, 0.5942331, 1.0124367, -0.1840856, 0.4101228
1: -0.0419131, 0.0269977, -0.0537544, 0.0842084, -0.1261215, 0.0807521
2: -0.0091076, 0.0746209, -0.0257415, 0.1251421, -0.1342497, 0.1003624
3: -0.0253107, 0.0634558, -0.0648244, 0.0968331, -0.1221438, 0.1282802
4: -0.0557280, 0.0203141, -0.0928307, 0.0385298, -0.0942578, 0.1131448
5: -0.0253908, 0.1442779, -0.0771402, 0.1998812, -0.2252720, 0.2214181
6: -0.0381463, 0.0402637, -0.0589630, 0.1068446, -0.1449909, 0.0992267
7: -0.0655396, 0.0359021, -0.0999632, 0.1329397, -0.1984793, 0.1358653
8: -0.0409765, 0.0627370, -0.0583211, 0.1168000, -0.1577764, 0.1210581
9: -0.0429893, 0.0512261, -0.0768099, 0.0805325, -0.1235217, 0.1280359

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 113
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 188

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5251388, upper bound: 0.5277525
time: 3.27 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5252313, upper bound: 0.5311397
time: 2.47 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8904165, 1.0046444, 0.8635628, 1.0046105, -0.1141940, 0.1410816
1: -0.0259053, 0.0001140, -0.0328670, 0.0077562, -0.0336614, 0.0329810
2: -0.0094886, 0.0320771, -0.0094437, 0.0499393, -0.0594280, 0.0415208
3: -0.0106761, 0.0337868, -0.0143696, 0.0475665, -0.0582426, 0.0481564
4: -0.0225811, 0.0072065, -0.0358836, 0.0104870, -0.0330681, 0.0430901
5: -0.0129747, 0.0971342, -0.0129183, 0.1190409, -0.1320156, 0.1100525
6: -0.0183830, 0.0173107, -0.0262964, 0.0254194, -0.0438024, 0.0436071
7: -0.0321352, 0.0093693, -0.0435946, 0.0213045, -0.0534397, 0.0529639
8: -0.0192997, 0.0326559, -0.0281309, 0.0431405, -0.0624402, 0.0607868
9: -0.0124734, 0.0265280, -0.0250538, 0.0335148, -0.0459883, 0.0515818

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311417, upper bound: 0.5390430
time: 2.19 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5310857, upper bound: 0.5276742
time: 1.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.8785998, 1.0049850, 0.8611902, 1.0047473, -0.1261474, 0.1437948
1: -0.0291835, 0.0028255, -0.0334147, 0.0084962, -0.0376798, 0.0362402
2: -0.0099382, 0.0397339, -0.0096243, 0.0515431, -0.0614813, 0.0493582
3: -0.0120529, 0.0402689, -0.0147722, 0.0486516, -0.0607045, 0.0550411
4: -0.0288452, 0.0076749, -0.0369302, 0.0110780, -0.0399232, 0.0446051
5: -0.0135402, 0.1074499, -0.0131454, 0.1207642, -0.1343045, 0.1205954
6: -0.0217918, 0.0200817, -0.0269671, 0.0263860, -0.0481778, 0.0470489
7: -0.0363038, 0.0146002, -0.0448019, 0.0223014, -0.0586052, 0.0594021
8: -0.0230590, 0.0369834, -0.0288861, 0.0441566, -0.0672156, 0.0658695
9: -0.0174438, 0.0298090, -0.0262787, 0.0340659, -0.0515096, 0.0560876

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311417, upper bound: 0.5391008
time: 2.34 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5310857, upper bound: 0.5277500
time: 2.06 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8904165, 1.0046444, 0.8277438, 1.0047812, -0.1143647, 0.1769006
1: -0.0259053, 0.0001140, -0.0411353, 0.0277581, -0.0536633, 0.0412493
2: -0.0094886, 0.0320771, -0.0096691, 0.0741505, -0.0836391, 0.0417463
3: -0.0106761, 0.0337868, -0.0245678, 0.0639476, -0.0746236, 0.0583546
4: -0.0225811, 0.0072065, -0.0547408, 0.0194088, -0.0419899, 0.0619472
5: -0.0129747, 0.0971342, -0.0244897, 0.1450589, -0.1580337, 0.1216239
6: -0.0183830, 0.0173107, -0.0380264, 0.0400116, -0.0583946, 0.0553371
7: -0.0321352, 0.0093693, -0.0652375, 0.0363539, -0.0684890, 0.0746068
8: -0.0192997, 0.0326559, -0.0395325, 0.0628979, -0.0821976, 0.0721884
9: -0.0124734, 0.0265280, -0.0435444, 0.0472985, -0.0597719, 0.0700724

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311607, upper bound: 0.5390430
time: 2.15 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5310972, upper bound: 0.5276738
time: 2.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.8785998, 1.0049850, 0.8253499, 1.0049274, -0.1263276, 0.1796351
1: -0.0291835, 0.0028255, -0.0416879, 0.0294508, -0.0586343, 0.0445134
2: -0.0099382, 0.0397339, -0.0098622, 0.0757686, -0.0857068, 0.0495961
3: -0.0120529, 0.0402689, -0.0254154, 0.0650423, -0.0770952, 0.0656844
4: -0.0288452, 0.0076749, -0.0561243, 0.0200050, -0.0488502, 0.0637993
5: -0.0135402, 0.1074499, -0.0258271, 0.1467978, -0.1603381, 0.1332770
6: -0.0217918, 0.0200817, -0.0388750, 0.0409868, -0.0627786, 0.0589568
7: -0.0363038, 0.0146002, -0.0668215, 0.0373598, -0.0736636, 0.0814217
8: -0.0230590, 0.0369834, -0.0402945, 0.0643963, -0.0874553, 0.0772780
9: -0.0174438, 0.0298090, -0.0447802, 0.0484399, -0.0658837, 0.0745891

Time for backsubstitution: 2.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311607, upper bound: 0.5391008
time: 2.30 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5310972, upper bound: 0.5277485
time: 2.55 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8515053, 1.0048305, 0.8687410, 1.0046151, -0.1531097, 0.1360894
1: -0.0356503, 0.0115169, -0.0316717, 0.0061411, -0.0417914, 0.0431886
2: -0.0097344, 0.0580893, -0.0094498, 0.0464393, -0.0561737, 0.0675391
3: -0.0164156, 0.0530807, -0.0134909, 0.0451984, -0.0616140, 0.0665717
4: -0.0412019, 0.0134903, -0.0335997, 0.0091972, -0.0503992, 0.0470899
5: -0.0132838, 0.1277991, -0.0129259, 0.1152796, -0.1285634, 0.1407250
6: -0.0297049, 0.0303314, -0.0248327, 0.0233099, -0.0530148, 0.0551641
7: -0.0497301, 0.0263704, -0.0409596, 0.0191289, -0.0688591, 0.0673301
8: -0.0319689, 0.0483043, -0.0264826, 0.0409228, -0.0728917, 0.0747869
9: -0.0312781, 0.0363151, -0.0223807, 0.0323123, -0.0635904, 0.0586958

Time for backsubstitution: 2.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5159981, upper bound: 0.5094411
time: 2.19 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5130159, upper bound: 0.4940920
time: 1.59 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.7227170, 1.0164382, 0.8784272, 1.0043285, -0.2816114, 0.1380110
1: -0.0653791, 0.1020205, -0.0292314, 0.0028715, -0.0682506, 0.1312519
2: -0.0377989, 0.1451411, -0.0090713, 0.0398522, -0.0776511, 0.1542124
3: -0.0617562, 0.1119792, -0.0120733, 0.0403637, -0.1021199, 0.1240525
4: -0.1154402, 0.0455685, -0.0289368, 0.0076822, -0.1231224, 0.0745052
5: -0.0831668, 0.2213478, -0.0124499, 0.1076007, -0.1907675, 0.2337977
6: -0.0752578, 0.0827981, -0.0218442, 0.0201222, -0.0953800, 0.1046423
7: -0.1347383, 0.0804808, -0.0363702, 0.0146874, -0.1494257, 0.1168510
8: -0.0729638, 0.1286402, -0.0231176, 0.0370467, -0.1100105, 0.1517578
9: -0.0977613, 0.0973782, -0.0175192, 0.0298571, -0.1276185, 0.1148974

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4942860, upper bound: 0.5060701
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4941783, upper bound: 0.4938899
time: 1.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.8489954, 1.0049803, 0.8516055, 1.0050014, -0.1560060, 0.1533749
1: -0.0362297, 0.0127314, -0.0356271, 0.0114858, -0.0477154, 0.0483585
2: -0.0099320, 0.0597859, -0.0099601, 0.0580217, -0.0679538, 0.0697459
3: -0.0170429, 0.0542286, -0.0163986, 0.0530349, -0.0700778, 0.0706272
4: -0.0424585, 0.0141154, -0.0411578, 0.0134654, -0.0559239, 0.0552732
5: -0.0135325, 0.1296222, -0.0135677, 0.1277263, -0.1412588, 0.1431899
6: -0.0304928, 0.0313540, -0.0296766, 0.0302907, -0.0607836, 0.0610306
7: -0.0511743, 0.0274250, -0.0496792, 0.0263285, -0.0775028, 0.0771042
8: -0.0327678, 0.0495952, -0.0319371, 0.0482615, -0.0810293, 0.0815323
9: -0.0325738, 0.0371651, -0.0312266, 0.0362918, -0.0688656, 0.0683917

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5159955, upper bound: 0.5086783
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5130055, upper bound: 0.4926417
time: 2.03 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.7202330, 1.0168777, 0.8622903, 1.0047196, -0.2844866, 0.1545874
1: -0.0659526, 0.1037770, -0.0331608, 0.0081531, -0.0741057, 0.1369378
2: -0.0384800, 0.1468202, -0.0095879, 0.0507996, -0.0892796, 0.1564081
3: -0.0626358, 0.1131153, -0.0145855, 0.0481485, -0.1107843, 0.1277009
4: -0.1168758, 0.0461872, -0.0364450, 0.0108040, -0.1276798, 0.0826322
5: -0.0845547, 0.2231523, -0.0130996, 0.1199652, -0.2045199, 0.2362518
6: -0.0761384, 0.0838101, -0.0266562, 0.0259378, -0.1020763, 0.1104663
7: -0.1363821, 0.0815245, -0.0442422, 0.0218392, -0.1582214, 0.1257667
8: -0.0737546, 0.1301952, -0.0285360, 0.0436855, -0.1174401, 0.1587311
9: -0.0990437, 0.0985627, -0.0257108, 0.0338104, -0.1328541, 0.1242735

Time for backsubstitution: 2.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 11

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4943554, upper bound: 0.5054495
time: 1.85 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4942220, upper bound: 0.4924439
time: 1.59 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: 0.8898028, 1.0046473, 0.8241454, 1.0048623, -0.1150595, 0.1805018
1: -0.0260756, 0.0002491, -0.0385392, 0.0216058, -0.0476814, 0.0387884
2: -0.0094924, 0.0323450, -0.0097544, 0.0679731, -0.0774655, 0.0420994
3: -0.0107418, 0.0341235, -0.0222747, 0.0595964, -0.0703382, 0.0563983
4: -0.0229065, 0.0072233, -0.0488822, 0.0169826, -0.0398891, 0.0561056
5: -0.0129795, 0.0976701, -0.0204275, 0.1382896, -0.1512691, 0.1180976
6: -0.0185505, 0.0174547, -0.0341909, 0.0389190, -0.0574695, 0.0516456
7: -0.0322966, 0.0093724, -0.0576496, 0.0379216, -0.0702182, 0.0670221
8: -0.0194733, 0.0328806, -0.0360877, 0.0577319, -0.0772052, 0.0689684
9: -0.0127316, 0.0266933, -0.0382331, 0.0426335, -0.0553651, 0.0649264

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454475
time: 2.11 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454485
time: 6.03 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: 0.8780914, 1.0049870, 0.8177329, 1.0050333, -0.1269419, 0.1872541
1: -0.0293246, 0.0029953, -0.0391269, 0.0238482, -0.0531729, 0.0421221
2: -0.0099411, 0.0400822, -0.0099723, 0.0700282, -0.0799693, 0.0500545
3: -0.0121130, 0.0405483, -0.0236200, 0.0609420, -0.0730550, 0.0641683
4: -0.0291148, 0.0076972, -0.0505093, 0.0176999, -0.0468147, 0.0582065
5: -0.0135438, 0.1078939, -0.0223702, 0.1404706, -0.1540145, 0.1302641
6: -0.0219583, 0.0202010, -0.0351257, 0.0408672, -0.0628256, 0.0553267
7: -0.0365151, 0.0148570, -0.0592971, 0.0405929, -0.0771080, 0.0741541
8: -0.0232461, 0.0371697, -0.0369371, 0.0597642, -0.0830103, 0.0741068
9: -0.0176820, 0.0299509, -0.0396605, 0.0440259, -0.0617079, 0.0696114

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454546
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454556
time: 2.18 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: 0.8955851, 1.0043106, 0.5692257, 1.0165075, -0.1209224, 0.4350849
1: -0.0244714, -0.0001899, -0.0704794, 0.0930634, -0.1175348, 0.0702895
2: -0.0090478, 0.0298209, -0.0293490, 0.1480974, -0.1571452, 0.0591699
3: -0.0101228, 0.0309516, -0.0797516, 0.1028976, -0.1130204, 0.1107032
4: -0.0198413, 0.0070643, -0.1154965, 0.0587611, -0.0786023, 0.1225609
5: -0.0124204, 0.0926222, -0.1067856, 0.2080700, -0.2204903, 0.1994078
6: -0.0169727, 0.0160987, -0.0722180, 0.1143848, -0.1313575, 0.0883167
7: -0.0307759, 0.0090052, -0.1185004, 0.1404112, -0.1711872, 0.1275057
8: -0.0178385, 0.0307631, -0.0792762, 0.1334033, -0.1512418, 0.1100393
9: -0.0102995, 0.0251359, -0.0835317, 0.1361863, -0.1464858, 0.1086676

Time for backsubstitution: 2.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5248929
time: 1.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5308661
time: 2.11 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: 0.8839338, 1.0046548, 0.5608083, 1.0171850, -0.1332512, 0.4438465
1: -0.0277038, 0.0015928, -0.0715460, 0.0953080, -0.1230118, 0.0731388
2: -0.0095023, 0.0360798, -0.0301029, 0.1506699, -0.1601722, 0.0661827
3: -0.0114225, 0.0373431, -0.0815836, 0.1042854, -0.1157080, 0.1189267
4: -0.0260177, 0.0074521, -0.1176557, 0.0601348, -0.0861524, 0.1251078
5: -0.0129920, 0.1027936, -0.1095982, 0.2103181, -0.2233101, 0.2123917
6: -0.0202385, 0.0188309, -0.0734710, 0.1168077, -0.1370462, 0.0923019
7: -0.0343382, 0.0119068, -0.1204877, 0.1437627, -0.1781009, 0.1323945
8: -0.0213290, 0.0350300, -0.0806904, 0.1358469, -0.1571758, 0.1157205
9: -0.0152002, 0.0283202, -0.0849869, 0.1392916, -0.1544918, 0.1133072

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5249759
time: 2.33 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5309469
time: 2.22 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: 0.8603496, 1.0047088, 0.8261426, 1.0048531, -0.1445035, 0.1785662
1: -0.0336087, 0.0087584, -0.0381448, 0.0203792, -0.0539879, 0.0469033
2: -0.0095735, 0.0521113, -0.0097443, 0.0667970, -0.0763705, 0.0618556
3: -0.0149148, 0.0490360, -0.0216416, 0.0588009, -0.0737158, 0.0706776
4: -0.0373009, 0.0112873, -0.0478840, 0.0165527, -0.0538536, 0.0591713
5: -0.0130815, 0.1213748, -0.0194335, 0.1370265, -0.1501081, 0.1408083
6: -0.0272047, 0.0267284, -0.0335828, 0.0381619, -0.0653667, 0.0603112
7: -0.0452296, 0.0226546, -0.0565215, 0.0370882, -0.0823178, 0.0791761
8: -0.0291537, 0.0445166, -0.0355416, 0.0566310, -0.0857846, 0.0800582
9: -0.0267126, 0.0342611, -0.0373429, 0.0418064, -0.0685190, 0.0716040

Time for backsubstitution: 2.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 251

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453693
time: 2.28 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453693
time: 1.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: 0.8448214, 1.0051037, 0.8197278, 1.0050225, -0.1602011, 0.1853759
1: -0.0371932, 0.0156828, -0.0387293, 0.0226244, -0.0598176, 0.0544122
2: -0.0100950, 0.0626072, -0.0099610, 0.0688489, -0.0789440, 0.0725682
3: -0.0185209, 0.0561375, -0.0229840, 0.0601468, -0.0786677, 0.0791215
4: -0.0448709, 0.0151551, -0.0495016, 0.0172706, -0.0621415, 0.0646567
5: -0.0149487, 0.1326542, -0.0213750, 0.1392083, -0.1541569, 0.1540292
6: -0.0319725, 0.0330545, -0.0345173, 0.0401091, -0.0720816, 0.0675717
7: -0.0539365, 0.0291788, -0.0581627, 0.0397623, -0.0936988, 0.0873414
8: -0.0340965, 0.0522080, -0.0363805, 0.0586647, -0.0927612, 0.0885885
9: -0.0347286, 0.0391554, -0.0387699, 0.0431793, -0.0779079, 0.0779253

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 188
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 245

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453926
time: 1.99 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453926
time: 2.06 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: 0.8676245, 1.0043753, 0.5711756, 1.0161183, -0.1484938, 0.4331998
1: -0.0319294, 0.0064894, -0.0698704, 0.0920683, -0.1239977, 0.0763598
2: -0.0091331, 0.0471940, -0.0289495, 0.1468403, -0.1559734, 0.0761435
3: -0.0136804, 0.0457090, -0.0789863, 0.1022242, -0.1159046, 0.1246952
4: -0.0340921, 0.0094754, -0.1143602, 0.0580179, -0.0921100, 0.1238355
5: -0.0125276, 0.1160905, -0.1054982, 0.2070152, -0.2195429, 0.2215887
6: -0.0251482, 0.0237647, -0.0715273, 0.1136575, -0.1388058, 0.0952921
7: -0.0415278, 0.0195981, -0.1173522, 0.1396707, -0.1811985, 0.1369503
8: -0.0268380, 0.0414010, -0.0784888, 0.1323167, -0.1591547, 0.1198898
9: -0.0229571, 0.0325715, -0.0827796, 0.1345079, -0.1574651, 0.1153511

Time for backsubstitution: 2.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5249473
time: 2.30 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5309394
time: 2.29 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: 0.8521202, 1.0047715, 0.5627425, 1.0167955, -0.1646754, 0.4420290
1: -0.0355084, 0.0113252, -0.0709370, 0.0943156, -0.1298240, 0.0822622
2: -0.0096564, 0.0576738, -0.0297036, 0.1494152, -0.1590716, 0.0873774
3: -0.0163113, 0.0527996, -0.0808216, 0.1036129, -0.1199242, 0.1336211
4: -0.0409308, 0.0133371, -0.1165199, 0.0593929, -0.1003237, 0.1298570
5: -0.0131858, 0.1273526, -0.1083130, 0.2092659, -0.2224517, 0.2356655
6: -0.0295311, 0.0300810, -0.0727804, 0.1160861, -0.1456172, 0.1028614
7: -0.0494173, 0.0261122, -0.1193392, 0.1430262, -0.1924435, 0.1454514
8: -0.0317732, 0.0480410, -0.0799030, 0.1347624, -0.1665356, 0.1279440
9: -0.0309608, 0.0361723, -0.0842353, 0.1376137, -0.1685745, 0.1204076

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 168
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 187
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 199
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 235
type: B, layer: 1, pos: 109
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 70
type: B, layer: 1, pos: 165
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 170
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5250286
time: 2.22 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5250286
time: 2.32 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 7.16 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505543
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505533
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505533
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5394581, upper bound: 0.5505533
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5260687, upper bound: 0.5216633
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5261368, upper bound: 0.5251925
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5257938, upper bound: 0.5217408
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5258640, upper bound: 0.5252633
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5384525, upper bound: 0.5501970
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5385368, upper bound: 0.5500629
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5384525, upper bound: 0.5501960
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5385368, upper bound: 0.5500639
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5253942, upper bound: 0.5217003
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5254547, upper bound: 0.5252296
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5252426, upper bound: 0.5217861
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5253122, upper bound: 0.5253122
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563066
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563055
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563066
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5393527, upper bound: 0.5563066
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5259109, upper bound: 0.5276695
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5260228, upper bound: 0.5310478
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5256471, upper bound: 0.5277422
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5257506, upper bound: 0.5311229
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5384086, upper bound: 0.5559014
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5384834, upper bound: 0.5558897
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5384086, upper bound: 0.5559024
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5384834, upper bound: 0.5558907
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5252866, upper bound: 0.5276763
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5253692, upper bound: 0.5310632
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5251388, upper bound: 0.5277525
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5252313, upper bound: 0.5311397
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311417, upper bound: 0.5390430
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5310857, upper bound: 0.5276742
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311417, upper bound: 0.5391008
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5310857, upper bound: 0.5277500
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311607, upper bound: 0.5390430
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5310972, upper bound: 0.5276738
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311607, upper bound: 0.5391008
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5310972, upper bound: 0.5277485
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5159981, upper bound: 0.5094411
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5130159, upper bound: 0.4940920
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.4942860, upper bound: 0.5060701
NS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.4941783, upper bound: 0.4938899
NS_A2_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5159955, upper bound: 0.5086783
NS_A2_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5130055, upper bound: 0.4926417
NS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.4943554, upper bound: 0.5054495
NS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.4942220, upper bound: 0.4924439
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454475
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454485
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454546
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312173, upper bound: 0.5454556
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5248929
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5308661
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5249759
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5309469
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453693
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453693
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453926
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5312102, upper bound: 0.5453926
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5249473
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5309394
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5250286
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 7.16
Output dim: 0, lower bound: -0.5311397, upper bound: 0.5250286

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.8475226, 1.0045170, 0.8475226, 1.0045170, -0.1569943, 0.1569943
1: -0.0344528, 0.0111502, -0.0344528, 0.0111502, -0.0456030, 0.0456030
2: -0.0093207, 0.0555231, -0.0093207, 0.0555231, -0.0648438, 0.0648438
3: -0.0166171, 0.0512180, -0.0166171, 0.0512180, -0.0678351, 0.0678351
4: -0.0393484, 0.0124539, -0.0393484, 0.0124539, -0.0518022, 0.0518022
5: -0.0127443, 0.1249436, -0.0127443, 0.1249436, -0.1376879, 0.1376879
6: -0.0283471, 0.0303835, -0.0283471, 0.0303835, -0.0587306, 0.0587306
7: -0.0470183, 0.0281106, -0.0470183, 0.0281106, -0.0751289, 0.0751289
8: -0.0304056, 0.0473210, -0.0304056, 0.0473210, -0.0777266, 0.0777266
9: -0.0289359, 0.0355937, -0.0289359, 0.0355937, -0.0645296, 0.0645296

Time for backsubstitution: 2.34 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5572678, upper bound: 0.5562405
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5568804, upper bound: 0.5563061
time: 2.85 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.8475226, 1.0045170, 0.7972652, 1.0047929, -0.1572703, 0.2072517
1: -0.0344528, 0.0111502, -0.0436170, 0.0375616, -0.0720145, 0.0547671
2: -0.0093207, 0.0555231, -0.0121755, 0.0832017, -0.0925224, 0.0676986
3: -0.0166171, 0.0512180, -0.0305547, 0.0698839, -0.0865010, 0.0817726
4: -0.0393484, 0.0124539, -0.0617830, 0.0225320, -0.0618804, 0.0742369
5: -0.0127443, 0.1249436, -0.0333552, 0.1546409, -0.1673851, 0.1582988
6: -0.0283471, 0.0303835, -0.0420338, 0.0488745, -0.0772216, 0.0724173
7: -0.0470183, 0.0281106, -0.0721642, 0.0490165, -0.0960348, 0.1002748
8: -0.0304056, 0.0473210, -0.0431298, 0.0720161, -0.1024217, 0.0904509
9: -0.0289359, 0.0355937, -0.0497345, 0.0533228, -0.0822587, 0.0853282

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5572678, upper bound: 0.5562415
time: 2.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5568804, upper bound: 0.5563061
time: 2.33 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.6050580, 1.0060393, 0.8475226, 1.0045170, -0.3994590, 0.1585166
1: -0.0521063, 0.0787459, -0.0344528, 0.0111502, -0.0632565, 0.1131987
2: -0.0237023, 0.1198277, -0.0093207, 0.0555231, -0.0792255, 0.1291485
3: -0.0600483, 0.0933216, -0.0166171, 0.0512180, -0.1112662, 0.1099388
4: -0.0884303, 0.0344361, -0.0393484, 0.0124539, -0.1008842, 0.0737845
5: -0.0724631, 0.1930384, -0.0127443, 0.1249436, -0.1974067, 0.2057826
6: -0.0563515, 0.0994952, -0.0283471, 0.0303835, -0.0867350, 0.1278423
7: -0.0952857, 0.1288399, -0.0470183, 0.0281106, -0.1233964, 0.1758582
8: -0.0559764, 0.1118198, -0.0304056, 0.0473210, -0.1032974, 0.1422254
9: -0.0728827, 0.0766899, -0.0289359, 0.0355937, -0.1084764, 0.1056258

Time for backsubstitution: 2.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5234073, upper bound: 0.5386071
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264206, upper bound: 0.5393443
time: 1.94 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.6050580, 1.0060393, 0.7972652, 1.0047929, -0.3997350, 0.2087740
1: -0.0521063, 0.0787459, -0.0436170, 0.0375616, -0.0896680, 0.1223628
2: -0.0237023, 0.1198277, -0.0121755, 0.0832017, -0.1069040, 0.1320032
3: -0.0600483, 0.0933216, -0.0305547, 0.0698839, -0.1299321, 0.1238763
4: -0.0884303, 0.0344361, -0.0617830, 0.0225320, -0.1109624, 0.0962191
5: -0.0724631, 0.1930384, -0.0333552, 0.1546409, -0.2271039, 0.2263935
6: -0.0563515, 0.0994952, -0.0420338, 0.0488745, -0.1052260, 0.1415290
7: -0.0952857, 0.1288399, -0.0721642, 0.0490165, -0.1443022, 0.2010041
8: -0.0559764, 0.1118198, -0.0431298, 0.0720161, -0.1279925, 0.1549496
9: -0.0728827, 0.0766899, -0.0497345, 0.0533228, -0.1262055, 0.1264244

Time for backsubstitution: 2.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5234073, upper bound: 0.5386071
time: 2.13 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5264206, upper bound: 0.5393443
time: 2.35 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.8911656, 1.0037946, 0.7955844, 1.0063168, -0.1151512, 0.2082102
1: -0.0256975, -0.0000509, -0.0542779, 0.0478070, -0.0735045, 0.0542271
2: -0.0083664, 0.0317502, -0.0167777, 0.1005705, -0.1089369, 0.0485279
3: -0.0105959, 0.0333759, -0.0409842, 0.0769149, -0.0875108, 0.0743602
4: -0.0221841, 0.0071858, -0.0786876, 0.0360105, -0.0581946, 0.0858734
5: -0.0115634, 0.0964803, -0.0509616, 0.1656550, -0.1772183, 0.1474419
6: -0.0181786, 0.0171351, -0.0521520, 0.0548059, -0.0729845, 0.0692870
7: -0.0319382, 0.0084424, -0.0888771, 0.0482670, -0.0802052, 0.0973195
8: -0.0190880, 0.0323816, -0.0569520, 0.0848117, -0.1038996, 0.0893335
9: -0.0121584, 0.0263263, -0.0581816, 0.0851389, -0.0972974, 0.0845078

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5129362, upper bound: 0.4976857
time: 2.17 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5011834, upper bound: 0.4975914
time: 1.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.8908312, 1.0037967, 0.7567444, 1.0099293, -0.1190981, 0.2470523
1: -0.0257903, 0.0000227, -0.0599656, 0.0599262, -0.0857165, 0.0599883
2: -0.0083693, 0.0318961, -0.0207948, 0.1145368, -0.1229061, 0.0526910
3: -0.0106317, 0.0335594, -0.0506673, 0.0843733, -0.0950050, 0.0842267
4: -0.0223613, 0.0071950, -0.0902794, 0.0436143, -0.0659756, 0.0974744
5: -0.0115669, 0.0967723, -0.0661122, 0.1776235, -0.1891904, 0.1628845
6: -0.0182699, 0.0172135, -0.0588590, 0.0666874, -0.0849573, 0.0760725
7: -0.0320261, 0.0084448, -0.0994668, 0.0637930, -0.0958191, 0.1079116
8: -0.0191825, 0.0325041, -0.0645156, 0.0977722, -0.1169547, 0.0970197
9: -0.0122990, 0.0264163, -0.0660071, 0.1017420, -0.1140411, 0.0924235

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5132745, upper bound: 0.4989991
time: 1.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5012175, upper bound: 0.4989104
time: 1.50 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.8771819, 1.0041431, 0.7928085, 1.0069871, -0.1298052, 0.2113346
1: -0.0295769, 0.0033304, -0.0553254, 0.0495699, -0.0791468, 0.0586558
2: -0.0088266, 0.0407052, -0.0174612, 0.1027688, -0.1115954, 0.0581664
3: -0.0122204, 0.0410482, -0.0423120, 0.0780551, -0.0902755, 0.0833602
4: -0.0295968, 0.0077376, -0.0806326, 0.0373403, -0.0669371, 0.0883702
5: -0.0121421, 0.1086878, -0.0531278, 0.1674658, -0.1796079, 0.1618157
6: -0.0222673, 0.0204143, -0.0533385, 0.0560379, -0.0783051, 0.0737527
7: -0.0369075, 0.0153161, -0.0908542, 0.0493145, -0.0862220, 0.1061703
8: -0.0235939, 0.0375027, -0.0583053, 0.0866817, -0.1102757, 0.0958080
9: -0.0181226, 0.0302047, -0.0594686, 0.0880120, -0.1061346, 0.0896733

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5124785, upper bound: 0.4977463
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4998477, upper bound: 0.4976456
time: 1.76 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.8768486, 1.0041449, 0.7495416, 1.0105885, -0.1337399, 0.2546033
1: -0.0296693, 0.0034532, -0.0610035, 0.0621209, -0.0917902, 0.0644568
2: -0.0088291, 0.0409335, -0.0215254, 0.1170621, -0.1258911, 0.0624589
3: -0.0122598, 0.0412314, -0.0524336, 0.0857219, -0.0979818, 0.0936649
4: -0.0297735, 0.0077524, -0.0923851, 0.0449916, -0.0747652, 0.1001375
5: -0.0121452, 0.1089787, -0.0688437, 0.1797905, -0.1919357, 0.1778224
6: -0.0223805, 0.0204924, -0.0600806, 0.0688868, -0.0912673, 0.0805730
7: -0.0370513, 0.0154844, -0.1014008, 0.0666649, -0.1037162, 0.1168852
8: -0.0237214, 0.0376248, -0.0658940, 0.1001114, -0.1238329, 0.1035188
9: -0.0182842, 0.0302978, -0.0674276, 0.1047621, -0.1230463, 0.0977253

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5128251, upper bound: 0.4990652
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4998771, upper bound: 0.4989668
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.8170037, 1.0044222, 0.8705440, 1.0042251, -0.1872215, 0.1338782
1: -0.0444564, 0.0318318, -0.0312555, 0.0055788, -0.0500352, 0.0630873
2: -0.0104319, 0.0804085, -0.0089349, 0.0452206, -0.0556525, 0.0893434
3: -0.0290451, 0.0664890, -0.0131850, 0.0443738, -0.0734190, 0.0796741
4: -0.0606660, 0.0236471, -0.0328044, 0.0087482, -0.0694142, 0.0564515
5: -0.0313584, 0.1491324, -0.0122783, 0.1139699, -0.1453283, 0.1614107
6: -0.0410825, 0.0442633, -0.0243230, 0.0225754, -0.0636579, 0.0685862
7: -0.0703095, 0.0403024, -0.0400422, 0.0183714, -0.0886809, 0.0803446
8: -0.0443075, 0.0678730, -0.0259087, 0.0401506, -0.0844581, 0.0937817
9: -0.0462891, 0.0584111, -0.0214501, 0.0318935, -0.0781826, 0.0798612

Time for backsubstitution: 2.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5482889, upper bound: 0.5372279
time: 1.90 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5358953, upper bound: 0.5371510
time: 2.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.8110968, 1.0045958, 0.8515717, 1.0046142, -0.1935174, 0.1530241
1: -0.0454885, 0.0339425, -0.0356350, 0.0114963, -0.0569848, 0.0695775
2: -0.0111479, 0.0828771, -0.0094488, 0.0580445, -0.0691924, 0.0923259
3: -0.0307009, 0.0677946, -0.0164043, 0.0530504, -0.0837514, 0.0841989
4: -0.0627280, 0.0250262, -0.0411727, 0.0134737, -0.0762017, 0.0661990
5: -0.0339797, 0.1512237, -0.0129247, 0.1277510, -0.1617307, 0.1641484
6: -0.0422884, 0.0461971, -0.0296862, 0.0303044, -0.0725928, 0.0758833
7: -0.0722366, 0.0426970, -0.0496964, 0.0263426, -0.0985793, 0.0923934
8: -0.0456709, 0.0701231, -0.0319478, 0.0482759, -0.0939468, 0.1020709
9: -0.0476743, 0.0613862, -0.0312439, 0.0362996, -0.0839739, 0.0926301

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5482979, upper bound: 0.5360275
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5359521, upper bound: 0.5359510
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.5712763, 1.0207844, 0.8705440, 1.0042251, -0.4329488, 0.1502404
1: -0.0771428, 0.1020013, -0.0312555, 0.0055788, -0.0827216, 0.1332568
2: -0.0335181, 0.1604821, -0.0089349, 0.0452206, -0.0787387, 0.1694169
3: -0.0859900, 0.1093982, -0.0131850, 0.0443738, -0.1303638, 0.1225832
4: -0.1273292, 0.0666595, -0.0328044, 0.0087482, -0.1360774, 0.0994639
5: -0.1185568, 0.2180459, -0.0122783, 0.1139699, -0.2325267, 0.2303242
6: -0.0796424, 0.1167725, -0.0243230, 0.0225754, -0.1022178, 0.1410955
7: -0.1311641, 0.1393347, -0.0400422, 0.0183714, -0.1495355, 0.1793769
8: -0.0877837, 0.1431159, -0.0259087, 0.0401506, -0.1279344, 0.1690246
9: -0.0912946, 0.1538682, -0.0214501, 0.0318935, -0.1231882, 0.1753183

Time for backsubstitution: 2.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5217705, upper bound: 0.5380894
time: 2.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5253032, upper bound: 0.5389841
time: 2.03 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.5638478, 1.0214434, 0.8515717, 1.0046142, -0.4407665, 0.1698717
1: -0.0781795, 0.1041187, -0.0356350, 0.0114963, -0.0896758, 0.1397537
2: -0.0342428, 0.1629302, -0.0094488, 0.0580445, -0.0922873, 0.1723790
3: -0.0877025, 0.1107221, -0.0164043, 0.0530504, -0.1407529, 0.1271265
4: -0.1294054, 0.0679847, -0.0411727, 0.0134737, -0.1428791, 0.1091575
5: -0.1212238, 0.2201694, -0.0129247, 0.1277510, -0.2489747, 0.2330941
6: -0.0808539, 0.1189697, -0.0296862, 0.0303044, -0.1111583, 0.1486559
7: -0.1330984, 0.1423415, -0.0496964, 0.0263426, -0.1594411, 0.1920378
8: -0.0891532, 0.1454142, -0.0319478, 0.0482759, -0.1374291, 0.1773620
9: -0.0926898, 0.1568666, -0.0312439, 0.0362996, -0.1289895, 0.1881105

Time for backsubstitution: 2.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5218584, upper bound: 0.5379950
time: 1.97 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5253820, upper bound: 0.5387908
time: 2.04 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.8550828, 1.0039293, 0.7977136, 1.0058030, -0.1507202, 0.2062156
1: -0.0348245, 0.0104011, -0.0534744, 0.0464547, -0.0812792, 0.0638755
2: -0.0085443, 0.0556713, -0.0162534, 0.0988841, -0.1074284, 0.0719247
3: -0.0158086, 0.0514446, -0.0399657, 0.0760402, -0.0918488, 0.0914103
4: -0.0396240, 0.0125992, -0.0771955, 0.0349905, -0.0746145, 0.0897947
5: -0.0117870, 0.1252006, -0.0492999, 0.1642657, -0.1760528, 0.1745005
6: -0.0286936, 0.0288740, -0.0512418, 0.0538609, -0.0825544, 0.0801158
7: -0.0479097, 0.0248674, -0.0873605, 0.0474635, -0.0953732, 0.1122279
8: -0.0308302, 0.0467722, -0.0559138, 0.0833771, -0.1142073, 0.1026860
9: -0.0294314, 0.0354842, -0.0571943, 0.0829351, -0.1123665, 0.0926785

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5125949, upper bound: 0.4977326
time: 1.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5000955, upper bound: 0.4976151
time: 1.58 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.8545061, 1.0039310, 0.7591536, 1.0094599, -0.1549538, 0.2447774
1: -0.0349576, 0.0105810, -0.0592308, 0.0586517, -0.0936093, 0.0698118
2: -0.0085465, 0.0560611, -0.0203097, 0.1129642, -0.1215107, 0.0763708
3: -0.0159064, 0.0517084, -0.0496946, 0.0835509, -0.0994573, 0.1014030
4: -0.0398784, 0.0127428, -0.0888965, 0.0426802, -0.0825586, 0.1016394
5: -0.0117899, 0.1256195, -0.0645322, 0.1763178, -0.1881077, 0.1901516
6: -0.0288566, 0.0291090, -0.0580219, 0.0657311, -0.0945877, 0.0871309
7: -0.0482032, 0.0251098, -0.0980825, 0.0628677, -0.1110709, 0.1231923
8: -0.0310138, 0.0470192, -0.0635624, 0.0964134, -0.1274272, 0.1105816
9: -0.0297292, 0.0356182, -0.0650893, 0.0997087, -0.1294379, 0.1007075

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5129114, upper bound: 0.4990377
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5001051, upper bound: 0.4989272
time: 1.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.8385739, 1.0043248, 0.7949489, 1.0064702, -0.1678963, 0.2093759
1: -0.0386353, 0.0201003, -0.0545178, 0.0482107, -0.0868461, 0.0746181
2: -0.0090666, 0.0668300, -0.0169343, 0.1010738, -0.1101404, 0.0837643
3: -0.0207330, 0.0589946, -0.0412883, 0.0771760, -0.0979090, 0.1002829
4: -0.0484816, 0.0167112, -0.0791330, 0.0363150, -0.0847965, 0.0958442
5: -0.0184390, 0.1371922, -0.0514577, 0.1660696, -0.1845086, 0.1886498
6: -0.0341872, 0.0355996, -0.0524236, 0.0550880, -0.0892752, 0.0880232
7: -0.0580707, 0.0318036, -0.0893299, 0.0485068, -0.1065775, 0.1211334
8: -0.0360851, 0.0561186, -0.0572619, 0.0852399, -0.1213250, 0.1133805
9: -0.0379536, 0.0421344, -0.0584763, 0.0857969, -0.1237505, 0.1006107

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5122385, upper bound: 0.4978051
time: 1.57 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4989751, upper bound: 0.4976650
time: 1.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.8381522, 1.0043263, 0.7519824, 1.0101143, -0.1719621, 0.2523440
1: -0.0387326, 0.0203984, -0.0602611, 0.0608336, -0.0995663, 0.0806595
2: -0.0090686, 0.0671151, -0.0210353, 0.1154745, -0.1245431, 0.0881504
3: -0.0208823, 0.0591875, -0.0514509, 0.0848922, -0.1057746, 0.1106384
4: -0.0487253, 0.0168162, -0.0909877, 0.0440487, -0.0927740, 0.1078039
5: -0.0186746, 0.1374985, -0.0672505, 0.1784717, -0.1971463, 0.2047490
6: -0.0343367, 0.0357714, -0.0592345, 0.0679202, -0.1022569, 0.0950059
7: -0.0583497, 0.0319808, -0.1000021, 0.0657262, -0.1240759, 0.1319829
8: -0.0362194, 0.0563825, -0.0649306, 0.0987400, -0.1349594, 0.1213132
9: -0.0381714, 0.0423354, -0.0664996, 0.1027088, -0.1408802, 0.1088350

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5125619, upper bound: 0.4991166
time: 1.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4989866, upper bound: 0.4989866
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.8475226, 1.0045170, 0.8458595, 1.0050617, -0.1575391, 0.1586575
1: -0.0344528, 0.0111502, -0.0325976, 0.0100284, -0.0444813, 0.0437477
2: -0.0093207, 0.0555231, -0.0100555, 0.0510353, -0.0603560, 0.0655786
3: -0.0166171, 0.0512180, -0.0164351, 0.0480373, -0.0646544, 0.0676531
4: -0.0393484, 0.0124539, -0.0362244, 0.0107282, -0.0500766, 0.0486783
5: -0.0127443, 0.1249436, -0.0136440, 0.1200337, -0.1327780, 0.1385876
6: -0.0283471, 0.0303835, -0.0261801, 0.0295344, -0.0578815, 0.0565637
7: -0.0470183, 0.0281106, -0.0428585, 0.0289063, -0.0759246, 0.0709691
8: -0.0304056, 0.0473210, -0.0279323, 0.0451045, -0.0755101, 0.0752534
9: -0.0289359, 0.0355937, -0.0251360, 0.0341967, -0.0631326, 0.0607297

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5571186, upper bound: 0.5622095
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5567426, upper bound: 0.5622717
time: 2.12 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.8475226, 1.0045170, 0.8017706, 1.0052558, -0.1577332, 0.2027464
1: -0.0344528, 0.0111502, -0.0405371, 0.0292916, -0.0637444, 0.0516873
2: -0.0093207, 0.0555231, -0.0102476, 0.0750475, -0.0843682, 0.0657707
3: -0.0166171, 0.0512180, -0.0268848, 0.0642298, -0.0808470, 0.0781027
4: -0.0393484, 0.0124539, -0.0544451, 0.0194474, -0.0587957, 0.0668990
5: -0.0127443, 0.1249436, -0.0271561, 0.1458046, -0.1585488, 0.1520998
6: -0.0283471, 0.0303835, -0.0373944, 0.0455524, -0.0738995, 0.0677779
7: -0.0470183, 0.0281106, -0.0632525, 0.0470691, -0.0940873, 0.0913631
8: -0.0304056, 0.0473210, -0.0389634, 0.0647671, -0.0951727, 0.0862845
9: -0.0289359, 0.0355937, -0.0431388, 0.0473549, -0.0762908, 0.0787325

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 251

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 199

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5571186, upper bound: 0.5622085
time: 2.07 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5567426, upper bound: 0.5622728
time: 2.67 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: 0.6050580, 1.0060393, 0.8458595, 1.0050617, -0.4000038, 0.1601798
1: -0.0521063, 0.0787459, -0.0325976, 0.0100284, -0.0621348, 0.1113435
2: -0.0237023, 0.1198277, -0.0100555, 0.0510353, -0.0747376, 0.1298832
3: -0.0600483, 0.0933216, -0.0164351, 0.0480373, -0.1080855, 0.1097567
4: -0.0884303, 0.0344361, -0.0362244, 0.0107282, -0.0991585, 0.0706605
5: -0.0724631, 0.1930384, -0.0136440, 0.1200337, -0.1924968, 0.2066823
6: -0.0563515, 0.0994952, -0.0261801, 0.0295344, -0.0858859, 0.1256753
7: -0.0952857, 0.1288399, -0.0428585, 0.0289063, -0.1241920, 0.1716984
8: -0.0559764, 0.1118198, -0.0279323, 0.0451045, -0.1010809, 0.1397522
9: -0.0728827, 0.0766899, -0.0251360, 0.0341967, -0.1070794, 0.1018258

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5232894, upper bound: 0.5445009
time: 2.21 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5263193, upper bound: 0.5453594
time: 2.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: 0.6050580, 1.0060393, 0.8017706, 1.0052558, -0.4001979, 0.2042687
1: -0.0521063, 0.0787459, -0.0405371, 0.0292916, -0.0813979, 0.1192830
2: -0.0237023, 0.1198277, -0.0102476, 0.0750475, -0.0987498, 0.1300754
3: -0.0600483, 0.0933216, -0.0268848, 0.0642298, -0.1242781, 0.1202064
4: -0.0884303, 0.0344361, -0.0544451, 0.0194474, -0.1078777, 0.0888812
5: -0.0724631, 0.1930384, -0.0271561, 0.1458046, -0.2182676, 0.2201945
6: -0.0563515, 0.0994952, -0.0373944, 0.0455524, -0.1019039, 0.1368896
7: -0.0952857, 0.1288399, -0.0632525, 0.0470691, -0.1423548, 0.1920924
8: -0.0559764, 0.1118198, -0.0389634, 0.0647671, -0.1207435, 0.1507832
9: -0.0728827, 0.0766899, -0.0431388, 0.0473549, -0.1202377, 0.1198287

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 188

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 113

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5232894, upper bound: 0.5445009
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5263193, upper bound: 0.5453583
time: 1.99 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: 0.8911656, 1.0037946, 0.8113632, 1.0048403, -0.1136746, 0.1924313
1: -0.0256975, -0.0000509, -0.0465406, 0.0393406, -0.0650381, 0.0464897
2: -0.0083664, 0.0317502, -0.0134949, 0.0875314, -0.0958978, 0.0452450
3: -0.0105959, 0.0333759, -0.0329369, 0.0714389, -0.0820348, 0.0663128
4: -0.0221841, 0.0071858, -0.0693463, 0.0234888, -0.0456728, 0.0765321
5: -0.0115634, 0.0964803, -0.0351122, 0.1569575, -0.1685208, 0.1315925
6: -0.0181786, 0.0171351, -0.0438333, 0.0488892, -0.0670678, 0.0609684
7: -0.0319382, 0.0084424, -0.0792103, 0.0432363, -0.0751744, 0.0876527
8: -0.0190880, 0.0323816, -0.0504522, 0.0731515, -0.0922394, 0.0828337
9: -0.0121584, 0.0263263, -0.0520004, 0.0660764, -0.0782349, 0.0783267

Time for backsubstitution: 2.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5091367, upper bound: 0.4925432
time: 2.07 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4992416, upper bound: 0.4924547
time: 1.88 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: 0.8908312, 1.0037967, 0.7719873, 1.0068235, -0.1159923, 0.2318094
1: -0.0257903, 0.0000227, -0.0506813, 0.0520402, -0.0778305, 0.0507040
2: -0.0083693, 0.0318961, -0.0176553, 0.1005429, -0.1089122, 0.0495514
3: -0.0106317, 0.0335594, -0.0420309, 0.0792024, -0.0898341, 0.0755903
4: -0.0223613, 0.0071950, -0.0813590, 0.0275347, -0.0498960, 0.0885540
5: -0.0115669, 0.0967723, -0.0474542, 0.1694575, -0.1810245, 0.1442265
6: -0.0182699, 0.0172135, -0.0490946, 0.0613802, -0.0796501, 0.0663081
7: -0.0320261, 0.0084448, -0.0900161, 0.0595800, -0.0916062, 0.0984609
8: -0.0191825, 0.0325041, -0.0582657, 0.0850139, -0.1041964, 0.0907697
9: -0.0122990, 0.0264163, -0.0601175, 0.0799004, -0.0921994, 0.0865339

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5096347, upper bound: 0.4943439
time: 1.84 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4994631, upper bound: 0.4942536
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: 0.8771819, 1.0041431, 0.8087596, 1.0049760, -0.1277941, 0.1953835
1: -0.0295769, 0.0033304, -0.0473080, 0.0411815, -0.0707584, 0.0506384
2: -0.0088266, 0.0407052, -0.0142087, 0.0895277, -0.0983543, 0.0549139
3: -0.0122204, 0.0410482, -0.0341219, 0.0726296, -0.0848501, 0.0751701
4: -0.0295968, 0.0077376, -0.0713774, 0.0241373, -0.0537341, 0.0791150
5: -0.0121421, 0.1086878, -0.0367175, 0.1588487, -0.1709908, 0.1454054
6: -0.0222673, 0.0204143, -0.0447562, 0.0501758, -0.0724430, 0.0651705
7: -0.0369075, 0.0153161, -0.0812542, 0.0443301, -0.0812376, 0.0965703
8: -0.0235939, 0.0375027, -0.0518655, 0.0747812, -0.0983751, 0.0893682
9: -0.0181226, 0.0302047, -0.0533445, 0.0684415, -0.0865642, 0.0835492

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5087390, upper bound: 0.4926010
time: 8.22 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4981155, upper bound: 0.4925073
time: 1.48 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: 0.8768486, 1.0041449, 0.7639868, 1.0075090, -0.1306604, 0.2401581
1: -0.0296693, 0.0034532, -0.0514444, 0.0544306, -0.0840999, 0.0548976
2: -0.0088291, 0.0409335, -0.0184264, 0.1029711, -0.1118001, 0.0593599
3: -0.0122598, 0.0412314, -0.0437746, 0.0806528, -0.0929126, 0.0850059
4: -0.0297735, 0.0077524, -0.0835888, 0.0282846, -0.0580582, 0.0913412
5: -0.0121452, 0.1089787, -0.0497827, 0.1717965, -0.1839417, 0.1587614
6: -0.0223805, 0.0204924, -0.0500685, 0.0638523, -0.0862328, 0.0705609
7: -0.0370513, 0.0154844, -0.0920043, 0.0629071, -0.0999583, 0.1074888
8: -0.0237214, 0.0376248, -0.0597086, 0.0872497, -0.1109711, 0.0973333
9: -0.0182842, 0.0302978, -0.0616266, 0.0824636, -0.1007478, 0.0919243

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 93

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5093111, upper bound: 0.4944068
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.4983462, upper bound: 0.4943096
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: 0.8170037, 1.0044222, 0.8743925, 1.0047072, -0.1877035, 0.1300297
1: -0.0444564, 0.0318318, -0.0303507, 0.0043585, -0.0488148, 0.0621825
2: -0.0104319, 0.0804085, -0.0095715, 0.0426160, -0.0530479, 0.0899800
3: -0.0290451, 0.0664890, -0.0125501, 0.0425813, -0.0716264, 0.0790391
4: -0.0606660, 0.0236471, -0.0310755, 0.0078616, -0.0685276, 0.0547226
5: -0.0313584, 0.1491324, -0.0130790, 0.1111228, -0.1424811, 0.1622114
6: -0.0410825, 0.0442633, -0.0232149, 0.0210683, -0.0621508, 0.0674782
7: -0.0703095, 0.0403024, -0.0381110, 0.0167246, -0.0870341, 0.0784134
8: -0.0443075, 0.0678730, -0.0246610, 0.0385242, -0.0828317, 0.0925340
9: -0.0462891, 0.0584111, -0.0194744, 0.0309832, -0.0772723, 0.0778856

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 93

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449345, upper bound: 0.5334128
time: 1.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352469, upper bound: 0.5333344
time: 1.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: 0.8110968, 1.0045958, 0.8585290, 1.0050907, -0.1939939, 0.1460668
1: -0.0454885, 0.0339425, -0.0340290, 0.0093262, -0.0548147, 0.0679715
2: -0.0111479, 0.0828771, -0.0100779, 0.0533419, -0.0644897, 0.0929551
3: -0.0307009, 0.0677946, -0.0152238, 0.0498686, -0.0805696, 0.0830183
4: -0.0627280, 0.0250262, -0.0381039, 0.0117408, -0.0744689, 0.0631302
5: -0.0339797, 0.1512237, -0.0137160, 0.1226973, -0.1566770, 0.1649397
6: -0.0422884, 0.0461971, -0.0277194, 0.0274702, -0.0697585, 0.0739165
7: -0.0722366, 0.0426970, -0.0461561, 0.0234194, -0.0956561, 0.0888531
8: -0.0456709, 0.0701231, -0.0297332, 0.0452963, -0.0909672, 0.0998563
9: -0.0476743, 0.0613862, -0.0276524, 0.0346838, -0.0823581, 0.0890386

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 199
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 168
type: A, layer: 1, pos: 187
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 70
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 113
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 188
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 235
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 170
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 165
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 93

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5449237, upper bound: 0.5328370
time: 2.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5352814, upper bound: 0.5327568
time: 1.84 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.46 + 595.58 = 601.04 seconds
