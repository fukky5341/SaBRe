## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.761135001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.2870998, -8.4714384, -10.2870998, -8.4714384, -1.3250217, 1.3250217)
1: (-4.4247866, -3.1038651, -4.4247866, -3.1038651, -0.9551978, 0.9551978)
2: (-5.5983901, -4.1420717, -5.5983901, -4.1420717, -1.2082760, 1.2082758)
3: (5.5888500, 7.1866598, 5.5888500, 7.1866598, -1.1771312, 1.1771309)
4: (-14.5030413, -12.7428188, -14.5030413, -12.7428188, -1.1844063, 1.1844063)
5: (-7.6343851, -6.0283117, -7.6343851, -6.0283117, -1.2325692, 1.2325692)
6: (-11.4755154, -9.7592678, -11.4755154, -9.7592678, -1.1429083, 1.1429081)
7: (-6.1191373, -4.7268972, -6.1191373, -4.7268972, -0.9562483, 0.9562483)
8: (-4.7203951, -3.0313549, -4.7203951, -3.0313549, -1.1519935, 1.1519938)
9: (-5.1718807, -3.9263830, -5.1718807, -3.9263830, -0.9297078, 0.9297078)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.94 + 34.58 = 57.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.7649598, upper bound: 0.7649598

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
time: 4.50 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649481, upper bound: 0.7649500
time: 4.04 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.85
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.85
Output dim: 3, lower bound: -0.7649481, upper bound: 0.7649500

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.2675762, -8.5447874, -10.2865744, -8.4935942, -1.2678101, 1.2510595
1: -4.4029226, -3.1111782, -4.4183965, -3.1042633, -0.9324427, 0.9407456
2: -5.5853395, -4.1681480, -5.5972347, -4.1498270, -1.1883202, 1.1813648
3: 5.6298337, 7.1697688, 5.6011419, 7.1851101, -1.1346617, 1.1414118
4: -14.4746246, -12.7493382, -14.4945068, -12.7429733, -1.1563010, 1.1661477
5: -7.5760717, -6.0480328, -7.6167908, -6.0296378, -1.1734376, 1.1868165
6: -11.4509459, -9.8315115, -11.4735041, -9.7811041, -1.0876963, 1.0686278
7: -6.1124492, -4.7291355, -6.1172724, -4.7271070, -0.9484534, 0.9508204
8: -4.6940975, -3.1410227, -4.7200871, -3.0643144, -1.0551949, 1.0422616
9: -5.1456189, -3.9325752, -5.1644073, -3.9265993, -0.9020061, 0.9145904

Time for backsubstitution: 21.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7510384
time: 4.35 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
time: 4.37 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.2871017, -8.4714489, -10.2870998, -8.4714384, -1.3204339, 1.2590466
1: -4.4247818, -3.1038661, -4.4247866, -3.1038651, -0.9389150, 0.9551978
2: -5.5983896, -4.1420784, -5.5983901, -4.1420717, -1.2082756, 1.1873574
3: 5.5888567, 7.1866589, 5.5888500, 7.1866598, -1.1431341, 1.1745310
4: -14.5030327, -12.7428207, -14.5030413, -12.7428188, -1.1705532, 1.1829133
5: -7.6343751, -6.0283108, -7.6343851, -6.0283117, -1.1794205, 1.2313757
6: -11.4755144, -9.7592764, -11.4755154, -9.7592678, -1.1429064, 1.0790203
7: -6.1191349, -4.7268991, -6.1191373, -4.7268972, -0.9549584, 0.9561050
8: -4.7203956, -3.0313711, -4.7203951, -3.0313549, -1.1348341, 1.0532589
9: -5.1718755, -3.9263833, -5.1718807, -3.9263830, -0.9115934, 0.9294877

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649484, upper bound: 0.7510385
time: 3.95 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649484, upper bound: 0.7649487
time: 4.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.29 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.29
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7510384
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 3, lower bound: -0.7649484, upper bound: 0.7510385
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 3, lower bound: -0.7649484, upper bound: 0.7649487

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.2675762, -8.5447874, -10.2871017, -8.4714489, -1.2678888, 1.2469716
1: -4.4029226, -3.1111782, -4.4247818, -3.1038661, -0.9328303, 0.9479005
2: -5.5853395, -4.1681480, -5.5983896, -4.1420784, -1.1961751, 1.1821063
3: 5.6298337, 7.1697688, 5.5888567, 7.1866589, -1.1332760, 1.1418055
4: -14.4746246, -12.7493382, -14.5030327, -12.7428207, -1.1550298, 1.1764474
5: -7.5760717, -6.0480328, -7.6343751, -6.0283108, -1.1734662, 1.1872222
6: -11.4509459, -9.8315115, -11.4755144, -9.7592764, -1.0876966, 1.0706377
7: -6.1124492, -4.7291355, -6.1191349, -4.7268991, -0.9486485, 0.9536850
8: -4.6940975, -3.1410227, -4.7203956, -3.0313711, -1.0560308, 1.0251554
9: -5.1456189, -3.9325752, -5.1718755, -3.9263833, -0.9020147, 0.9176145

Time for backsubstitution: 22.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7646574
time: 3.96 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510330, upper bound: 0.7649434
time: 4.61 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.2871017, -8.4714489, -10.2675762, -8.5447874, -1.2469716, 1.2678890
1: -4.4247818, -3.1038661, -4.4029226, -3.1111782, -0.9479005, 0.9328306
2: -5.5983896, -4.1420784, -5.5853395, -4.1681480, -1.1821060, 1.1961751
3: 5.5888567, 7.1866589, 5.6298337, 7.1697688, -1.1418056, 1.1332761
4: -14.5030327, -12.7428207, -14.4746246, -12.7493382, -1.1764474, 1.1550298
5: -7.6343751, -6.0283108, -7.5760717, -6.0480328, -1.1872222, 1.1734664
6: -11.4755144, -9.7592764, -11.4509459, -9.8315115, -1.0706379, 1.0876966
7: -6.1191349, -4.7268991, -6.1124492, -4.7291355, -0.9536850, 0.9486485
8: -4.7203956, -3.0313711, -4.6940975, -3.1410227, -1.0251555, 1.0560308
9: -5.1718755, -3.9263833, -5.1456189, -3.9325752, -0.9176142, 0.9020147

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7601811, upper bound: 0.7507332
time: 4.45 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649411, upper bound: 0.7510335
time: 4.74 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.2871017, -8.4714489, -10.2871017, -8.4714489, -1.2590466, 1.2590466
1: -4.4247818, -3.1038661, -4.4247818, -3.1038661, -0.9389150, 0.9389150
2: -5.5983896, -4.1420784, -5.5983896, -4.1420784, -1.1873569, 1.1873567
3: 5.5888567, 7.1866589, 5.5888567, 7.1866589, -1.1431339, 1.1431336
4: -14.5030327, -12.7428207, -14.5030327, -12.7428207, -1.1705523, 1.1705523
5: -7.6343751, -6.0283108, -7.6343751, -6.0283108, -1.1794190, 1.1794188
6: -11.4755144, -9.7592764, -11.4755144, -9.7592764, -1.0790191, 1.0790188
7: -6.1191349, -4.7268991, -6.1191349, -4.7268991, -0.9549587, 0.9549584
8: -4.7203956, -3.0313711, -4.7203956, -3.0313711, -1.0532582, 1.0532579
9: -5.1718755, -3.9263833, -5.1718755, -3.9263833, -0.9115925, 0.9115925

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7601830, upper bound: 0.7507602
time: 4.31 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649430, upper bound: 0.7510333
time: 4.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.49 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.49
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7646574
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.49
Output dim: 3, lower bound: -0.7510330, upper bound: 0.7649434
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 31.49
Output dim: 3, lower bound: -0.7601811, upper bound: 0.7507332
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.49
Output dim: 3, lower bound: -0.7649411, upper bound: 0.7510335
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 31.49
Output dim: 3, lower bound: -0.7601830, upper bound: 0.7507602
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.49
Output dim: 3, lower bound: -0.7649430, upper bound: 0.7510333

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.2640533, -8.5491810, -10.2869205, -8.4729729, -1.2616930, 1.2422204
1: -4.3969965, -3.1145272, -4.4231052, -3.1039121, -0.9265423, 0.9425387
2: -5.5811148, -4.1738710, -5.5982180, -4.1438589, -1.1900425, 1.1757278
3: 5.6345739, 7.1662846, 5.5904307, 7.1865072, -1.1280272, 1.1356206
4: -14.4749680, -12.7497768, -14.5029736, -12.7429276, -1.1552773, 1.1757293
5: -7.5733805, -6.0504522, -7.6335306, -6.0285282, -1.1702888, 1.1834242
6: -11.4479856, -9.8321095, -11.4746799, -9.7593040, -1.0843852, 1.0690866
7: -6.1091843, -4.7313113, -6.1181364, -4.7269478, -0.9451394, 0.9500964
8: -4.6933947, -3.1426163, -4.7201777, -3.0316291, -1.0550241, 1.0230101
9: -5.1407452, -3.9348564, -5.1715221, -3.9271629, -0.8959622, 0.9148583

Time for backsubstitution: 22.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7601812
time: 4.19 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7646554
time: 3.97 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.2675781, -8.5447989, -10.2870998, -8.4714537, -1.2648735, 1.2419183
1: -4.4029140, -3.1111794, -4.4247799, -3.1038671, -0.9270949, 0.9477992
2: -5.5853395, -4.1681595, -5.5983877, -4.1420832, -1.1961374, 1.1761994
3: 5.6298428, 7.1697674, 5.5888596, 7.1866570, -1.1285949, 1.1389861
4: -14.4746199, -12.7500153, -14.5030346, -12.7430115, -1.1548014, 1.1779857
5: -7.5760689, -6.0480328, -7.6343727, -6.0283103, -1.1712527, 1.1857922
6: -11.4509430, -9.8315096, -11.4755154, -9.7592773, -1.0875053, 1.0706365
7: -6.1124468, -4.7291355, -6.1191339, -4.7268982, -0.9447761, 0.9535117
8: -4.6940947, -3.1410236, -4.7203960, -3.0313740, -1.0556822, 1.0248158
9: -5.1456165, -3.9325786, -5.1718740, -3.9263840, -0.9020114, 0.9150655

Time for backsubstitution: 22.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7507332, upper bound: 0.7601812
time: 4.20 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7507332, upper bound: 0.7601811
time: 4.24 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.2870970, -8.4714584, -10.2675781, -8.5447884, -1.2439568, 1.2628310
1: -4.4247723, -3.1038666, -4.4029207, -3.1111791, -0.9421651, 0.9327281
2: -5.5983882, -4.1420884, -5.5853386, -4.1681509, -1.1820683, 1.1902690
3: 5.5888658, 7.1866560, 5.6298351, 7.1697650, -1.1371164, 1.1304563
4: -14.5030327, -12.7435007, -14.4746227, -12.7495279, -1.1762176, 1.1565681
5: -7.6343708, -6.0283136, -7.5760708, -6.0480323, -1.1850033, 1.1720488
6: -11.4755125, -9.7592745, -11.4509459, -9.8315105, -1.0705376, 1.0874144
7: -6.1191320, -4.7268982, -6.1124477, -4.7291327, -0.9498136, 0.9484723
8: -4.7203951, -3.0313716, -4.6940956, -3.1410217, -1.0248024, 1.0556912
9: -5.1718736, -3.9263849, -5.1456184, -3.9325762, -0.9156296, 0.8997595

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646553, upper bound: 0.7462585
time: 4.47 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646553, upper bound: 0.7462596
time: 3.73 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.2870970, -8.4714584, -10.2870998, -8.4714537, -1.2588468, 1.2536812
1: -4.4247723, -3.1038666, -4.4247799, -3.1038671, -0.9331789, 0.9388127
2: -5.5983882, -4.1420884, -5.5983877, -4.1420832, -1.1873200, 1.1814518
3: 5.5888658, 7.1866560, 5.5888596, 7.1866570, -1.1381116, 1.1429121
4: -14.5030327, -12.7435007, -14.5030346, -12.7430115, -1.1703234, 1.1720910
5: -7.6343708, -6.0283136, -7.6343727, -6.0283103, -1.1772966, 1.1794171
6: -11.4755125, -9.7592745, -11.4755154, -9.7592773, -1.0789189, 1.0790174
7: -6.1191320, -4.7268982, -6.1191339, -4.7268982, -0.9510860, 0.9547815
8: -4.7203951, -3.0313716, -4.7203960, -3.0313740, -1.0526106, 1.0530813
9: -5.1718736, -3.9263849, -5.1718740, -3.9263840, -0.9115901, 0.9093370

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462852
time: 4.41 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462865
time: 4.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.84 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7601812
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7646554
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7507332, upper bound: 0.7601812
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7507332, upper bound: 0.7601811
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7646553, upper bound: 0.7462585
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7646553, upper bound: 0.7462596
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462852
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.84
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462865

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -10.2640533, -8.5491810, -10.2870970, -8.4714584, -1.2617307, 1.2394073
1: -4.3969965, -3.1145272, -4.4247723, -3.1038666, -0.9265101, 0.9443944
2: -5.5811148, -4.1738710, -5.5983882, -4.1420884, -1.1918929, 1.1758742
3: 5.6345739, 7.1662846, 5.5888658, 7.1866560, -1.1253984, 1.1358083
4: -14.4749680, -12.7497768, -14.5030327, -12.7435007, -1.1545663, 1.1759257
5: -7.5733805, -6.0504522, -7.6343708, -6.0283136, -1.1691074, 1.1835785
6: -11.4479856, -9.8321095, -11.4755125, -9.7592745, -1.0841520, 1.0700073
7: -6.1091843, -4.7313113, -6.1191320, -4.7268982, -0.9450357, 0.9512851
8: -4.6933947, -3.1426163, -4.7203951, -3.0313716, -1.0549605, 1.0230865
9: -5.1407452, -3.9348564, -5.1718736, -3.9263849, -0.8967421, 0.9133077

Time for backsubstitution: 21.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7451270, upper bound: 0.7646405
time: 4.54 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462425, upper bound: 0.7646406
time: 4.13 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.2870970, -8.4714584, -10.2640533, -8.5491810, -1.2394075, 1.2617304
1: -4.4247723, -3.1038666, -4.3969965, -3.1145272, -0.9443946, 0.9265101
2: -5.5983882, -4.1420884, -5.5811148, -4.1738710, -1.1758742, 1.1918929
3: 5.5888658, 7.1866560, 5.6345739, 7.1662846, -1.1358082, 1.1253984
4: -14.5030327, -12.7435007, -14.4749680, -12.7497768, -1.1759257, 1.1545663
5: -7.6343708, -6.0283136, -7.5733805, -6.0504522, -1.1835785, 1.1691074
6: -11.4755125, -9.7592745, -11.4479856, -9.8321095, -1.0700073, 1.0841519
7: -6.1191320, -4.7268982, -6.1091843, -4.7313113, -0.9512849, 0.9450357
8: -4.7203951, -3.0313716, -4.6933947, -3.1426163, -1.0230865, 1.0549606
9: -5.1718736, -3.9263849, -5.1407452, -3.9348564, -0.9133077, 0.8967421

Time for backsubstitution: 21.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7635170, upper bound: 0.7462447
time: 3.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646391, upper bound: 0.7462439
time: 4.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.2870970, -8.4714584, -10.2675781, -8.5447989, -1.2432837, 1.2628307
1: -4.4247723, -3.1038666, -4.4029140, -3.1111794, -0.9420938, 0.9270232
2: -5.5983882, -4.1420884, -5.5853395, -4.1681595, -1.1761749, 1.1902440
3: 5.5888658, 7.1866560, 5.6298428, 7.1697674, -1.1371160, 1.1307192
4: -14.5030327, -12.7435007, -14.4746199, -12.7500153, -1.1779857, 1.1565676
5: -7.6343708, -6.0283136, -7.5760689, -6.0480328, -1.1848707, 1.1713388
6: -11.4755125, -9.7592745, -11.4509430, -9.8315096, -1.0705376, 1.0874127
7: -6.1191320, -4.7268982, -6.1124468, -4.7291355, -0.9496903, 0.9446497
8: -4.7203951, -3.0313716, -4.6940947, -3.1410236, -1.0248017, 1.0559410
9: -5.1718736, -3.9263849, -5.1456165, -3.9325786, -0.9156294, 0.8997581

Time for backsubstitution: 21.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7635174, upper bound: 0.7462438
time: 4.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646396, upper bound: 0.7466554
time: 4.02 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.2870970, -8.4714584, -10.2835798, -8.4758396, -1.2543073, 1.2553105
1: -4.4247723, -3.1038666, -4.4188619, -3.1072130, -0.9354088, 0.9325824
2: -5.5983882, -4.1420884, -5.5941639, -4.1478071, -1.1811173, 1.1831131
3: 5.5888658, 7.1866560, 5.5935993, 7.1831865, -1.1393542, 1.1378994
4: -14.5030327, -12.7435007, -14.5033827, -12.7432604, -1.1700320, 1.1700907
5: -7.6343708, -6.0283136, -7.6316829, -6.0307202, -1.1770315, 1.1764812
6: -11.4755125, -9.7592745, -11.4725523, -9.7598772, -1.0783877, 1.0757709
7: -6.1191320, -4.7268982, -6.1158743, -4.7290702, -0.9525578, 0.9513381
8: -4.7203951, -3.0313716, -4.7196932, -3.0329571, -1.0514507, 1.0523500
9: -5.1718736, -3.9263849, -5.1670270, -3.9286644, -0.9092689, 0.9063063

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7635183, upper bound: 0.7462706
time: 4.38 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646405, upper bound: 0.7462704
time: 3.98 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.2870970, -8.4714584, -10.2870970, -8.4714584, -1.2535405, 1.2535405
1: -4.4247723, -3.1038666, -4.4247723, -3.1038666, -0.9331074, 0.9331074
2: -5.5983882, -4.1420884, -5.5983882, -4.1420884, -1.1814265, 1.1814265
3: 5.5888658, 7.1866560, 5.5888658, 7.1866560, -1.1379561, 1.1379559
4: -14.5030327, -12.7435007, -14.5030327, -12.7435007, -1.1720901, 1.1720901
5: -7.6343708, -6.0283136, -7.6343708, -6.0283136, -1.1772957, 1.1772959
6: -11.4755125, -9.7592745, -11.4755125, -9.7592745, -1.0789192, 1.0789192
7: -6.1191320, -4.7268982, -6.1191320, -4.7268982, -0.9509597, 0.9509599
8: -4.7203951, -3.0313716, -4.7203951, -3.0313716, -1.0524848, 1.0524843
9: -5.1718736, -3.9263849, -5.1718736, -3.9263849, -0.9093356, 0.9093359

Time for backsubstitution: 21.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7635187, upper bound: 0.7466765
time: 4.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646415, upper bound: 0.7462704
time: 3.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 30.32 seconds
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7451270, upper bound: 0.7646405
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7462425, upper bound: 0.7646406
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7635170, upper bound: 0.7462447
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7646391, upper bound: 0.7462439
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7635174, upper bound: 0.7462438
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7646396, upper bound: 0.7466554
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7635183, upper bound: 0.7462706
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7646405, upper bound: 0.7462704
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7635187, upper bound: 0.7466765
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 30.32
Output dim: 3, lower bound: -0.7646415, upper bound: 0.7462704

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.2626991, -8.5491886, -10.2870970, -8.4714584, -1.2602954, 1.2393692
1: -4.3968925, -3.1148489, -4.4247723, -3.1038666, -0.9264083, 0.9440732
2: -5.5807366, -4.1740265, -5.5983882, -4.1420884, -1.1914432, 1.1756678
3: 5.6358504, 7.1661406, 5.5888658, 7.1866560, -1.1240592, 1.1356006
4: -14.4748545, -12.7498646, -14.5030327, -12.7435007, -1.1539364, 1.1753612
5: -7.5732679, -6.0529652, -7.6343708, -6.0283136, -1.1689086, 1.1809192
6: -11.4475164, -9.8322296, -11.4755125, -9.7592745, -1.0836756, 1.0698848
7: -6.1083126, -4.7313776, -6.1191320, -4.7268982, -0.9441311, 0.9512007
8: -4.6933775, -3.1444464, -4.7203951, -3.0313716, -1.0549417, 1.0212340
9: -5.1399002, -3.9348831, -5.1718736, -3.9263849, -0.8957603, 0.9132805

Time for backsubstitution: 21.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7451260, upper bound: 0.7635160
time: 4.60 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7451260, upper bound: 0.7646401
time: 4.64 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.2667856, -8.5207939, -10.2870951, -8.4714584, -1.2663870, 1.2397346
1: -4.4110746, -3.1136901, -4.4247723, -3.1038673, -0.9414845, 0.9465356
2: -5.5836611, -4.1629491, -5.5983868, -4.1420884, -1.1949644, 1.1868324
3: 5.6313071, 7.1970549, 5.5888658, 7.1866574, -1.1322069, 1.1384488
4: -14.4785423, -12.7463245, -14.5030336, -12.7434998, -1.1559501, 1.1808741
5: -7.6283426, -6.0487461, -7.6343689, -6.0283165, -1.1718426, 1.1855006
6: -11.4568272, -9.8318501, -11.4755135, -9.7592754, -1.0883315, 1.0704923
7: -6.1122437, -4.7137289, -6.1191311, -4.7268991, -0.9501803, 0.9569345
8: -4.7332945, -3.1384525, -4.7203951, -3.0313759, -1.0552292, 1.0295668
9: -5.1451616, -3.9191809, -5.1718712, -3.9263854, -0.9038434, 0.9146941

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462434, upper bound: 0.7635160
time: 4.66 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462434, upper bound: 0.7646393
time: 3.94 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -10.2857399, -8.4714642, -10.2640533, -8.5491810, -1.2379713, 1.2616923
1: -4.4246678, -3.1041889, -4.3969965, -3.1145272, -0.9442906, 0.9261887
2: -5.5980110, -4.1422486, -5.5811148, -4.1738710, -1.1754231, 1.1916819
3: 5.5901403, 7.1865172, 5.6345739, 7.1662846, -1.1344690, 1.1251910
4: -14.5029240, -12.7435932, -14.4749680, -12.7497768, -1.1752977, 1.1540012
5: -7.6342545, -6.0308218, -7.5733805, -6.0504522, -1.1833797, 1.1664491
6: -11.4750538, -9.7593975, -11.4479856, -9.8321095, -1.0695429, 1.0840282
7: -6.1182613, -4.7269621, -6.1091843, -4.7313113, -0.9503827, 0.9449537
8: -4.7203784, -3.0332041, -4.6933947, -3.1426163, -1.0230672, 1.0531088
9: -5.1710186, -3.9264131, -5.1407452, -3.9348564, -0.9123185, 0.8967142

Time for backsubstitution: 22.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 63

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7635160, upper bound: 0.7451273
time: 3.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7635160, upper bound: 0.7462447
time: 3.93 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.2897987, -8.4430771, -10.2640505, -8.5491819, -1.2441947, 1.2620559
1: -4.4388275, -3.1030407, -4.3969970, -3.1145277, -0.9531342, 0.9286306
2: -5.6008787, -4.1310768, -5.5811133, -4.1738710, -1.1789131, 1.1967030
3: 5.5855942, 7.2173734, 5.6345739, 7.1662827, -1.1424577, 1.1280246
4: -14.5066528, -12.7400475, -14.4749689, -12.7497768, -1.1774325, 1.1708093
5: -7.6895151, -6.0265751, -7.5733795, -6.0504570, -1.1863341, 1.1707568
6: -11.4842281, -9.7590179, -11.4479885, -9.8321114, -1.0776095, 1.0843005
7: -6.1221695, -4.7093306, -6.1091819, -4.7313128, -0.9564493, 0.9507172
8: -4.7602959, -3.0271883, -4.6933956, -3.1426182, -1.0233524, 1.0621823
9: -5.1763473, -3.9107084, -5.1407418, -3.9348569, -0.9191308, 0.9000909

Time for backsubstitution: 22.47 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.52 + 554.05 = 611.56 seconds
