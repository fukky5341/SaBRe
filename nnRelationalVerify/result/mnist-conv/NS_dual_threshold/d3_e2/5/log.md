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
execution time: IAR + RelationalAnalysis = 23.96 + 34.43 = 58.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.7649598, upper bound: 0.7649598

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 536
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 536

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
time: 4.61 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649481, upper bound: 0.7649500
time: 4.12 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.06 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 9.06
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
NS_A2, status: Status.UNKNOWN, split count: 1, time: 9.06
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

Time for backsubstitution: 21.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7510384
time: 4.46 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
time: 4.45 seconds

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

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 536
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 536

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649484, upper bound: 0.7510385
time: 3.94 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649484, upper bound: 0.7649487
time: 4.84 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.42 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 31.42
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7510384
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.42
Output dim: 3, lower bound: -0.7510384, upper bound: 0.7649483
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.42
Output dim: 3, lower bound: -0.7649484, upper bound: 0.7510385
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.42
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

Time for backsubstitution: 22.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7646574
time: 4.00 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510330, upper bound: 0.7649434
time: 4.70 seconds

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

Time for backsubstitution: 22.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 5817

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646553, upper bound: 0.7462585
time: 4.30 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649415, upper bound: 0.7510331
time: 4.08 seconds

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

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7601830, upper bound: 0.7507602
time: 4.33 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649430, upper bound: 0.7510333
time: 4.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.64 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.64
Output dim: 3, lower bound: -0.7462583, upper bound: 0.7646574
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.64
Output dim: 3, lower bound: -0.7510330, upper bound: 0.7649434
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 31.64
Output dim: 3, lower bound: -0.7646553, upper bound: 0.7462585
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 31.64
Output dim: 3, lower bound: -0.7649415, upper bound: 0.7510331
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 31.64
Output dim: 3, lower bound: -0.7601830, upper bound: 0.7507602
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.64
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

Time for backsubstitution: 22.24 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7451270, upper bound: 0.7646405
time: 4.79 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462425, upper bound: 0.7646406
time: 4.17 seconds

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

Time for backsubstitution: 22.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 442

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7499014, upper bound: 0.7649266
time: 4.37 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7510173, upper bound: 0.7649266
time: 4.70 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -10.2869205, -8.4729729, -10.2640533, -8.5491810, -1.2422204, 1.2616930
1: -4.4231052, -3.1039121, -4.3969965, -3.1145272, -0.9425387, 0.9265423
2: -5.5982180, -4.1438589, -5.5811148, -4.1738710, -1.1757278, 1.1900425
3: 5.5904307, 7.1865072, 5.6345739, 7.1662846, -1.1356206, 1.1280272
4: -14.5029736, -12.7429276, -14.4749680, -12.7497768, -1.1757293, 1.1552773
5: -7.6335306, -6.0285282, -7.5733805, -6.0504522, -1.1834245, 1.1702888
6: -11.4746799, -9.7593040, -11.4479856, -9.8321095, -1.0690866, 1.0843852
7: -6.1181364, -4.7269478, -6.1091843, -4.7313113, -0.9500966, 0.9451394
8: -4.7201777, -3.0316291, -4.6933947, -3.1426163, -1.0230103, 1.0550241
9: -5.1715221, -3.9271629, -5.1407452, -3.9348564, -0.9148583, 0.8959627

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 5817

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646405, upper bound: 0.7451272
time: 4.42 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646405, upper bound: 0.7462430
time: 4.78 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -10.2870998, -8.4714537, -10.2675781, -8.5447989, -1.2419186, 1.2648735
1: -4.4247799, -3.1038671, -4.4029140, -3.1111794, -0.9477992, 0.9270949
2: -5.5983877, -4.1420832, -5.5853395, -4.1681595, -1.1761992, 1.1961374
3: 5.5888596, 7.1866570, 5.6298428, 7.1697674, -1.1389861, 1.1285949
4: -14.5030346, -12.7430115, -14.4746199, -12.7500153, -1.1779857, 1.1548014
5: -7.6343727, -6.0283103, -7.5760689, -6.0480328, -1.1857924, 1.1712525
6: -11.4755154, -9.7592773, -11.4509430, -9.8315096, -1.0706363, 1.0875053
7: -6.1191339, -4.7268982, -6.1124468, -4.7291355, -0.9535115, 0.9447761
8: -4.7203960, -3.0313740, -4.6940947, -3.1410236, -1.0248158, 1.0556821
9: -5.1718740, -3.9263840, -5.1456165, -3.9325786, -0.9150655, 0.9020119

Time for backsubstitution: 22.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 442

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649266, upper bound: 0.7499014
time: 4.97 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7649266, upper bound: 0.7510172
time: 4.42 seconds

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

Time for backsubstitution: 22.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 442
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462852
time: 4.43 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462865
time: 4.17 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.11 seconds
NS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7451270, upper bound: 0.7646405
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7462425, upper bound: 0.7646406
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7499014, upper bound: 0.7649266
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7510173, upper bound: 0.7649266
NS_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7646405, upper bound: 0.7451272
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7646405, upper bound: 0.7462430
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7649266, upper bound: 0.7499014
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7649266, upper bound: 0.7510172
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462852
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.11
Output dim: 3, lower bound: -0.7646573, upper bound: 0.7462865

## BFS NS instance: NS_A1_B2_A1_A1

### Backsubstitution after applying NS history:
0: -10.2626991, -8.5491886, -10.2869205, -8.4729729, -1.2602582, 1.2421820
1: -4.3968925, -3.1148489, -4.4231052, -3.1039121, -0.9264405, 0.9422174
2: -5.5807366, -4.1740265, -5.5982180, -4.1438589, -1.1895928, 1.1755216
3: 5.6358504, 7.1661406, 5.5904307, 7.1865072, -1.1266880, 1.1354127
4: -14.4748545, -12.7498646, -14.5029736, -12.7429276, -1.1546474, 1.1751642
5: -7.5732679, -6.0529652, -7.6335306, -6.0285282, -1.1700897, 1.1807652
6: -11.4475164, -9.8322296, -11.4746799, -9.7593040, -1.0839090, 1.0689635
7: -6.1083126, -4.7313776, -6.1181364, -4.7269478, -0.9442348, 0.9500120
8: -4.6933775, -3.1444464, -4.7201777, -3.0316291, -1.0550053, 1.0211576
9: -5.1399002, -3.9348831, -5.1715221, -3.9271629, -0.8949809, 0.9148312

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A1_A1_B1

### Relational analysis result of NS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7451270, upper bound: 0.7627784
time: 4.10 seconds

## Relational analysis of NS_A1_B2_A1_A1_B2

### Relational analysis result of NS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7451270, upper bound: 0.7646405
time: 4.59 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -10.2667856, -8.5207939, -10.2869205, -8.4729710, -1.2663498, 1.2425475
1: -4.4110746, -3.1136901, -4.4231052, -3.1039128, -0.9416490, 0.9446800
2: -5.5836611, -4.1629491, -5.5982175, -4.1438589, -1.1931143, 1.1866870
3: 5.6313071, 7.1970549, 5.5904312, 7.1865096, -1.1348286, 1.1382616
4: -14.4785423, -12.7463245, -14.5029755, -12.7429276, -1.1566615, 1.1816013
5: -7.6283426, -6.0487461, -7.6335287, -6.0285349, -1.1730232, 1.1853471
6: -11.4568272, -9.8318501, -11.4746790, -9.7593040, -1.0885649, 1.0695715
7: -6.1122437, -4.7137289, -6.1181355, -4.7269468, -0.9502838, 0.9567901
8: -4.7332945, -3.1384525, -4.7201777, -3.0316315, -1.0552921, 1.0294995
9: -5.1451616, -3.9191809, -5.1715212, -3.9271631, -0.9037664, 0.9162445

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A1_A2_B1

### Relational analysis result of NS_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462425, upper bound: 0.7627783
time: 4.29 seconds

## Relational analysis of NS_A1_B2_A1_A2_B2

### Relational analysis result of NS_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7462425, upper bound: 0.7646406
time: 4.51 seconds

## BFS NS instance: NS_A1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -10.2662191, -8.5448046, -10.2870998, -8.4714537, -1.2634382, 1.2418807
1: -4.4028096, -3.1115007, -4.4247799, -3.1038671, -0.9269922, 0.9474781
2: -5.5849609, -4.1683154, -5.5983877, -4.1420832, -1.1956873, 1.1759920
3: 5.6311169, 7.1696262, 5.5888596, 7.1866570, -1.1272559, 1.1387784
4: -14.4745045, -12.7501040, -14.5030346, -12.7430115, -1.1541700, 1.1774201
5: -7.5759525, -6.0505409, -7.6343727, -6.0283103, -1.1710534, 1.1831341
6: -11.4504719, -9.8316288, -11.4755154, -9.7592773, -1.0870275, 1.0705135
7: -6.1115723, -4.7291985, -6.1191339, -4.7268982, -0.9438720, 0.9534278
8: -4.6940794, -3.1428547, -4.7203960, -3.0313740, -1.0556626, 1.0229630
9: -5.1447744, -3.9326036, -5.1718740, -3.9263840, -0.9010305, 0.9150381

Time for backsubstitution: 23.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.36 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7496014, upper bound: 0.7601663
time: 4.50 seconds

## Relational analysis of NS_A1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7496019, upper bound: 0.7601660
time: 4.95 seconds

## BFS NS instance: NS_A1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -10.2703037, -8.5164118, -10.2870932, -8.4714537, -1.2698135, 1.2422423
1: -4.4169760, -3.1103439, -4.4247789, -3.1038685, -0.9421844, 0.9499393
2: -5.5878839, -4.1572423, -5.5983877, -4.1420822, -1.1991758, 1.1871443
3: 5.6265712, 7.2005472, 5.5888600, 7.1866574, -1.1352992, 1.1416312
4: -14.4781857, -12.7465611, -14.5030365, -12.7430153, -1.1561804, 1.1821003
5: -7.6310377, -6.0463066, -7.6343741, -6.0283165, -1.1739907, 1.1879539
6: -11.4597549, -9.8312502, -11.4755154, -9.7592754, -1.0916784, 1.0711207
7: -6.1155005, -4.7115526, -6.1191330, -4.7268982, -0.9499102, 0.9585645
8: -4.7339959, -3.1368551, -4.7203960, -3.0313735, -1.0559511, 1.0315222
9: -5.1500597, -3.9168983, -5.1718721, -3.9263840, -0.9090555, 0.9164560

Time for backsubstitution: 22.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: B, layer: 1, pos: 442
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5817
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 64
type: A, layer: 1, pos: 5817

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 943

## Relational analysis of NS_A1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7507169, upper bound: 0.7601664
time: 4.14 seconds

## Relational analysis of NS_A1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_A2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7507174, upper bound: 0.7605512
time: 4.07 seconds

## BFS NS instance: NS_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -10.2869205, -8.4729729, -10.2626991, -8.5491886, -1.2421823, 1.2602582
1: -4.4231052, -3.1039121, -4.3968925, -3.1148489, -0.9422174, 0.9264402
2: -5.5982180, -4.1438589, -5.5807366, -4.1740265, -1.1755216, 1.1895928
3: 5.5904307, 7.1865072, 5.6358504, 7.1661406, -1.1354127, 1.1266881
4: -14.5029736, -12.7429276, -14.4748545, -12.7498646, -1.1751642, 1.1546474
5: -7.6335306, -6.0285282, -7.5732679, -6.0529652, -1.1807652, 1.1700900
6: -11.4746799, -9.7593040, -11.4475164, -9.8322296, -1.0689635, 1.0839090
7: -6.1181364, -4.7269478, -6.1083126, -4.7313776, -0.9500117, 0.9442348
8: -4.7201777, -3.0316291, -4.6933775, -3.1444464, -1.0211577, 1.0550053
9: -5.1715221, -3.9271629, -5.1399002, -3.9348831, -0.9148312, 0.8949811

Time for backsubstitution: 22.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 5857
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 5817

Time for candidate selection: 0.33 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A2_B1_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7627782, upper bound: 0.7451272
time: 4.24 seconds

## Relational analysis of NS_A2_B1_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7627782, upper bound: 0.7451270
time: 8.49 seconds

## BFS NS instance: NS_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -10.2869205, -8.4729710, -10.2667856, -8.5207939, -1.2425475, 1.2663498
1: -4.4231052, -3.1039128, -4.4110746, -3.1136901, -0.9446800, 0.9416490
2: -5.5982175, -4.1438589, -5.5836611, -4.1629491, -1.1866865, 1.1931143
3: 5.5904312, 7.1865096, 5.6313071, 7.1970549, -1.1382616, 1.1348284
4: -14.5029755, -12.7429276, -14.4785423, -12.7463245, -1.1816013, 1.1566610
5: -7.6335287, -6.0285349, -7.6283426, -6.0487461, -1.1853473, 1.1730230
6: -11.4746790, -9.7593040, -11.4568272, -9.8318501, -1.0695715, 1.0885649
7: -6.1181355, -4.7269468, -6.1122437, -4.7137289, -0.9567902, 0.9502840
8: -4.7201777, -3.0316315, -4.7332945, -3.1384525, -1.0294995, 1.0552921
9: -5.1715212, -3.9271631, -5.1451616, -3.9191809, -0.9162447, 0.9037664

Time for backsubstitution: 22.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: A, layer: 1, pos: 131
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5857
type: A, layer: 1, pos: 5817
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: B, layer: 1, pos: 5817

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A2_B1_B1_B2_A1

### Relational analysis result of NS_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7627782, upper bound: 0.7462426
time: 4.18 seconds

## Relational analysis of NS_A2_B1_B1_B2_A2

### Relational analysis result of NS_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.7627782, upper bound: 0.7462427
time: 5.89 seconds

## BFS NS instance: NS_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -10.2870998, -8.4714537, -10.2662191, -8.5448046, -1.2418804, 1.2634382
1: -4.4247799, -3.1038671, -4.4028096, -3.1115007, -0.9474778, 0.9269922
2: -5.5983877, -4.1420832, -5.5849609, -4.1683154, -1.1759920, 1.1956875
3: 5.5888596, 7.1866570, 5.6311169, 7.1696262, -1.1387784, 1.1272559
4: -14.5030346, -12.7430115, -14.4745045, -12.7501040, -1.1774201, 1.1541700
5: -7.6343727, -6.0283103, -7.5759525, -6.0505409, -1.1831341, 1.1710534
6: -11.4755154, -9.7592773, -11.4504719, -9.8316288, -1.0705130, 1.0870275
7: -6.1191339, -4.7268982, -6.1115723, -4.7291985, -0.9534278, 0.9438720
8: -4.7203960, -3.0313740, -4.6940794, -3.1428547, -1.0229630, 1.0556626
9: -5.1718740, -3.9263840, -5.1447744, -3.9326036, -0.9150381, 0.9010308

Time for backsubstitution: 22.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 943
type: B, layer: 1, pos: 131
type: A, layer: 1, pos: 131
type: A, layer: 1, pos: 442
type: A, layer: 1, pos: 4665
type: B, layer: 1, pos: 5857
type: B, layer: 1, pos: 4665
type: A, layer: 1, pos: 5817
type: A, layer: 1, pos: 63
type: B, layer: 1, pos: 63
type: A, layer: 1, pos: 64
type: B, layer: 1, pos: 64
type: A, layer: 1, pos: 5857
type: B, layer: 1, pos: 5817

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 943

## Relational analysis of NS_A2_B1_B2_B1_A1

### Relational analysis result of NS_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7601662, upper bound: 0.7496014
time: 4.68 seconds

## Relational analysis of NS_A2_B1_B2_B1_A2

### Relational analysis result of NS_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.7601662, upper bound: 0.7451271
time: 4.65 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.39 + 550.64 = 609.02 seconds
