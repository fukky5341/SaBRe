## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.23880936800000002


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-14.1278248, -13.1164131, -14.1278248, -13.1164131, -0.4508286, 0.4508287)
1: (-7.6832905, -6.7711983, -7.6832905, -6.7711983, -0.4796782, 0.4796782)
2: (2.9860601, 3.9267602, 2.9860601, 3.9267602, -0.6314707, 0.6314707)
3: (0.4996758, 1.3109438, 0.4996758, 1.3109438, -0.5271082, 0.5271082)
4: (-6.9687753, -6.0825634, -6.9687753, -6.0825634, -0.5958090, 0.5958092)
5: (-5.8701153, -4.9690604, -5.8701153, -4.9690604, -0.5077269, 0.5077269)
6: (-11.7345448, -10.5224504, -11.7345448, -10.5224504, -0.5451224, 0.5451224)
7: (-0.7013762, 0.0902328, -0.7013762, 0.0902328, -0.4733357, 0.4733357)
8: (-3.6651654, -2.8292532, -3.6651654, -2.8292532, -0.5264854, 0.5264852)
9: (-9.5412092, -8.4602833, -9.5412092, -8.4602833, -0.4916582, 0.4916582)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.47 + 32.68 = 55.14 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.2595754, upper bound: 0.2595754

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1103
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 1103

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431952, upper bound: 0.2529797
time: 2.93 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2463280, upper bound: 0.2463279
time: 2.86 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.23 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.23
Output dim: 3, lower bound: -0.2431952, upper bound: 0.2529797
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.23
Output dim: 3, lower bound: -0.2463280, upper bound: 0.2463279

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -14.1198282, -13.1325445, -14.1278152, -13.1217270, -0.4395416, 0.4381816
1: -7.6754642, -6.8137913, -7.6832428, -6.7858410, -0.4427769, 0.4355962
2: 3.0116167, 3.9312010, 2.9951777, 3.9262028, -0.6079779, 0.6316767
3: 0.4800911, 1.2981946, 0.5002279, 1.3065939, -0.5232682, 0.5056891
4: -6.9616060, -6.0790162, -6.9662838, -6.0825639, -0.5879521, 0.5959561
5: -5.8443642, -4.9777365, -5.8616505, -4.9690962, -0.4868681, 0.4949665
6: -11.7140036, -10.5325069, -11.7277527, -10.5226879, -0.5280523, 0.5356214
7: -0.6841836, 0.0937436, -0.6956723, 0.0901227, -0.4520175, 0.4649923
8: -3.6544161, -2.8709717, -3.6650290, -2.8433406, -0.4739289, 0.4718940
9: -9.5297289, -8.4863558, -9.5411787, -8.4688263, -0.4749277, 0.4686649

Time for backsubstitution: 8.62 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 2131

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
time: 3.07 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2362359, upper bound: 0.2462170
time: 3.10 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -14.1278172, -13.1180420, -14.1278257, -13.1168022, -0.4481146, 0.4414500
1: -7.6832528, -6.7845669, -7.6832819, -6.7746367, -0.4773278, 0.4314884
2: 3.0014381, 3.9263191, 2.9896865, 3.9266458, -0.6210074, 0.6277909
3: 0.5001144, 1.2905920, 0.4997878, 1.3060856, -0.5248725, 0.5018830
4: -6.9645395, -6.0825663, -6.9677505, -6.0825648, -0.5974042, 0.5929031
5: -5.8641930, -4.9690814, -5.8687072, -4.9690666, -0.4916186, 0.5037849
6: -11.7305813, -10.5226126, -11.7336073, -10.5224924, -0.5329032, 0.5439434
7: -0.6912494, 0.0901403, -0.6989675, 0.0902092, -0.4536686, 0.4719350
8: -3.6650419, -2.8415611, -3.6651359, -2.8321533, -0.5260372, 0.4511812
9: -9.5411921, -8.4648104, -9.5412054, -8.4613562, -0.4903958, 0.4701620

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 2131

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2399676
time: 3.00 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2399676, upper bound: 0.2399676
time: 2.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 14.86 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 14.86
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 14.86
Output dim: 3, lower bound: -0.2362359, upper bound: 0.2462170
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 14.86
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2399676
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 14.86
Output dim: 3, lower bound: -0.2399676, upper bound: 0.2399676

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -14.1197929, -13.1388874, -14.1278095, -13.1229095, -0.4381664, 0.4312657
1: -7.6657677, -6.8138075, -7.6816826, -6.7858434, -0.4321866, 0.4339862
2: 3.0116506, 3.9084082, 2.9951854, 3.9221635, -0.6044655, 0.6108413
3: 0.4971709, 1.2975218, 0.5034158, 1.3064771, -0.5022159, 0.5019348
4: -6.9603667, -6.0848308, -6.9660463, -6.0837193, -0.5846024, 0.5849121
5: -5.8427539, -4.9777393, -5.8613596, -4.9690981, -0.4850323, 0.4946618
6: -11.7012482, -10.5325184, -11.7252970, -10.5226917, -0.5187345, 0.5336497
7: -0.6840563, 0.0764937, -0.6956482, 0.0870359, -0.4490588, 0.4484177
8: -3.6528485, -2.8714333, -3.6647215, -2.8434224, -0.4729915, 0.4713976
9: -9.5282259, -8.4865980, -9.5408974, -8.4688692, -0.4732904, 0.4681013

Time for backsubstitution: 8.08 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2131

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2331090
time: 3.17 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
time: 3.12 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1278048, -13.1237726, -0.4499936, 0.4328758
1: -7.6686587, -6.8011436, -7.6808734, -6.7858429, -0.4345449, 0.4494877
2: 2.9750814, 3.9252968, 2.9951963, 3.9231272, -0.6343517, 0.6193395
3: 0.4928286, 1.3196907, 0.5047483, 1.3064032, -0.5082846, 0.5361915
4: -6.9649348, -6.0918393, -6.9659066, -6.0868120, -0.6009502, 0.5800743
5: -5.8433108, -4.9774771, -5.8609066, -4.9690962, -0.4858654, 0.4958491
6: -11.7114992, -10.5123320, -11.7269897, -10.5226936, -0.5247540, 0.5444503
7: -0.7068467, 0.0818155, -0.6956229, 0.0853450, -0.4707086, 0.4539483
8: -3.6529422, -2.8693893, -3.6645942, -2.8434758, -0.4720178, 0.4710653
9: -9.5267200, -8.4853220, -9.5401535, -8.4688969, -0.4732091, 0.4694687

Time for backsubstitution: 8.18 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2362359, upper bound: 0.2462169
time: 3.26 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2362359, upper bound: 0.2462169
time: 3.25 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -14.1277828, -13.1243706, -14.1278152, -13.1179848, -0.4467387, 0.4344429
1: -7.6749611, -6.7845874, -7.6817212, -6.7746410, -0.4694154, 0.4293541
2: 3.0014744, 3.9047198, 2.9896927, 3.9226122, -0.6170521, 0.6097283
3: 0.5171700, 1.2899661, 0.5029712, 1.3059683, -0.5058539, 0.4978695
4: -6.9632931, -6.0874853, -6.9675217, -6.0837193, -0.5940318, 0.5817783
5: -5.8626537, -4.9690847, -5.8684206, -4.9690685, -0.4897966, 0.5034659
6: -11.7192116, -10.5226269, -11.7311468, -10.5224953, -0.5235081, 0.5419059
7: -0.6911311, 0.0736084, -0.6989448, 0.0871222, -0.4503236, 0.4577823
8: -3.6635315, -2.8419929, -3.6648312, -2.8322320, -0.5251050, 0.4507041
9: -9.5396852, -8.4650412, -9.5409241, -8.4613953, -0.4890239, 0.4696137

Time for backsubstitution: 8.33 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362360
time: 3.14 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362367
time: 3.14 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -14.1364698, -13.1227446, -14.1278143, -13.1188612, -0.4577396, 0.4360303
1: -7.6770892, -6.7727680, -7.6809087, -6.7746396, -0.4719996, 0.4471735
2: 2.9678011, 3.9180298, 2.9897046, 3.9235754, -0.6477928, 0.6177764
3: 0.5129824, 1.3119073, 0.5043054, 1.3058929, -0.5110402, 0.5323822
4: -6.9668255, -6.0950527, -6.9673858, -6.0868125, -0.6102412, 0.5769508
5: -5.8629947, -4.9688282, -5.8679757, -4.9690695, -0.4904876, 0.5048556
6: -11.7282677, -10.5030575, -11.7328396, -10.5225010, -0.5287027, 0.5547876
7: -0.7128882, 0.0767579, -0.6989205, 0.0854337, -0.4729424, 0.4630213
8: -3.6638453, -2.8401210, -3.6646998, -2.8322830, -0.5240345, 0.4502892
9: -9.5381775, -8.4638376, -9.5401840, -8.4614248, -0.4889460, 0.4709253

Time for backsubstitution: 8.33 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2394754, upper bound: 0.2362360
time: 3.03 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2394756, upper bound: 0.2362367
time: 3.28 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 14.84 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2331090
NS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
NS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2362359, upper bound: 0.2462169
NS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2362359, upper bound: 0.2462169
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362360
NS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362367
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2394754, upper bound: 0.2362360
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 14.84
Output dim: 3, lower bound: -0.2394756, upper bound: 0.2362367

## BFS NS instance: NS_A1_A1_B2

### Backsubstitution after applying NS history:
0: -14.1197929, -13.1388874, -14.1364660, -13.1263504, -0.4365377, 0.4422572
1: -7.6657677, -6.8138075, -7.6770792, -6.7740922, -0.4482859, 0.4258854
2: 3.0116506, 3.9084082, 2.9591098, 3.9180164, -0.5977039, 0.6408348
3: 0.4971709, 1.2975218, 0.5129800, 1.3280401, -0.5348201, 0.4981296
4: -6.9603667, -6.0848308, -6.9694438, -6.0950508, -0.5788503, 0.6008065
5: -5.8427539, -4.9777393, -5.8605890, -4.9688411, -0.4866748, 0.4940784
6: -11.7012482, -10.5325184, -11.7254620, -10.5031042, -0.5321629, 0.5288737
7: -0.6840563, 0.0764937, -0.7176931, 0.0767601, -0.4358814, 0.4698780
8: -3.6528485, -2.8714333, -3.6638513, -2.8418028, -0.4730635, 0.4688897
9: -9.5282259, -8.4865980, -9.5381641, -8.4678049, -0.4748604, 0.4658996

Time for backsubstitution: 8.95 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462177
time: 3.61 seconds

## Relational analysis of NS_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
time: 3.24 seconds

## BFS NS instance: NS_A1_A2_B1

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1198187, -13.1345520, -0.4428639, 0.4269441
1: -7.6686587, -6.8011436, -7.6731710, -6.8137927, -0.4075772, 0.4313513
2: 2.9750814, 3.9252968, 3.0116315, 3.9289384, -0.6441054, 0.6048799
3: 0.4928286, 1.3196907, 0.4846127, 1.2980201, -0.4943326, 0.5407472
4: -6.9649348, -6.0918393, -6.9612961, -6.0836029, -0.6041005, 0.5752532
5: -5.8433108, -4.9774771, -5.8436408, -4.9777369, -0.4805984, 0.4825735
6: -11.7114992, -10.5123320, -11.7130661, -10.5325089, -0.5215847, 0.5333960
7: -0.7068467, 0.0818155, -0.6841369, 0.0897295, -0.4715247, 0.4405856
8: -3.6529422, -2.8693893, -3.6538787, -2.8710918, -0.4366956, 0.4377987
9: -9.5267200, -8.4853220, -9.5287066, -8.4864206, -0.4589653, 0.4614675

Time for backsubstitution: 8.23 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2224844, upper bound: 0.2417674
time: 3.19 seconds

## Relational analysis of NS_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2315047, upper bound: 0.2426862
time: 3.07 seconds

## BFS NS instance: NS_A1_A2_B2

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1278086, -13.1200895, -0.4531329, 0.4307939
1: -7.6686587, -6.8011436, -7.6808853, -6.7845693, -0.4500420, 0.4482942
2: 2.9750814, 3.9252968, 3.0014534, 3.9232492, -0.6344361, 0.6168637
3: 0.4928286, 1.3196907, 0.5046320, 1.2904034, -0.5086749, 0.5362091
4: -6.9649348, -6.0918393, -6.9641719, -6.0868149, -0.6007137, 0.5755112
5: -5.8433108, -4.9774771, -5.8634567, -4.9690819, -0.4841480, 0.4968562
6: -11.7114992, -10.5123320, -11.7297945, -10.5226183, -0.5247951, 0.5461860
7: -0.7068467, 0.0818155, -0.6912031, 0.0853629, -0.4707246, 0.4562664
8: -3.6529422, -2.8693893, -3.6646059, -2.8416920, -0.4899979, 0.4710796
9: -9.5267200, -8.4853220, -9.5401707, -8.4648829, -0.4793181, 0.4689041

Time for backsubstitution: 9.01 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2224844, upper bound: 0.2417675
time: 3.16 seconds

## Relational analysis of NS_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2315047, upper bound: 0.2426864
time: 3.11 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -14.1364698, -13.1227446, -14.1198187, -13.1345520, -0.4461851, 0.4377429
1: -7.6770892, -6.7727680, -7.6731710, -6.8137927, -0.4290346, 0.4664607
2: 2.9678011, 3.9180298, 3.0116315, 3.9289384, -0.6511869, 0.5980473
3: 0.5129824, 1.3119073, 0.4846127, 1.2980201, -0.4918754, 0.5505064
4: -6.9668255, -6.0950527, -6.9612961, -6.0836029, -0.6043179, 0.5717909
5: -5.8629947, -4.9688282, -5.8436408, -4.9777369, -0.4947701, 0.4863021
6: -11.7282677, -10.5030575, -11.7130661, -10.5325089, -0.5332966, 0.5386035
7: -0.7128882, 0.0767579, -0.6841369, 0.0897295, -0.4803240, 0.4431221
8: -3.6638453, -2.8401210, -3.6538787, -2.8710918, -0.4698997, 0.4910016
9: -9.5381775, -8.4638376, -9.5287066, -8.4864206, -0.4666505, 0.4813145

Time for backsubstitution: 8.91 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2235570, upper bound: 0.2311149
time: 3.11 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2358245, upper bound: 0.2315051
time: 3.20 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -14.1364698, -13.1227446, -14.1278086, -13.1200895, -0.4506726, 0.4354576
1: -7.6770892, -6.7727680, -7.6808853, -6.7845693, -0.4223194, 0.4467083
2: 2.9678011, 3.9180298, 3.0014534, 3.9232492, -0.6475205, 0.6097312
3: 0.5129824, 1.3119073, 0.5046320, 1.2904034, -0.4869452, 0.5323141
4: -6.9668255, -6.0950527, -6.9641719, -6.0868149, -0.6093154, 0.5799415
5: -5.8629947, -4.9688282, -5.8634567, -4.9690819, -0.4895041, 0.4915872
6: -11.7282677, -10.5030575, -11.7297945, -10.5226183, -0.5286274, 0.5414050
7: -0.7128882, 0.0767579, -0.6912031, 0.0853629, -0.4728746, 0.4433210
8: -3.6638453, -2.8401210, -3.6646059, -2.8416920, -0.4490952, 0.4502263
9: -9.5381775, -8.4638376, -9.5401707, -8.4648829, -0.4680824, 0.4705784

Time for backsubstitution: 9.01 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2235570, upper bound: 0.2311155
time: 3.15 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2358245, upper bound: 0.2315050
time: 3.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.05 seconds
NS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462177
NS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
NS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2224844, upper bound: 0.2417674
NS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2315047, upper bound: 0.2426862
NS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2224844, upper bound: 0.2417675
NS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2315047, upper bound: 0.2426864
NS_A2_A2_B1_B1, status: Status.VERIFIED, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2235570, upper bound: 0.2311149
NS_A2_A2_B1_B2, status: Status.VERIFIED, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2358245, upper bound: 0.2315051
NS_A2_A2_B2_B1, status: Status.VERIFIED, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2235570, upper bound: 0.2311155
NS_A2_A2_B2_B2, status: Status.VERIFIED, split count: 4, time: 16.05
Output dim: 3, lower bound: -0.2358245, upper bound: 0.2315050

## BFS NS instance: NS_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -14.1197929, -13.1388874, -14.1284847, -13.1371698, -0.4293382, 0.4369476
1: -7.6657677, -6.8138075, -7.6686587, -6.8011436, -0.4225504, 0.4106545
2: 3.0116506, 3.9084082, 2.9750814, 3.9252968, -0.6080360, 0.6265769
3: 0.4971709, 1.2975218, 0.4928286, 1.3196907, -0.5216732, 0.5041289
4: -6.9603667, -6.0848308, -6.9649348, -6.0918393, -0.5820789, 0.5960104
5: -5.8427539, -4.9777393, -5.8433108, -4.9774771, -0.4812067, 0.4808960
6: -11.7012482, -10.5325184, -11.7114992, -10.5123320, -0.5269806, 0.5172622
7: -0.6840563, 0.0764937, -0.7068467, 0.0818155, -0.4404378, 0.4593778
8: -3.6528485, -2.8714333, -3.6529422, -2.8693893, -0.4378221, 0.4358368
9: -9.5282259, -8.4865980, -9.5267200, -8.4853220, -0.4608784, 0.4578539

Time for backsubstitution: 8.95 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1_A1_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285048
time: 3.62 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426858
time: 3.21 seconds

## BFS NS instance: NS_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -14.1197929, -13.1388874, -14.1364698, -13.1227446, -0.4395046, 0.4402688
1: -7.6657677, -6.8138075, -7.6770892, -6.7727680, -0.4576597, 0.4246883
2: 3.0116506, 3.9084082, 2.9678011, 3.9180298, -0.5977159, 0.6336584
3: 0.4971709, 1.2975218, 0.5129824, 1.3119073, -0.5314326, 0.4981272
4: -6.9603667, -6.0848308, -6.9668255, -6.0950527, -0.5786209, 0.5962281
5: -5.8427539, -4.9777393, -5.8629947, -4.9688282, -0.4849353, 0.4951062
6: -11.7012482, -10.5325184, -11.7282677, -10.5030575, -0.5321882, 0.5306919
7: -0.6840563, 0.0764937, -0.7128882, 0.0767579, -0.4358797, 0.4681771
8: -3.6528485, -2.8714333, -3.6638453, -2.8401210, -0.4910250, 0.4688880
9: -9.5282259, -8.4865980, -9.5381775, -8.4638376, -0.4807255, 0.4653113

Time for backsubstitution: 8.87 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285054
time: 3.38 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426860
time: 3.27 seconds

## BFS NS instance: NS_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -14.1280546, -13.1373358, -14.1181803, -13.1331835, -0.4446013, 0.4257413
1: -7.6663818, -6.8011456, -7.6673417, -6.8097210, -0.4147649, 0.4290962
2: 2.9841933, 3.9250870, 3.0363197, 3.9377108, -0.6424961, 0.5758152
3: 0.4930515, 1.3085663, 0.4809053, 1.2585626, -0.4527640, 0.5283606
4: -6.9573641, -6.0918446, -6.9396625, -6.0700154, -0.5928543, 0.5431828
5: -5.8430037, -4.9851475, -5.8483715, -5.0062213, -0.4544575, 0.4786901
6: -11.7114983, -10.5222740, -11.7108183, -10.5667887, -0.4829376, 0.5157692
7: -0.6948671, 0.0817854, -0.6530972, 0.1091940, -0.4526842, 0.3925873
8: -3.6529064, -2.8726745, -3.6660337, -2.8814414, -0.4242871, 0.4417970
9: -9.5211267, -8.4854307, -9.5071640, -8.4742470, -0.4643750, 0.4403169

Time for backsubstitution: 8.87 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 2131

## Relational analysis of NS_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2285047, upper bound: 0.2271683
time: 3.21 seconds

## Relational analysis of NS_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_B1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2285049, upper bound: 0.2271677
time: 3.08 seconds

## BFS NS instance: NS_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1196108, -13.1346445, -0.4428368, 0.4267520
1: -7.6686587, -6.8011436, -7.6719475, -6.8137927, -0.4075764, 0.4361153
2: 2.9750814, 3.9252968, 3.0135307, 3.9288201, -0.6434650, 0.5890331
3: 0.4928286, 1.3196907, 0.4847379, 1.2961073, -0.4626358, 0.5407157
4: -6.9649348, -6.0918393, -6.9573207, -6.0836067, -0.6040959, 0.5470016
5: -5.8433108, -4.9774771, -5.8434730, -4.9796586, -0.4673274, 0.4815311
6: -11.7114992, -10.5123320, -11.7130632, -10.5339785, -0.4904909, 0.5331447
7: -0.7068467, 0.0818155, -0.6795580, 0.0897126, -0.4715061, 0.3990972
8: -3.6529422, -2.8693893, -3.6538572, -2.8736982, -0.4297514, 0.4377789
9: -9.5267200, -8.4853220, -9.5253649, -8.4864788, -0.4587348, 0.4580618

Time for backsubstitution: 8.92 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2131

## Relational analysis of NS_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2426859, upper bound: 0.2282591
time: 3.24 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2426861, upper bound: 0.2282592
time: 3.31 seconds

## BFS NS instance: NS_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -14.1280546, -13.1373358, -14.1261883, -13.1187210, -0.4542694, 0.4291972
1: -7.6663818, -6.8011456, -7.6761050, -6.7805610, -0.4538345, 0.4432604
2: 2.9841933, 3.9250870, 3.0229416, 3.9371977, -0.6398649, 0.5940595
3: 0.4930515, 1.3085663, 0.5013316, 1.2510614, -0.4788151, 0.5359530
4: -6.9573641, -6.0918446, -6.9452310, -6.0718627, -0.5919735, 0.5468104
5: -5.8430037, -4.9851475, -5.8676105, -4.9986629, -0.4585321, 0.4936664
6: -11.7114983, -10.5222740, -11.7287359, -10.5559196, -0.4902742, 0.5301757
7: -0.6948671, 0.0817854, -0.6611633, 0.1078551, -0.4674554, 0.4211662
8: -3.6529064, -2.8726745, -3.6750314, -2.8533001, -0.4784751, 0.4766302
9: -9.5211267, -8.4854307, -9.5186253, -8.4526091, -0.4841986, 0.4485871

Time for backsubstitution: 8.85 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2131

## Relational analysis of NS_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2224843, upper bound: 0.2271677
time: 3.26 seconds

## Relational analysis of NS_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2224845, upper bound: 0.2271677
time: 3.32 seconds

## BFS NS instance: NS_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1276016, -13.1201954, -0.4530847, 0.4305993
1: -7.6686587, -6.8011436, -7.6794896, -6.7845702, -0.4500415, 0.4503188
2: 2.9750814, 3.9252968, 3.0037069, 3.9231310, -0.6338935, 0.6077719
3: 0.4928286, 1.3196907, 0.5047703, 1.2882402, -0.4866614, 0.5361762
4: -6.9649348, -6.0918393, -6.9600105, -6.0868187, -0.6007102, 0.5500665
5: -5.8433108, -4.9774771, -5.8632617, -4.9710035, -0.4710131, 0.4958091
6: -11.7114992, -10.5123320, -11.7297955, -10.5242777, -0.4976482, 0.5459375
7: -0.7068467, 0.0818155, -0.6875083, 0.0853426, -0.4707053, 0.4280117
8: -3.6529422, -2.8693893, -3.6645863, -2.8444653, -0.4845457, 0.4710622
9: -9.5267200, -8.4853220, -9.5368299, -8.4649487, -0.4790950, 0.4659618

Time for backsubstitution: 8.93 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 2131

## Relational analysis of NS_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2315047, upper bound: 0.2282593
time: 3.35 seconds

## Relational analysis of NS_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_B2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2315049, upper bound: 0.2282593
time: 3.27 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 15.76 seconds
NS_A1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285048
NS_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426858
NS_A1_A1_B2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285054
NS_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426860
NS_A1_A2_B1_B1_B1, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2285047, upper bound: 0.2271683
NS_A1_A2_B1_B1_B2, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2285049, upper bound: 0.2271677
NS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2426859, upper bound: 0.2282591
NS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2426861, upper bound: 0.2282592
NS_A1_A2_B2_B1_B1, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2224843, upper bound: 0.2271677
NS_A1_A2_B2_B1_B2, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2224845, upper bound: 0.2271677
NS_A1_A2_B2_B2_B1, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2315047, upper bound: 0.2282593
NS_A1_A2_B2_B2_B2, status: Status.VERIFIED, split count: 5, time: 15.76
Output dim: 3, lower bound: -0.2315049, upper bound: 0.2282593

## BFS NS instance: NS_A1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -14.1195831, -13.1389828, -14.1284847, -13.1371698, -0.4290826, 0.4369211
1: -7.6645393, -6.8138089, -7.6686587, -6.8011436, -0.4268456, 0.4106541
2: 3.0135508, 3.9082971, 2.9750814, 3.9252968, -0.5846267, 0.6259274
3: 0.4972992, 1.2956083, 0.4928286, 1.3196907, -0.5216413, 0.4706805
4: -6.9564004, -6.0848365, -6.9649348, -6.0918393, -0.5489266, 0.5960064
5: -5.8425903, -4.9796600, -5.8433108, -4.9774771, -0.4801481, 0.4664040
6: -11.7012472, -10.5339890, -11.7114992, -10.5123320, -0.5267286, 0.4800807
7: -0.6794827, 0.0764730, -0.7068467, 0.0818155, -0.3875411, 0.4593549
8: -3.6528277, -2.8740451, -3.6529422, -2.8693893, -0.4378016, 0.4281192
9: -9.5248880, -8.4866562, -9.5267200, -8.4853220, -0.4563792, 0.4576198

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2417673
time: 3.18 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2426868
time: 3.18 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -14.1195831, -13.1389828, -14.1364698, -13.1227446, -0.4391820, 0.4402423
1: -7.6645393, -6.8138089, -7.6770892, -6.7727680, -0.4596322, 0.4246879
2: 3.0135508, 3.9082971, 2.9678011, 3.9180298, -0.5805697, 0.6330085
3: 0.4972992, 1.2956083, 0.5129824, 1.3119073, -0.5314007, 0.4750485
4: -6.9564004, -6.0848365, -6.9668255, -6.0950527, -0.5474060, 0.5962238
5: -5.8425903, -4.9796600, -5.8629947, -4.9688282, -0.4838767, 0.4812617
6: -11.7012472, -10.5339890, -11.7282677, -10.5030575, -0.5319362, 0.4950304
7: -0.6794827, 0.0764730, -0.7128882, 0.0767579, -0.3961229, 0.4681542
8: -3.6528277, -2.8740451, -3.6638453, -2.8401210, -0.4910045, 0.4649007
9: -9.5248880, -8.4866562, -9.5381775, -8.4638376, -0.4745426, 0.4650772

Time for backsubstitution: 8.90 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2534
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2417674
time: 3.23 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2426861
time: 3.96 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1195831, -13.1389828, -0.4369211, 0.4290825
1: -7.6686587, -6.8011436, -7.6645393, -6.8138089, -0.4106541, 0.4268457
2: 2.9750814, 3.9252968, 3.0135508, 3.9082971, -0.6259274, 0.5846267
3: 0.4928286, 1.3196907, 0.4972992, 1.2956083, -0.4706805, 0.5216413
4: -6.9649348, -6.0918393, -6.9564004, -6.0848365, -0.5960064, 0.5489264
5: -5.8433108, -4.9774771, -5.8425903, -4.9796600, -0.4664040, 0.4801481
6: -11.7114992, -10.5123320, -11.7012472, -10.5339890, -0.4800806, 0.5267286
7: -0.7068467, 0.0818155, -0.6794827, 0.0764730, -0.4593549, 0.3875409
8: -3.6529422, -2.8693893, -3.6528277, -2.8740451, -0.4281192, 0.4378016
9: -9.5267200, -8.4853220, -9.5248880, -8.4866562, -0.4576197, 0.4563792

Time for backsubstitution: 8.85 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1_A2_B1_B2_B1_A1

### Relational analysis result of NS_A1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2285047, upper bound: 0.2209559
time: 3.12 seconds

## Relational analysis of NS_A1_A2_B1_B2_B1_A2

### Relational analysis result of NS_A1_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2285047, upper bound: 0.2282589
time: 3.71 seconds

## BFS NS instance: NS_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1282749, -13.1372623, -0.4278984, 0.4277328
1: -7.6686587, -6.8011436, -7.6674328, -6.8011422, -0.4071679, 0.4107552
2: 2.9750814, 3.9252968, 2.9769773, 3.9251852, -0.6049523, 0.5896599
3: 0.4928286, 1.3196907, 0.4929504, 1.3177781, -0.4669967, 0.4988832
4: -6.9649348, -6.0918393, -6.9619350, -6.0918431, -0.5754139, 0.5471232
5: -5.8433108, -4.9774771, -5.8431463, -4.9794011, -0.4674077, 0.4799047
6: -11.7114992, -10.5123320, -11.7114954, -10.5131693, -0.4905982, 0.5212853
7: -0.7068467, 0.0818155, -0.7045352, 0.0817976, -0.4424229, 0.4000334
8: -3.6529422, -2.8693893, -3.6529229, -2.8719113, -0.4288130, 0.4368274
9: -9.5267200, -8.4853220, -9.5233822, -8.4853792, -0.4587982, 0.4527123

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1_A2_B1_B2_B2_A1

### Relational analysis result of NS_A1_A2_B1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2285049, upper bound: 0.2209551
time: 3.69 seconds

## Relational analysis of NS_A1_A2_B1_B2_B2_A2

### Relational analysis result of NS_A1_A2_B1_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2285049, upper bound: 0.2282590
time: 3.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 16.20 seconds
NS_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2417673
NS_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2426868
NS_A1_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2417674
NS_A1_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2426861
NS_A1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2285047, upper bound: 0.2209559
NS_A1_A2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2285047, upper bound: 0.2282589
NS_A1_A2_B1_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2285049, upper bound: 0.2209551
NS_A1_A2_B1_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 16.20
Output dim: 3, lower bound: -0.2285049, upper bound: 0.2282590

## BFS NS instance: NS_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -14.1195831, -13.1389828, -14.1268463, -13.1357937, -0.4310179, 0.4354674
1: -7.6645393, -6.8138089, -7.6631017, -6.7964735, -0.4300938, 0.4086807
2: 3.0135508, 3.9082971, 3.0056868, 3.9336753, -0.6135116, 0.5965023
3: 0.4972992, 1.2956083, 0.4894519, 1.2794170, -0.4780951, 0.5027158
4: -6.9564004, -6.0848365, -6.9398422, -6.0807381, -0.5792108, 0.5607502
5: -5.8425903, -4.9796600, -5.8479571, -5.0060048, -0.4534361, 0.4855654
6: -11.7012472, -10.5339890, -11.7074184, -10.5502958, -0.4862375, 0.5104805
7: -0.6794827, 0.0764730, -0.6710916, 0.0988388, -0.4369035, 0.4048007
8: -3.6528277, -2.8740451, -3.6649773, -2.8798482, -0.4237969, 0.4427593
9: -9.5248880, -8.4866562, -9.5051813, -8.4731607, -0.4755201, 0.4373726

Time for backsubstitution: 8.95 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 66

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2374311
time: 3.55 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2191168, upper bound: 0.2399721
time: 3.27 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -14.1195831, -13.1389828, -14.1282749, -13.1372623, -0.4290562, 0.4367278
1: -7.6645393, -6.8138089, -7.6674328, -6.8011422, -0.4268448, 0.4152005
2: 3.0135508, 3.9082971, 2.9769773, 3.9251852, -0.5842052, 0.6038814
3: 0.4972992, 1.2956083, 0.4929504, 1.3177781, -0.4881938, 0.4706540
4: -6.9564004, -6.0848365, -6.9619350, -6.0918431, -0.5489223, 0.5638278
5: -5.8425903, -4.9796600, -5.8431463, -4.9794011, -0.4663370, 0.4659169
6: -11.7012472, -10.5339890, -11.7114954, -10.5131693, -0.4903104, 0.4798707
7: -0.6794827, 0.0764730, -0.7045352, 0.0817976, -0.3875258, 0.4093504
8: -3.6528277, -2.8740451, -3.6529229, -2.8719113, -0.4305174, 0.4280980
9: -9.5248880, -8.4866562, -9.5233822, -8.4853792, -0.4562819, 0.4533147

Time for backsubstitution: 8.31 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 66

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2169580, upper bound: 0.2399731
time: 3.54 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2408506
time: 4.12 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -14.1195831, -13.1389828, -14.1348419, -13.1213856, -0.4405806, 0.4385970
1: -7.6645393, -6.8138089, -7.6725454, -6.7669835, -0.4625599, 0.4205914
2: 3.0135508, 3.9082971, 2.9958811, 3.9299984, -0.6097503, 0.6078806
3: 0.4972992, 1.2956083, 0.5100150, 1.2712705, -0.4967260, 0.5081046
4: -6.9564004, -6.0848365, -6.9434876, -6.0826416, -0.5779564, 0.5637264
5: -5.8425903, -4.9796600, -5.8670506, -4.9984107, -0.4575379, 0.5006554
6: -11.7012472, -10.5339890, -11.7254219, -10.5401173, -0.4946916, 0.5255301
7: -0.6794827, 0.0764730, -0.6805267, 0.0974998, -0.4457533, 0.4252243
8: -3.6528277, -2.8740451, -3.6739066, -2.8518271, -0.4794636, 0.4781804
9: -9.5248880, -8.4866562, -9.5168867, -8.4507647, -0.4939544, 0.4450543

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 66
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 66

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2374309
time: 3.45 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2399722
time: 3.35 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -14.1195831, -13.1389828, -14.1362553, -13.1228466, -0.4391377, 0.4400473
1: -7.6645393, -6.8138089, -7.6756415, -6.7727690, -0.4596312, 0.4270372
2: 3.0135508, 3.9082971, 2.9700522, 3.9179144, -0.5801611, 0.6154552
3: 0.4972992, 1.2956083, 0.5131114, 1.3097668, -0.5059187, 0.4750195
4: -6.9564004, -6.0848365, -6.9637356, -6.0950556, -0.5474017, 0.5661817
5: -5.8425903, -4.9796600, -5.8628097, -4.9707499, -0.4703078, 0.4805949
6: -11.7012472, -10.5339890, -11.7282639, -10.5038967, -0.4988537, 0.4948039
7: -0.6794827, 0.0764730, -0.7102232, 0.0767384, -0.3961084, 0.4297204
8: -3.6528277, -2.8740451, -3.6638265, -2.8424692, -0.4865665, 0.4648831
9: -9.5248880, -8.4866562, -9.5348396, -8.4639006, -0.4743764, 0.4606552

Time for backsubstitution: 8.99 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 66
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 66

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2128879, upper bound: 0.2399724
time: 4.13 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2408507
time: 3.64 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 16.96 seconds
NS_A1_A1_B2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2374311
NS_A1_A1_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2191168, upper bound: 0.2399721
NS_A1_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2169580, upper bound: 0.2399731
NS_A1_A1_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2408506
NS_A1_A1_B2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2374309
NS_A1_A1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2399722
NS_A1_A1_B2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2128879, upper bound: 0.2399724
NS_A1_A1_B2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 16.96
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2408507

## BFS NS instance: NS_A1_A1_B2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -14.1190739, -13.1389914, -14.1267576, -13.1357956, -0.4285514, 0.4354198
1: -7.6642680, -6.8138084, -7.6630239, -6.7964730, -0.4298745, 0.4084724
2: 3.0135541, 3.9015384, 3.0056877, 3.9318895, -0.6113410, 0.5926700
3: 0.4973059, 1.2925997, 0.4894524, 1.2786145, -0.4773049, 0.4877608
4: -6.9498410, -6.0848370, -6.9386892, -6.0807395, -0.5652819, 0.5587428
5: -5.8396301, -4.9796648, -5.8471069, -5.0060053, -0.4394646, 0.4840481
6: -11.6987152, -10.5340042, -11.7066212, -10.5503006, -0.4742777, 0.5098672
7: -0.6793671, 0.0746784, -0.6710587, 0.0983722, -0.4358883, 0.4005523
8: -3.6514878, -2.8741782, -3.6646276, -2.8798943, -0.4201896, 0.4426012
9: -9.5248804, -8.4868488, -9.5051794, -8.4732132, -0.4753648, 0.4374745

Time for backsubstitution: 8.97 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 3, pos: 1754

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2190225, upper bound: 0.2288340
time: 3.40 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2155583, upper bound: 0.2364138
time: 3.22 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -14.1191788, -13.1389847, -14.1268101, -13.1378975, -0.4272897, 0.4349425
1: -7.6642718, -6.8138080, -7.6666555, -6.8010716, -0.4265300, 0.4141365
2: 3.0135508, 3.9056864, 2.9744010, 3.9171386, -0.5783310, 0.6038761
3: 0.4972999, 1.2928398, 0.4954605, 1.3092098, -0.4782567, 0.4601355
4: -6.9538746, -6.0848351, -6.9539390, -6.0915346, -0.5413139, 0.5518477
5: -5.8395329, -4.9798055, -5.8354082, -4.9808474, -0.4578259, 0.4543881
6: -11.6989918, -10.5345583, -11.7044353, -10.5166206, -0.4821419, 0.4719775
7: -0.6763406, 0.0754445, -0.6954112, 0.0788803, -0.3808861, 0.3972745
8: -3.6524401, -2.8746662, -3.6517000, -2.8729784, -0.4264166, 0.4251912
9: -9.5247421, -8.4867592, -9.5229416, -8.4856796, -0.4555609, 0.4525648

Time for backsubstitution: 9.04 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1754

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2074041, upper bound: 0.2407560
time: 3.26 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2151438, upper bound: 0.2372925
time: 3.33 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -14.1194487, -13.1389828, -14.1279392, -13.1372671, -0.4289618, 0.4345670
1: -7.6644664, -6.8138089, -7.6671505, -6.8011427, -0.4266827, 0.4148526
2: 3.0135508, 3.9065304, 2.9769812, 3.9183950, -0.5810146, 0.6019726
3: 0.4973006, 1.2948213, 0.4929557, 1.3147540, -0.4743915, 0.4695599
4: -6.9546852, -6.0848351, -6.9577780, -6.0918436, -0.5466297, 0.5495141
5: -5.8417501, -4.9796629, -5.8401470, -4.9794040, -0.4648283, 0.4519632
6: -11.7004490, -10.5339909, -11.7084341, -10.5131912, -0.4899886, 0.4665138
7: -0.6794527, 0.0760074, -0.7044163, 0.0800028, -0.3824570, 0.4087961
8: -3.6524799, -2.8740828, -3.6515813, -2.8720536, -0.4303410, 0.4245236
9: -9.5248852, -8.4867058, -9.5233765, -8.4855766, -0.4564394, 0.4531059

Time for backsubstitution: 8.98 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1754

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2095363, upper bound: 0.2407559
time: 3.25 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2172763, upper bound: 0.2372925
time: 3.22 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -14.1190739, -13.1389914, -14.1347609, -13.1213865, -0.4381142, 0.4385493
1: -7.6642680, -6.8138084, -7.6724634, -6.7669835, -0.4622667, 0.4203911
2: 3.0135541, 3.9015384, 2.9958811, 3.9282169, -0.6075673, 0.6022491
3: 0.4973059, 1.2925997, 0.5100169, 1.2704711, -0.4959378, 0.4931121
4: -6.9498410, -6.0848370, -6.9422736, -6.0826411, -0.5640996, 0.5617237
5: -5.8396301, -4.9796648, -5.8661895, -4.9984112, -0.4442530, 0.4990911
6: -11.6987152, -10.5340042, -11.7246227, -10.5401249, -0.4844534, 0.5248885
7: -0.6793671, 0.0746784, -0.6804934, 0.0970321, -0.4448998, 0.4200950
8: -3.6514878, -2.8741782, -3.6735554, -2.8518660, -0.4757533, 0.4780219
9: -9.5248804, -8.4868488, -9.5168829, -8.4508171, -0.4937985, 0.4451553

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 3, pos: 1754

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2149492, upper bound: 0.2288349
time: 3.73 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2114830, upper bound: 0.2364145
time: 3.26 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -14.1191788, -13.1389847, -14.1347656, -13.1235256, -0.4373710, 0.4382861
1: -7.6642718, -6.8138080, -7.6748734, -6.7726951, -0.4592590, 0.4260762
2: 3.0135508, 3.9056864, 2.9674654, 3.9098759, -0.5725946, 0.6137938
3: 0.4972999, 1.2928398, 0.5155466, 1.3011813, -0.4959435, 0.4644654
4: -6.9538746, -6.0848351, -6.9560003, -6.0943875, -0.5398660, 0.5541928
5: -5.8395329, -4.9798055, -5.8559332, -4.9713702, -0.4624953, 0.4695365
6: -11.6989918, -10.5345583, -11.7212009, -10.5072784, -0.4927273, 0.4882972
7: -0.6763406, 0.0754445, -0.7010128, 0.0738311, -0.3892748, 0.4169364
8: -3.6524401, -2.8746662, -3.6626031, -2.8435464, -0.4822688, 0.4616690
9: -9.5247421, -8.4867592, -9.5344105, -8.4641991, -0.4736555, 0.4600433

Time for backsubstitution: 9.01 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 3, pos: 1754

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B1_A1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2029674, upper bound: 0.2407560
time: 3.40 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B1_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2107453, upper bound: 0.2372925
time: 3.28 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -14.1194487, -13.1389828, -14.1359406, -13.1228542, -0.4390419, 0.4378855
1: -7.6644664, -6.8138089, -7.6753554, -6.7727690, -0.4594574, 0.4266510
2: 3.0135508, 3.9065304, 2.9700584, 3.9111242, -0.5749183, 0.6135464
3: 0.4973006, 1.2948213, 0.5131178, 1.3067503, -0.4921205, 0.4739246
4: -6.9546852, -6.0848351, -6.9593763, -6.0950565, -0.5451097, 0.5518696
5: -5.8417501, -4.9796629, -5.8598051, -4.9707537, -0.4688792, 0.4671476
6: -11.7004490, -10.5339909, -11.7251997, -10.5039158, -0.4985216, 0.4828588
7: -0.6794527, 0.0760074, -0.7101014, 0.0749428, -0.3908451, 0.4290154
8: -3.6524799, -2.8740828, -3.6624851, -2.8425937, -0.4864163, 0.4611671
9: -9.5248852, -8.4867058, -9.5348358, -8.4640942, -0.4745462, 0.4604452

Time for backsubstitution: 8.99 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1754
type: B, layer: 3, pos: 1754
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 1754

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B2_A1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2050993, upper bound: 0.2407561
time: 3.54 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B2_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2128773, upper bound: 0.2372925
time: 3.20 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 15.94 seconds
NS_A1_A1_B2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2190225, upper bound: 0.2288340
NS_A1_A1_B2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2155583, upper bound: 0.2364138
NS_A1_A1_B2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2074041, upper bound: 0.2407560
NS_A1_A1_B2_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2151438, upper bound: 0.2372925
NS_A1_A1_B2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2095363, upper bound: 0.2407559
NS_A1_A1_B2_B1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2172763, upper bound: 0.2372925
NS_A1_A1_B2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2149492, upper bound: 0.2288349
NS_A1_A1_B2_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2114830, upper bound: 0.2364145
NS_A1_A1_B2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2029674, upper bound: 0.2407560
NS_A1_A1_B2_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2107453, upper bound: 0.2372925
NS_A1_A1_B2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2050993, upper bound: 0.2407561
NS_A1_A1_B2_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 15.94
Output dim: 3, lower bound: -0.2128773, upper bound: 0.2372925

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -14.1175938, -13.1389875, -14.1262941, -13.1379004, -0.4234860, 0.4344951
1: -7.6574879, -6.8138084, -7.6644421, -6.8010702, -0.4191542, 0.4115257
2: 3.0225329, 3.9049072, 2.9772916, 3.9167719, -0.5705800, 0.6354785
3: 0.4973023, 1.2921462, 0.4954596, 1.3089080, -0.4742227, 0.3798087
4: -6.9526100, -6.0848355, -6.9533882, -6.0915341, -0.5680351, 0.5478578
5: -5.8386755, -4.9832592, -5.8350163, -4.9819560, -0.4573877, 0.4508867
6: -11.6972561, -10.5345592, -11.7038651, -10.5166197, -0.4777744, 0.4709666
7: -0.6741357, 0.0754433, -0.6946952, 0.0788791, -0.3762481, 0.3957356
8: -3.6524391, -2.8762207, -3.6516995, -2.8735363, -0.4239607, 0.4168189
9: -9.5247431, -8.4909763, -9.5229416, -8.4870672, -0.4542522, 0.4488233

Time for backsubstitution: 8.97 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 768

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2047143, upper bound: 0.2378237
time: 3.50 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2047143, upper bound: 0.2382762
time: 3.85 seconds

## BFS NS instance: NS_A1_A1_B2_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -14.1178598, -13.1389866, -14.1274261, -13.1372671, -0.4251533, 0.4341196
1: -7.6576734, -6.8138075, -7.6649356, -6.8011417, -0.4192840, 0.4122423
2: 3.0225348, 3.9057088, 2.9798727, 3.9180264, -0.5732679, 0.6338544
3: 0.4973013, 1.2940946, 0.4929571, 1.3144495, -0.4703639, 0.3907073
4: -6.9533610, -6.0848355, -6.9572372, -6.0918446, -0.5726600, 0.5455227
5: -5.8408284, -4.9831133, -5.8397522, -4.9805126, -0.4631391, 0.4484608
6: -11.6987123, -10.5339928, -11.7078609, -10.5131912, -0.4857221, 0.4655036
7: -0.6772447, 0.0760069, -0.7036984, 0.0800018, -0.3774695, 0.4071691
8: -3.6524796, -2.8756618, -3.6515803, -2.8726099, -0.4278841, 0.4154203
9: -9.5248852, -8.4909363, -9.5233755, -8.4869661, -0.4551295, 0.4493676

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 922
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 768

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2068468, upper bound: 0.2378241
time: 3.46 seconds

## Relational analysis of NS_A1_A1_B2_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A1_A1_B2_B1_A2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2068468, upper bound: 0.2382756
time: 3.34 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -14.1175938, -13.1389875, -14.1342506, -13.1235256, -0.4339442, 0.4378359
1: -7.6574879, -6.8138084, -7.6726565, -6.7726955, -0.4518766, 0.4234643
2: 3.0225329, 3.9049072, 2.9703565, 3.9095116, -0.5648265, 0.6501622
3: 0.4973023, 1.2921462, 0.5155466, 1.3008859, -0.4919159, 0.3852739
4: -6.9526100, -6.0848355, -6.9554486, -6.0943890, -0.5643811, 0.5502138
5: -5.8386755, -4.9832592, -5.8555517, -4.9724784, -0.4588106, 0.4660652
6: -11.6972561, -10.5345592, -11.7206345, -10.5072803, -0.4881330, 0.4872870
7: -0.6741357, 0.0754433, -0.7002950, 0.0738320, -0.3848720, 0.4153912
8: -3.6524391, -2.8762207, -3.6626017, -2.8440831, -0.4798493, 0.4441566
9: -9.5247431, -8.4909763, -9.5344114, -8.4655838, -0.4723473, 0.4567213

Time for backsubstitution: 8.94 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 768
type: B, layer: 3, pos: 768
type: A, layer: 3, pos: 327
type: B, layer: 3, pos: 327
type: A, layer: 3, pos: 1485
type: B, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: B, layer: 3, pos: 1706
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 221
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 899
type: A, layer: 3, pos: 899
type: B, layer: 3, pos: 3118
type: A, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: A, layer: 3, pos: 222
type: B, layer: 3, pos: 1438
type: A, layer: 3, pos: 963
type: B, layer: 3, pos: 963
type: A, layer: 3, pos: 1438
type: B, layer: 3, pos: 183
type: A, layer: 3, pos: 183
type: B, layer: 3, pos: 922
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 181
type: A, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: A, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 3, pos: 768

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B1_A1_A1

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007772, upper bound: 0.2378241
time: 3.36 seconds

## Relational analysis of NS_A1_A1_B2_B2_A2_B2_B1_A1_A2

### Relational analysis result of NS_A1_A1_B2_B2_A2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2007772, upper bound: 0.2382756
time: 3.42 seconds

## BFS NS instance: NS_A1_A1_B2_B2_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -14.1178598, -13.1389866, -14.1354237, -13.1228542, -0.4356105, 0.4374356
1: -7.6576734, -6.8138075, -7.6731396, -6.7727685, -0.4520085, 0.4240403
2: 3.0225348, 3.9057088, 2.9729462, 3.9107599, -0.5671554, 0.6509314
3: 0.4973013, 1.2940946, 0.5131195, 1.3064532, -0.4880986, 0.3961880
4: -6.9533610, -6.0848355, -6.9588323, -6.0950565, -0.5690081, 0.5478897
5: -5.8408284, -4.9831133, -5.8594227, -4.9718642, -0.4640372, 0.4636760
6: -11.6987123, -10.5339928, -11.7246323, -10.5039158, -0.4939103, 0.4818494
7: -0.6772447, 0.0760069, -0.7093849, 0.0749416, -0.3860934, 0.4273906
8: -3.6524796, -2.8756618, -3.6624851, -2.8431315, -0.4840002, 0.4427605
9: -9.5248852, -8.4909363, -9.5348358, -8.4654779, -0.4732363, 0.4571193

Time for backsubstitution: 9.02 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.14 + 548.37 = 603.51 seconds
