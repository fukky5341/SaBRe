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
execution time: IAR + RelationalAnalysis = 22.63 + 32.66 = 55.30 seconds
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
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.36 seconds

### Candidate
type: A, layer: 3, pos: 1103

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2431952, upper bound: 0.2529797
time: 3.08 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2463280, upper bound: 0.2463279
time: 2.89 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.35
Output dim: 3, lower bound: -0.2431952, upper bound: 0.2529797
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.35
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

Time for backsubstitution: 8.17 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 2131

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2362360, upper bound: 0.2331089
time: 3.30 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2362360, upper bound: 0.2462169
time: 3.79 seconds

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

Time for backsubstitution: 8.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.42 seconds

### Candidate
type: B, layer: 3, pos: 2131

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2399677, upper bound: 0.2295304
time: 3.10 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2399677, upper bound: 0.2399675
time: 2.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.32 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 15.32
Output dim: 3, lower bound: -0.2362360, upper bound: 0.2331089
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.32
Output dim: 3, lower bound: -0.2362360, upper bound: 0.2462169
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 15.32
Output dim: 3, lower bound: -0.2399677, upper bound: 0.2295304
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 15.32
Output dim: 3, lower bound: -0.2399677, upper bound: 0.2399675

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -14.1198187, -13.1345520, -14.1364660, -13.1263504, -0.4340878, 0.4481735
1: -7.6731710, -6.8137927, -7.6770792, -6.7740922, -0.4570869, 0.4302387
2: 3.0116315, 3.9289384, 2.9591098, 3.9180164, -0.5980358, 0.6583629
3: 0.4846127, 1.2980201, 0.5129800, 1.3280401, -0.5538936, 0.4918771
4: -6.9612961, -6.0836029, -6.9694438, -6.0950508, -0.5720510, 0.6088963
5: -5.8436408, -4.9777369, -5.8605890, -4.9688411, -0.4880419, 0.4941077
6: -11.7130661, -10.5325089, -11.7254620, -10.5031042, -0.5385780, 0.5321658
7: -0.6841369, 0.0897295, -0.7176931, 0.0767601, -0.4431236, 0.4820249
8: -3.6538787, -2.8710918, -3.6638513, -2.8418028, -0.4730401, 0.4699016
9: -9.5287066, -8.4864206, -9.5381641, -8.4678049, -0.4754493, 0.4671911

Time for backsubstitution: 8.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 2131

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
time: 3.36 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462177
time: 3.38 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -14.1278133, -13.1192226, -14.1277866, -13.1231308, -0.4409256, 0.4399613
1: -7.6816931, -6.7845712, -7.6749859, -6.7746558, -0.4756358, 0.4208674
2: 3.0014448, 3.9222832, 2.9897232, 3.9050493, -0.6003075, 0.6242828
3: 0.5033000, 1.2904766, 0.5168443, 1.3054430, -0.5211294, 0.4808588
4: -6.9643092, -6.0837197, -6.9665198, -6.0874825, -0.5864029, 0.5895100
5: -5.8639050, -4.9690838, -5.8671713, -4.9690704, -0.4912741, 0.5021324
6: -11.7282581, -10.5226154, -11.7214270, -10.5225124, -0.5309489, 0.5338531
7: -0.6912270, 0.0870531, -0.6988482, 0.0736785, -0.4371703, 0.4689829
8: -3.6647367, -2.8416405, -3.6636276, -2.8325834, -0.5256042, 0.4502103
9: -9.5409117, -8.4648533, -9.5397034, -8.4615860, -0.4898717, 0.4685206

Time for backsubstitution: 8.89 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.34 seconds

### Candidate
type: A, layer: 3, pos: 2131

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2295304
time: 3.07 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2295301
time: 3.70 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -14.1278086, -13.1200895, -14.1364746, -13.1214905, -0.4426837, 0.4513878
1: -7.6808853, -6.7845693, -7.6771173, -6.7625871, -0.4862769, 0.4227911
2: 3.0014534, 3.9232492, 2.9553170, 3.9183941, -0.6100535, 0.6529279
3: 0.5046320, 1.2904034, 0.5126226, 1.3275278, -0.5537271, 0.4870183
4: -6.9641719, -6.0868149, -6.9704518, -6.0950503, -0.5808477, 0.6058278
5: -5.8634567, -4.9690819, -5.8675814, -4.9688134, -0.4925747, 0.5028267
6: -11.7297945, -10.5226183, -11.7313147, -10.5029125, -0.5414972, 0.5398102
7: -0.6912031, 0.0853629, -0.7205627, 0.0768306, -0.4434018, 0.4854429
8: -3.6646059, -2.8416920, -3.6639450, -2.8307428, -0.5251448, 0.4491639
9: -9.5401707, -8.4648829, -9.5381937, -8.4603615, -0.4903975, 0.4684342

Time for backsubstitution: 9.16 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 2131

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2399676
time: 2.97 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2399679
time: 3.23 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.82 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.82
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462170
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.82
Output dim: 3, lower bound: -0.2284501, upper bound: 0.2462177
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 15.82
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2295304
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 15.82
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2295301
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 15.82
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2399676
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 15.82
Output dim: 3, lower bound: -0.2295306, upper bound: 0.2399679

## BFS NS instance: NS_A1_B2_A1

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

Time for backsubstitution: 9.38 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.43 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2462168
time: 3.40 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2462168
time: 3.35 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -14.1284847, -13.1371698, -14.1364660, -13.1263504, -0.4350686, 0.4339674
1: -7.6686587, -6.8011436, -7.6770792, -6.7740922, -0.4341686, 0.4298303
2: 2.9750814, 3.9252968, 2.9591098, 3.9180164, -0.5987630, 0.6200652
3: 0.4928286, 1.3196907, 0.5129800, 1.3280401, -0.5128639, 0.4964576
4: -6.9649348, -6.0918393, -6.9694438, -6.0950508, -0.5722156, 0.5801260
5: -5.8433108, -4.9774771, -5.8605890, -4.9688411, -0.4863017, 0.4944813
6: -11.7114992, -10.5123320, -11.7254620, -10.5031042, -0.5246234, 0.5321178
7: -0.7068467, 0.0818155, -0.7176931, 0.0767601, -0.4449821, 0.4555819
8: -3.6529422, -2.8693893, -3.6638513, -2.8418028, -0.4721131, 0.4700532
9: -9.5267200, -8.4853220, -9.5381641, -8.4678049, -0.4733195, 0.4672596

Time for backsubstitution: 9.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2331088
time: 3.57 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2331096
time: 3.30 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -14.1277828, -13.1243706, -14.1364746, -13.1214905, -0.4447267, 0.4456339
1: -7.6749611, -6.7845874, -7.6771173, -6.7625871, -0.4817998, 0.4263135
2: 3.0014744, 3.9047198, 2.9553170, 3.9183941, -0.6115313, 0.6386867
3: 0.5171700, 1.2899661, 0.5126226, 1.3275278, -0.5376177, 0.4961212
4: -6.9632931, -6.0874853, -6.9704518, -6.0950503, -0.5885401, 0.5976613
5: -5.8626537, -4.9690847, -5.8675814, -4.9688134, -0.4912617, 0.5029080
6: -11.7192116, -10.5226269, -11.7313147, -10.5029125, -0.5350101, 0.5373080
7: -0.6911311, 0.0736084, -0.7205627, 0.0768306, -0.4420164, 0.4769239
8: -3.6635315, -2.8419929, -3.6639450, -2.8307428, -0.5251915, 0.4482667
9: -9.5396852, -8.4650412, -9.5381937, -8.4603615, -0.4900539, 0.4673344

Time for backsubstitution: 9.01 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.34 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362360
time: 3.38 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362365
time: 3.18 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.1364698, -13.1227446, -14.1364746, -13.1214905, -0.4437735, 0.4369528
1: -7.6770892, -6.7727680, -7.6771173, -6.7625871, -0.4716759, 0.4223828
2: 2.9678011, 3.9180298, 2.9553170, 3.9183941, -0.6107802, 0.6184139
3: 0.5129824, 1.3119073, 0.5126226, 1.3275278, -0.5152020, 0.4915807
4: -6.9668255, -6.0950527, -6.9704518, -6.0950503, -0.5807719, 0.5769188
5: -5.8629947, -4.9688282, -5.8675814, -4.9688134, -0.4908857, 0.5032403
6: -11.7282677, -10.5030575, -11.7313147, -10.5029125, -0.5286424, 0.5396626
7: -0.7128882, 0.0767579, -0.7205627, 0.0768306, -0.4451654, 0.4646113
8: -3.6638453, -2.8401210, -3.6639450, -2.8307428, -0.5241406, 0.4492540
9: -9.5381775, -8.4638376, -9.5381937, -8.4603615, -0.4890361, 0.4684645

Time for backsubstitution: 9.30 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1103
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.44 seconds

### Candidate
type: B, layer: 3, pos: 1103

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2284499
time: 3.74 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2292391
time: 3.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.72 seconds
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2462168
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2462168
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2331088
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2284500, upper bound: 0.2331096
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362360
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2362365
NS_A2_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2284499
NS_A2_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 16.72
Output dim: 3, lower bound: -0.2288086, upper bound: 0.2292391

## BFS NS instance: NS_A1_B2_A1_B1

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

Time for backsubstitution: 9.38 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285048
time: 3.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426858
time: 3.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2

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

Time for backsubstitution: 9.32 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 2818

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285054
time: 3.54 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426860
time: 3.26 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 16.57 seconds
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 16.57
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285048
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.57
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426858
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 16.57
Output dim: 3, lower bound: -0.2211451, upper bound: 0.2285054
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.57
Output dim: 3, lower bound: -0.2220269, upper bound: 0.2426860

## BFS NS instance: NS_A1_B2_A1_B1_A2

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

Time for backsubstitution: 9.35 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.54 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2417673
time: 3.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2426868
time: 3.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

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

Time for backsubstitution: 9.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 2818

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2417674
time: 3.35 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2426861
time: 4.10 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 16.94 seconds
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.94
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2417673
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.94
Output dim: 3, lower bound: -0.2209552, upper bound: 0.2426868
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.94
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2417674
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.94
Output dim: 3, lower bound: -0.2168811, upper bound: 0.2426861

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

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

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 66

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2374311
time: 3.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2191168, upper bound: 0.2399721
time: 3.37 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

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

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.38 seconds

### Candidate
type: A, layer: 3, pos: 66

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2383269
time: 3.43 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2399732
time: 3.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

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

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 66

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2374309
time: 3.45 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2399722
time: 3.41 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

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

Time for backsubstitution: 9.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 66
type: A, layer: 3, pos: 327
type: A, layer: 3, pos: 1754
type: A, layer: 3, pos: 1485
type: A, layer: 3, pos: 1706
type: A, layer: 3, pos: 768
type: A, layer: 3, pos: 221
type: A, layer: 3, pos: 3118
type: A, layer: 3, pos: 222
type: A, layer: 3, pos: 963
type: A, layer: 3, pos: 922
type: A, layer: 3, pos: 3124
type: A, layer: 3, pos: 183
type: A, layer: 3, pos: 899
type: A, layer: 3, pos: 1438
type: A, layer: 3, pos: 181
type: A, layer: 3, pos: 2899
type: A, layer: 3, pos: 1511
type: A, layer: 3, pos: 2534

Time for candidate selection: 0.37 seconds

### Candidate
type: A, layer: 3, pos: 66

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2383263
time: 3.74 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2408513
time: 3.23 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 16.38 seconds
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2374311
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2191168, upper bound: 0.2399721
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2383269
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2191167, upper bound: 0.2399732
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2374309
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2399722
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2383263
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 16.38
Output dim: 3, lower bound: -0.2150413, upper bound: 0.2408513

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1_A2

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

Time for backsubstitution: 9.09 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 327

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2125432, upper bound: 0.2301669
time: 3.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2089319, upper bound: 0.2303242
time: 3.73 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.1190739, -13.1389914, -14.1281881, -13.1372633, -0.4265897, 0.4366798
1: -7.6642680, -6.8138084, -7.6673546, -6.8011432, -0.4266229, 0.4149907
2: 3.0135541, 3.9015384, 2.9769793, 3.9234052, -0.5820165, 0.6001306
3: 0.4973059, 1.2925997, 0.4929514, 1.3169785, -0.4874055, 0.4556801
4: -6.9498410, -6.0848370, -6.9608345, -6.0918446, -0.5349874, 0.5618219
5: -5.8396301, -4.9796648, -5.8422728, -4.9794006, -0.4524984, 0.4644017
6: -11.6987152, -10.5340042, -11.7106972, -10.5131741, -0.4783645, 0.4792550
7: -0.6793671, 0.0746784, -0.7045019, 0.0813317, -0.3864923, 0.4051257
8: -3.6514878, -2.8741782, -3.6525741, -2.8719552, -0.4269004, 0.4279299
9: -9.5248804, -8.4868488, -9.5233803, -8.4854355, -0.4561266, 0.4534274

Time for backsubstitution: 8.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.35 seconds

### Candidate
type: B, layer: 3, pos: 327

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2179596, upper bound: 0.2327585
time: 3.40 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2179596, upper bound: 0.2374530
time: 3.39 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1_A2

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

Time for backsubstitution: 9.10 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 327

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2086437, upper bound: 0.2301970
time: 3.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2080840, upper bound: 0.2316760
time: 3.72 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -14.1190739, -13.1389914, -14.1361732, -13.1228485, -0.4366717, 0.4399993
1: -7.6642680, -6.8138084, -7.6755619, -6.7727690, -0.4593358, 0.4268363
2: 3.0135541, 3.9015384, 2.9700537, 3.9161305, -0.5779624, 0.6098485
3: 0.4973059, 1.2925997, 0.5131121, 1.3089695, -0.5051315, 0.4600103
4: -6.9498410, -6.0848370, -6.9624190, -6.0950561, -0.5335391, 0.5641799
5: -5.8396301, -4.9796648, -5.8619180, -4.9707508, -0.4571562, 0.4790313
6: -11.6987152, -10.5340042, -11.7274647, -10.5038996, -0.4886353, 0.4941630
7: -0.6793671, 0.0746784, -0.7101896, 0.0762715, -0.3952365, 0.4246700
8: -3.6514878, -2.8741782, -3.6634769, -2.8425078, -0.4828529, 0.4647145
9: -9.5248804, -8.4868488, -9.5348387, -8.4639530, -0.4742193, 0.4607667

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 327
type: B, layer: 3, pos: 66
type: B, layer: 3, pos: 1754
type: B, layer: 3, pos: 1485
type: B, layer: 3, pos: 1706
type: B, layer: 3, pos: 768
type: B, layer: 3, pos: 221
type: B, layer: 3, pos: 3118
type: B, layer: 3, pos: 222
type: B, layer: 3, pos: 963
type: B, layer: 3, pos: 922
type: B, layer: 3, pos: 3124
type: B, layer: 3, pos: 183
type: B, layer: 3, pos: 899
type: B, layer: 3, pos: 1438
type: B, layer: 3, pos: 181
type: B, layer: 3, pos: 2899
type: B, layer: 3, pos: 1511
type: B, layer: 3, pos: 2534

Time for candidate selection: 0.47 seconds

### Candidate
type: B, layer: 3, pos: 327

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2133119, upper bound: 0.2327587
time: 3.59 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -0.2133118, upper bound: 0.2374530
time: 3.45 seconds

## Summary of splitting at layer (split count: 7)
- Time for NS candidates: 16.64 seconds
NS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2125432, upper bound: 0.2301669
NS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2089319, upper bound: 0.2303242
NS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2179596, upper bound: 0.2327585
NS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2179596, upper bound: 0.2374530
NS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2086437, upper bound: 0.2301970
NS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2080840, upper bound: 0.2316760
NS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2133119, upper bound: 0.2327587
NS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 16.64
Output dim: 3, lower bound: -0.2133118, upper bound: 0.2374530

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 55.30 + 349.92 = 405.21 seconds
