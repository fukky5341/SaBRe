## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.15455141599999997


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-12.1978397, -11.0387907, -12.1978397, -11.0387907, -0.6983652, 0.6983652)
1: (-10.2291489, -9.2301950, -10.2291489, -9.2301950, -0.5463943, 0.5463943)
2: (-8.6914253, -7.9429502, -8.6914253, -7.9429502, -0.4942775, 0.4942775)
3: (-8.3077478, -7.6052465, -8.3077478, -7.6052465, -0.4032021, 0.4032018)
4: (-3.5045655, -2.9025569, -3.5045655, -2.9025569, -0.3239938, 0.3239939)
5: (-8.5421009, -7.7244186, -8.5421009, -7.7244186, -0.4194160, 0.4194160)
6: (-13.7408257, -12.8323965, -13.7408257, -12.8323965, -0.4799125, 0.4799125)
7: (-3.5763383, -2.9654264, -3.5763383, -2.9654264, -0.4211464, 0.4211464)
8: (-0.4781022, 0.2466316, -0.4781022, 0.2466316, -0.5180578, 0.5180578)
9: (3.4875817, 4.0824232, 3.4875817, 4.0824232, -0.3047249, 0.3047249)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.85 + 33.51 = 56.36 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.1644164, upper bound: 0.1644164

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 538
type: A, layer: 1, pos: 538

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 538

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1644086, upper bound: 0.1625573
time: 3.91 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1644148, upper bound: 0.1644139
time: 3.66 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.80 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 9, lower bound: -0.1644086, upper bound: 0.1625573
NS_B2, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 9, lower bound: -0.1644148, upper bound: 0.1644139

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -12.1975031, -11.0425797, -12.1968737, -11.0496187, -0.6872334, 0.6937532
1: -10.2290459, -9.2314425, -10.2288456, -9.2337561, -0.5428162, 0.5450673
2: -8.6912594, -7.9438868, -8.6909456, -7.9456201, -0.4914379, 0.4927468
3: -8.3062801, -7.6055346, -8.3035488, -7.6060758, -0.4009781, 0.3987594
4: -3.5037756, -2.9025707, -3.5023153, -2.9025960, -0.3231505, 0.3216825
5: -8.5416403, -7.7246981, -8.5407772, -7.7252212, -0.4181695, 0.4178367
6: -13.7404699, -12.8359632, -13.7398052, -12.8425827, -0.4695399, 0.4758518
7: -3.5756140, -2.9654758, -3.5742784, -2.9655659, -0.4195490, 0.4181237
8: -0.4779358, 0.2433228, -0.4776235, 0.2371969, -0.5069695, 0.5129828
9: 3.4887967, 4.0824003, 3.4910545, 4.0823555, -0.3033664, 0.3011358

Time for backsubstitution: 21.34 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 3110
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: B, layer: 3, pos: 704
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 1689

## Relational analysis of NS_B1_B1

### Relational analysis result of NS_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1609170, upper bound: 0.1504926
time: 3.70 seconds

## Relational analysis of NS_B1_B2

### Relational analysis result of NS_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1609170, upper bound: 0.1590637
time: 4.75 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -12.1978388, -11.0387983, -12.2235670, -11.0381880, -0.6952124, 0.7063503
1: -10.2291517, -9.2301998, -10.2386217, -9.2301559, -0.5451107, 0.5564127
2: -8.6914234, -7.9429541, -8.6976194, -7.9428301, -0.4934735, 0.5004220
3: -8.3077450, -7.6052465, -8.3077869, -7.5957918, -0.4076977, 0.4019578
4: -3.5045638, -2.9025569, -3.5053174, -2.8989418, -0.3267231, 0.3241904
5: -8.5421009, -7.7244182, -8.5421095, -7.7198868, -0.4240417, 0.4194393
6: -13.7408247, -12.8324060, -13.7640152, -12.8319912, -0.4769876, 0.4851284
7: -3.5763359, -2.9654269, -3.5778308, -2.9599838, -0.4248335, 0.4235435
8: -0.4781017, 0.2466230, -0.5014777, 0.2490604, -0.5203300, 0.5217521
9: 3.4875841, 4.0824246, 3.4870310, 4.0906291, -0.3092139, 0.3042519

Time for backsubstitution: 20.94 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1689
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: B, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414

Time for candidate selection: 0.37 seconds

### Candidate
type: B, layer: 3, pos: 1689

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1609236, upper bound: 0.1523496
time: 3.70 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1609236, upper bound: 0.1609221
time: 3.83 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 28.85 seconds
NS_B1_B1, status: Status.UNKNOWN, split count: 2, time: 28.85
Output dim: 9, lower bound: -0.1609170, upper bound: 0.1504926
NS_B1_B2, status: Status.UNKNOWN, split count: 2, time: 28.85
Output dim: 9, lower bound: -0.1609170, upper bound: 0.1590637
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 28.85
Output dim: 9, lower bound: -0.1609236, upper bound: 0.1523496
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 28.85
Output dim: 9, lower bound: -0.1609236, upper bound: 0.1609221

## BFS NS instance: NS_B1_B1

### Backsubstitution after applying NS history:
0: -12.1971655, -11.0428915, -12.1957874, -11.0506153, -0.6830153, 0.6875730
1: -10.2290430, -9.2362366, -10.2288389, -9.2483110, -0.5316019, 0.5404811
2: -8.6912556, -7.9493756, -8.6909332, -7.9624977, -0.4738154, 0.4867578
3: -8.3062763, -7.6153369, -8.3035393, -7.6363668, -0.3715472, 0.3895836
4: -3.4992127, -2.9025822, -3.4876485, -2.9026337, -0.3179500, 0.3061368
5: -8.5405750, -7.7248302, -8.5374537, -7.7256470, -0.4159403, 0.4150584
6: -13.7401733, -12.8359623, -13.7388639, -12.8425817, -0.4680102, 0.4722147
7: -3.5756023, -2.9688368, -3.5742383, -2.9763691, -0.4097424, 0.4149079
8: -0.4739685, 0.2433205, -0.4648695, 0.2371898, -0.5027757, 0.4995170
9: 3.4944692, 4.0823822, 3.5085974, 4.0823011, -0.2976692, 0.2872324

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3110
type: B, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 2817
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1739

## Relational analysis of NS_B1_B1_A1

### Relational analysis result of NS_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1568309, upper bound: 0.1413229
time: 3.57 seconds

## Relational analysis of NS_B1_B1_A2

### Relational analysis result of NS_B1_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1536730, upper bound: 0.1431563
time: 4.10 seconds

## BFS NS instance: NS_B1_B2

### Backsubstitution after applying NS history:
0: -12.1973162, -11.0596447, -12.1886177, -11.0851631, -0.6872563, 0.6877356
1: -10.2290421, -9.2397299, -10.2580509, -9.2516499, -0.5354271, 0.5654836
2: -8.6912422, -7.9531546, -8.7250900, -7.9649119, -0.4796124, 0.5350561
3: -8.3062649, -7.6096039, -8.3733511, -7.6143332, -0.3857057, 0.4672313
4: -3.4991341, -2.9025831, -3.4931655, -2.8693466, -0.3625410, 0.3123848
5: -8.5397339, -7.7247701, -8.5366611, -7.7179594, -0.4162891, 0.4216115
6: -13.7402973, -12.8359623, -13.7433090, -12.8419437, -0.4784245, 0.4713147
7: -3.5756047, -2.9688611, -3.5976233, -2.9718523, -0.4159150, 0.4361467
8: -0.4736218, 0.2433155, -0.4692717, 0.2644713, -0.5341072, 0.5045595
9: 3.4955120, 4.0823851, 3.5047998, 4.1212940, -0.3451018, 0.2962480

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1689
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3110
type: B, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1739

## Relational analysis of NS_B1_B2_A1

### Relational analysis result of NS_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1568309, upper bound: 0.1501154
time: 3.74 seconds

## Relational analysis of NS_B1_B2_A2

### Relational analysis result of NS_B1_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1536730, upper bound: 0.1518195
time: 3.90 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -12.1974964, -11.0391073, -12.2224712, -11.0391827, -0.6909924, 0.7000995
1: -10.2291498, -9.2349930, -10.2386198, -9.2447147, -0.5338945, 0.5517135
2: -8.6914177, -7.9484434, -8.6976032, -7.9597006, -0.4758501, 0.4944329
3: -8.3077402, -7.6150484, -8.3077774, -7.6260858, -0.3785129, 0.3927832
4: -3.5000007, -2.9025688, -3.4906495, -2.8989804, -0.3215284, 0.3086438
5: -8.5410376, -7.7245531, -8.5387878, -7.7203116, -0.4218071, 0.4166617
6: -13.7405300, -12.8324060, -13.7630463, -12.8319931, -0.4754570, 0.4814348
7: -3.5763226, -2.9687872, -3.5777929, -2.9707870, -0.4151597, 0.4203281
8: -0.4741340, 0.2466218, -0.4887209, 0.2490532, -0.5161371, 0.5082757
9: 3.4932561, 4.0824070, 3.5045724, 4.0905743, -0.3036897, 0.2903512

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 704
type: B, layer: 3, pos: 3110
type: A, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1739

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1568385, upper bound: 0.1431871
time: 3.76 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1536810, upper bound: 0.1450203
time: 4.46 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -12.1976509, -11.0558624, -12.2152405, -11.0737333, -0.6952348, 0.7022038
1: -10.2291498, -9.2384892, -10.2678280, -9.2480555, -0.5377226, 0.5767727
2: -8.6914043, -7.9522238, -8.7317638, -7.9621158, -0.4816480, 0.5413902
3: -8.3077316, -7.6093163, -8.3775864, -7.6040654, -0.3926389, 0.4704275
4: -3.4999211, -2.9025686, -3.4961686, -2.8656979, -0.3658819, 0.3148918
5: -8.5401936, -7.7244954, -8.5379925, -7.7126427, -0.4221308, 0.4232316
6: -13.7406511, -12.8324070, -13.7673702, -12.8313560, -0.4858718, 0.4806619
7: -3.5763264, -2.9688110, -3.6011841, -2.9662700, -0.4213736, 0.4415617
8: -0.4737892, 0.2466171, -0.4931180, 0.2763331, -0.5470958, 0.5135038
9: 3.4942975, 4.0824089, 3.5007811, 4.1295614, -0.3509808, 0.2993672

Time for backsubstitution: 24.54 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1739
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1739

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.1568385, upper bound: 0.1519802
time: 4.80 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1536810, upper bound: 0.1536795
time: 3.94 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.49 seconds
NS_B1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1568309, upper bound: 0.1413229
NS_B1_B1_A2, status: Status.VERIFIED, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1536730, upper bound: 0.1431563
NS_B1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1568309, upper bound: 0.1501154
NS_B1_B2_A2, status: Status.VERIFIED, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1536730, upper bound: 0.1518195
NS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1568385, upper bound: 0.1431871
NS_B2_B1_A2, status: Status.VERIFIED, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1536810, upper bound: 0.1450203
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1568385, upper bound: 0.1519802
NS_B2_B2_A2, status: Status.VERIFIED, split count: 3, time: 33.49
Output dim: 9, lower bound: -0.1536810, upper bound: 0.1536795

## BFS NS instance: NS_B1_B1_A1

### Backsubstitution after applying NS history:
0: -12.1966610, -11.0450144, -12.1956835, -11.0509186, -0.6827297, 0.6868701
1: -10.2279530, -9.2315502, -10.2287006, -9.2483253, -0.5307612, 0.5447402
2: -8.6906891, -7.9455137, -8.6908598, -7.9627032, -0.4724240, 0.4908228
3: -8.3062792, -7.6069217, -8.3035393, -7.6365409, -0.3713772, 0.3974833
4: -3.4996214, -2.9025707, -3.4871259, -2.9026339, -0.3165293, 0.3052363
5: -8.5378952, -7.7253594, -8.5369778, -7.7257309, -0.4124327, 0.4135401
6: -13.7365808, -12.8359728, -13.7383728, -12.8425827, -0.4659362, 0.4719391
7: -3.5755932, -2.9752252, -3.5742362, -2.9775934, -0.4082565, 0.4062853
8: -0.4771214, 0.2431865, -0.4647646, 0.2371716, -0.5055709, 0.4992523
9: 3.4929552, 4.0823998, 3.5091205, 4.0823011, -0.2965162, 0.2863598

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3110
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 166
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_B1_B1_A1_A1

### Relational analysis result of NS_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1496035, upper bound: 0.1413229
time: 3.81 seconds

## Relational analysis of NS_B1_B1_A1_A2

### Relational analysis result of NS_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1496035, upper bound: 0.1413229
time: 4.26 seconds

## BFS NS instance: NS_B1_B2_A1

### Backsubstitution after applying NS history:
0: -12.1966610, -11.0450144, -12.1885118, -11.0854540, -0.6860929, 0.6972208
1: -10.2279530, -9.2315502, -10.2579117, -9.2516594, -0.5345907, 0.5738897
2: -8.6906891, -7.9455137, -8.7250214, -7.9651155, -0.4782271, 0.5377717
3: -8.3062792, -7.6069217, -8.3733492, -7.6145096, -0.3855379, 0.4707451
4: -3.4996214, -2.9025707, -3.4926438, -2.8693461, -0.3582380, 0.3115387
5: -8.5378952, -7.7253594, -8.5361900, -7.7180390, -0.4149506, 0.4199274
6: -13.7365808, -12.8359728, -13.7428169, -12.8419466, -0.4767947, 0.4745243
7: -3.5755932, -2.9752252, -3.5976219, -2.9732265, -0.4144902, 0.4283772
8: -0.4771214, 0.2431865, -0.4691675, 0.2644546, -0.5360460, 0.5042987
9: 3.4929552, 4.0823998, 3.5053549, 4.1212950, -0.3432040, 0.2953796

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 2131
type: B, layer: 3, pos: 1739
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3110
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 655
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_B1_B2_A1_A1

### Relational analysis result of NS_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1483087, upper bound: 0.1501152
time: 4.24 seconds

## Relational analysis of NS_B1_B2_A1_A2

### Relational analysis result of NS_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1483087, upper bound: 0.1413229
time: 4.10 seconds

## BFS NS instance: NS_B2_B1_A1

### Backsubstitution after applying NS history:
0: -12.1970053, -11.0412312, -12.2223701, -11.0394869, -0.6907072, 0.6994872
1: -10.2280560, -9.2303028, -10.2384834, -9.2447271, -0.5330544, 0.5560918
2: -8.6908588, -7.9445810, -8.6975365, -7.9599066, -0.4744554, 0.4985027
3: -8.3077431, -7.6066341, -8.3077774, -7.6262555, -0.3783426, 0.4006875
4: -3.5004065, -2.9025581, -3.4901264, -2.8989799, -0.3201898, 0.3077412
5: -8.5383568, -7.7250791, -8.5383139, -7.7203922, -0.4183033, 0.4151471
6: -13.7369356, -12.8324165, -13.7625589, -12.8319931, -0.4733825, 0.4811788
7: -3.5763144, -2.9751766, -3.5777888, -2.9720101, -0.4136810, 0.4117022
8: -0.4772844, 0.2464876, -0.4886208, 0.2490349, -0.5189257, 0.5080123
9: 3.4917421, 4.0824232, 3.5050945, 4.0905743, -0.3021709, 0.2894781

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3110
type: B, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 2817
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_B2_B1_A1_A1

### Relational analysis result of NS_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1496118, upper bound: 0.1431872
time: 4.26 seconds

## Relational analysis of NS_B2_B1_A1_A2

### Relational analysis result of NS_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1496118, upper bound: 0.1431872
time: 3.97 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -12.1970053, -11.0412312, -12.2151394, -11.0740223, -0.6940718, 0.7117252
1: -10.2280560, -9.2303028, -10.2676907, -9.2480659, -0.5368834, 0.5852437
2: -8.6908588, -7.9445810, -8.7316942, -7.9623194, -0.4802589, 0.5443835
3: -8.3077431, -7.6066341, -8.3775864, -7.6042366, -0.3924711, 0.4739466
4: -3.5004065, -2.9025581, -3.4956431, -2.8656979, -0.3615751, 0.3140435
5: -8.5383568, -7.7250791, -8.5375233, -7.7127223, -0.4207954, 0.4215345
6: -13.7369356, -12.8324165, -13.7668791, -12.8313560, -0.4842424, 0.4837151
7: -3.5763144, -2.9751766, -3.6011825, -2.9676425, -0.4199240, 0.4337921
8: -0.4772844, 0.2464876, -0.4930172, 0.2763157, -0.5490141, 0.5132434
9: 3.4917421, 4.0824232, 3.5013361, 4.1295605, -0.3488123, 0.2984982

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1689
type: B, layer: 3, pos: 1689
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 2131
type: A, layer: 3, pos: 2131
type: B, layer: 3, pos: 1739
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1858
type: A, layer: 3, pos: 1858
type: B, layer: 3, pos: 1690
type: A, layer: 3, pos: 1690
type: A, layer: 3, pos: 3110
type: A, layer: 3, pos: 704
type: B, layer: 3, pos: 3110
type: B, layer: 3, pos: 704
type: A, layer: 3, pos: 2572
type: B, layer: 3, pos: 2572
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 655
type: A, layer: 3, pos: 655
type: A, layer: 3, pos: 724
type: B, layer: 3, pos: 724
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 166
type: A, layer: 3, pos: 166
type: A, layer: 3, pos: 2817
type: B, layer: 3, pos: 2817
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 1689

## Relational analysis of NS_B2_B2_A1_A1

### Relational analysis result of NS_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1483172, upper bound: 0.1519807
time: 4.27 seconds

## Relational analysis of NS_B2_B2_A1_A2

### Relational analysis result of NS_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.1483172, upper bound: 0.1431872
time: 4.21 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.18 seconds
NS_B1_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1496035, upper bound: 0.1413229
NS_B1_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1496035, upper bound: 0.1413229
NS_B1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1483087, upper bound: 0.1501152
NS_B1_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1483087, upper bound: 0.1413229
NS_B2_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1496118, upper bound: 0.1431872
NS_B2_B1_A1_A2, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1496118, upper bound: 0.1431872
NS_B2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1483172, upper bound: 0.1519807
NS_B2_B2_A1_A2, status: Status.VERIFIED, split count: 4, time: 31.18
Output dim: 9, lower bound: -0.1483172, upper bound: 0.1431872

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 56.36 + 311.11 = 367.47 seconds
