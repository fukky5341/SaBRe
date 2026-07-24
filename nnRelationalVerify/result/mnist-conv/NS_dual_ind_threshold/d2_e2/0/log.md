## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.32452505010000005


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-10.5224762, -9.6512756, -10.5224762, -9.6512756, -0.6905186, 0.6905186)
1: (-4.4979005, -3.7255995, -4.4979005, -3.7255995, -0.5279188, 0.5279183)
2: (10.2145462, 10.9128647, 10.2145462, 10.9128647, -0.5151484, 0.5151484)
3: (-3.8954210, -3.0341797, -3.8954210, -3.0341797, -0.6740751, 0.6740749)
4: (-6.6635189, -5.8304825, -6.6635189, -5.8304825, -0.5155964, 0.5155964)
5: (-10.9791746, -10.1283855, -10.9791746, -10.1283855, -0.6018240, 0.6018243)
6: (-13.1994486, -12.0746469, -13.1994486, -12.0746469, -0.6701522, 0.6701524)
7: (-4.3303819, -3.5318575, -4.3303819, -3.5318575, -0.6258245, 0.6258245)
8: (-4.2767587, -3.6702390, -4.2767587, -3.6702390, -0.4052844, 0.4052843)
9: (-10.8552456, -9.7669811, -10.8552456, -9.7669811, -0.6959014, 0.6959014)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.28 + 35.85 = 58.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3248499, upper bound: 0.3248496

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4637
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4637

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248329, upper bound: 0.3222888
time: 6.22 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248405, upper bound: 0.3248399
time: 6.18 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.59 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.59
Output dim: 2, lower bound: -0.3248329, upper bound: 0.3222888
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.59
Output dim: 2, lower bound: -0.3248405, upper bound: 0.3248399

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.5213499, -9.6512880, -10.5222073, -9.6512794, -0.6881011, 0.6892552
1: -4.4963279, -3.7257469, -4.4975171, -3.7256353, -0.5263090, 0.5273905
2: 10.2148781, 10.9095173, 10.2146263, 10.9120474, -0.5141523, 0.5117414
3: -3.8948269, -3.0363026, -3.8952770, -3.0346980, -0.6733437, 0.6719174
4: -6.6629868, -5.8307652, -6.6633825, -5.8305521, -0.5145619, 0.5151644
5: -10.9780111, -10.1284533, -10.9788990, -10.1283998, -0.5993204, 0.6011691
6: -13.1955643, -12.0748177, -13.1984978, -12.0746889, -0.6662216, 0.6690567
7: -4.3297710, -3.5361748, -4.3302326, -3.5329137, -0.6246605, 0.6216922
8: -4.2740793, -3.6704555, -4.2761021, -3.6702909, -0.4026055, 0.4044324
9: -10.8540497, -9.7671270, -10.8549414, -9.7670174, -0.6937571, 0.6953025

Time for backsubstitution: 21.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226732, upper bound: 0.3222829
time: 5.95 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248267, upper bound: 0.3222830
time: 6.41 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.5293617, -9.6499119, -10.5224752, -9.6512756, -0.7008815, 0.6911805
1: -4.4978981, -3.7224665, -4.4978981, -3.7256005, -0.5274189, 0.5311325
2: 10.2074833, 10.9128761, 10.2145452, 10.9128590, -0.5206146, 0.5142050
3: -3.8994017, -3.0339899, -3.8954215, -3.0341830, -0.6781464, 0.6742857
4: -6.6651974, -5.8286457, -6.6635189, -5.8304834, -0.5187693, 0.5174778
5: -10.9864902, -10.1253490, -10.9791718, -10.1283865, -0.6128607, 0.6048698
6: -13.1999245, -12.0680418, -13.1994362, -12.0746469, -0.6694212, 0.6766808
7: -4.3372245, -3.5313540, -4.3303814, -3.5318642, -0.6328063, 0.6260335
8: -4.2768660, -3.6652477, -4.2767549, -3.6702390, -0.4046373, 0.4104322
9: -10.8582878, -9.7633514, -10.8552427, -9.7669821, -0.7023654, 0.6995924

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248339
time: 7.50 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248343, upper bound: 0.3248345
time: 5.10 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 34.16 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 34.16
Output dim: 2, lower bound: -0.3226732, upper bound: 0.3222829
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 2, lower bound: -0.3248267, upper bound: 0.3222830
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248339
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.16
Output dim: 2, lower bound: -0.3248343, upper bound: 0.3248345

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -10.5213451, -9.6512890, -10.5241394, -9.6474571, -0.6922741, 0.6912050
1: -4.4963274, -3.7257500, -4.5018754, -3.7252669, -0.5260201, 0.5316231
2: 10.2148800, 10.9095135, 10.2082024, 10.9123402, -0.5152395, 0.5178545
3: -3.8948212, -3.0363016, -3.8968267, -3.0281000, -0.6792555, 0.6723018
4: -6.6629782, -5.8307657, -6.6636944, -5.8152504, -0.5239308, 0.5120273
5: -10.9780111, -10.1284561, -10.9843521, -10.1282072, -0.5990682, 0.6067269
6: -13.1955633, -12.0748215, -13.2081375, -12.0744038, -0.6645699, 0.6776114
7: -4.3297658, -3.5361729, -4.3320017, -3.5250053, -0.6324997, 0.6216657
8: -4.2740784, -3.6704574, -4.2801170, -3.6698744, -0.4027944, 0.4088840
9: -10.8540344, -9.7671270, -10.8552122, -9.7489443, -0.6996803, 0.6917167

Time for backsubstitution: 21.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3247635, upper bound: 0.3199172
time: 5.76 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248221, upper bound: 0.3222782
time: 5.20 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -10.5285234, -9.6501970, -10.5193071, -9.6523914, -0.6993506, 0.6879456
1: -4.4978933, -3.7232106, -4.4978833, -3.7284431, -0.5241938, 0.5301638
2: 10.2076378, 10.9122162, 10.2151442, 10.9103498, -0.5168420, 0.5125673
3: -3.8983345, -3.0340099, -3.8913441, -3.0342607, -0.6766415, 0.6698272
4: -6.6627197, -5.8287573, -6.6540518, -5.8309135, -0.5157881, 0.5076785
5: -10.9862242, -10.1258726, -10.9781380, -10.1303730, -0.6105945, 0.6032786
6: -13.1997309, -12.0695238, -13.1986637, -12.0802937, -0.6635666, 0.6743999
7: -4.3360500, -3.5314693, -4.3259015, -3.5323076, -0.6310768, 0.6212983
8: -4.2765455, -3.6658516, -4.2755046, -3.6725440, -0.4021780, 0.4092464
9: -10.8552017, -9.7635727, -10.8434563, -9.7678318, -0.6985192, 0.6874158

Time for backsubstitution: 21.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4653

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3226802
time: 5.74 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248344
time: 6.14 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -10.5293570, -9.6499128, -10.5244017, -9.6474552, -0.7042570, 0.6931255
1: -4.4978981, -3.7224689, -4.5022564, -3.7252326, -0.5271297, 0.5353649
2: 10.2074852, 10.9128733, 10.2081203, 10.9131527, -0.5198776, 0.5203352
3: -3.8993979, -3.0339899, -3.8969669, -3.0275850, -0.6810486, 0.6746697
4: -6.6651888, -5.8286457, -6.6638303, -5.8151822, -0.5269053, 0.5143390
5: -10.9864893, -10.1253548, -10.9846296, -10.1281900, -0.6126080, 0.6104321
6: -13.1999245, -12.0680447, -13.2090778, -12.0743647, -0.6677675, 0.6795573
7: -4.3372188, -3.5313549, -4.3321466, -3.5239553, -0.6354487, 0.6260033
8: -4.2768650, -3.6652484, -4.2807713, -3.6698239, -0.4048252, 0.4124565
9: -10.8582754, -9.7633553, -10.8555155, -9.7489109, -0.7056863, 0.6960058

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3247710, upper bound: 0.3224674
time: 5.57 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248296, upper bound: 0.3248286
time: 10.75 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 38.31 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 2, lower bound: -0.3247635, upper bound: 0.3199172
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 2, lower bound: -0.3248221, upper bound: 0.3222782
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 38.31
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3226802
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248344
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 2, lower bound: -0.3247710, upper bound: 0.3224674
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 2, lower bound: -0.3248296, upper bound: 0.3248286

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -10.5197477, -9.6521187, -10.5238066, -9.6476231, -0.6898789, 0.6892762
1: -4.4898787, -3.7292931, -4.4998727, -3.7256904, -0.5191212, 0.5244308
2: 10.2174797, 10.9023504, 10.2083998, 10.9101200, -0.5097694, 0.5104489
3: -3.8887901, -3.0381098, -3.8949671, -3.0281696, -0.6736691, 0.6680646
4: -6.6606445, -5.8394003, -6.6636696, -5.8178611, -0.5163050, 0.5034311
5: -10.9727440, -10.1299629, -10.9827957, -10.1282349, -0.5942042, 0.6035290
6: -13.1940880, -12.0754204, -13.2078609, -12.0745897, -0.6628294, 0.6766818
7: -4.3258362, -3.5480824, -4.3317380, -3.5286565, -0.6226084, 0.6096702
8: -4.2694821, -3.6719732, -4.2787290, -3.6699502, -0.3981040, 0.4060084
9: -10.8522396, -9.7730885, -10.8551617, -9.7507725, -0.6942220, 0.6857429

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222199, upper bound: 0.3199172
time: 6.53 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222199, upper bound: 0.3199165
time: 6.23 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -10.5213442, -9.6512871, -10.5241394, -9.6474571, -0.6952672, 0.6904588
1: -4.4963274, -3.7257509, -4.5018754, -3.7252669, -0.5230212, 0.5310926
2: 10.2148800, 10.9095106, 10.2082024, 10.9123402, -0.5151956, 0.5113065
3: -3.8948183, -3.0363007, -3.8968267, -3.0281000, -0.6754827, 0.6720786
4: -6.6629782, -5.8307686, -6.6636944, -5.8152504, -0.5216630, 0.5059717
5: -10.9780064, -10.1284561, -10.9843521, -10.1282072, -0.5957842, 0.6067271
6: -13.1955624, -12.0748224, -13.2081375, -12.0744038, -0.6643682, 0.6779568
7: -4.3297663, -3.5361786, -4.3320017, -3.5250053, -0.6305413, 0.6151984
8: -4.2740774, -3.6704564, -4.2801170, -3.6698744, -0.4009748, 0.4088838
9: -10.8540344, -9.7671299, -10.8552122, -9.7489443, -0.6980858, 0.6881320

Time for backsubstitution: 21.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222784, upper bound: 0.3222778
time: 5.83 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222784, upper bound: 0.3222783
time: 5.44 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5193071, -9.6523914, -0.7021029, 0.6921680
1: -4.5022573, -3.7220972, -4.4978833, -3.7284431, -0.5284653, 0.5312045
2: 10.2010536, 10.9131718, 10.2151442, 10.9103498, -0.5189172, 0.5127590
3: -3.9009418, -3.0273905, -3.8913441, -3.0342607, -0.6787267, 0.6760230
4: -6.6655350, -5.8133445, -6.6540518, -5.8309135, -0.5184207, 0.5156674
5: -10.9918747, -10.1251602, -10.9781380, -10.1303730, -0.6163433, 0.6039414
6: -13.2095709, -12.0677557, -13.1986637, -12.0802937, -0.6726964, 0.6760061
7: -4.3389769, -3.5234466, -4.3259015, -3.5323076, -0.6327863, 0.6292739
8: -4.2808857, -3.6648326, -4.2755046, -3.6725440, -0.4067822, 0.4101733
9: -10.8586035, -9.7452812, -10.8434563, -9.7678318, -0.6996162, 0.6900642

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201282, upper bound: 0.3248268
time: 6.38 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201282, upper bound: 0.3248273
time: 5.66 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -10.5277691, -9.6507425, -10.5240688, -9.6476221, -0.7018616, 0.6911986
1: -4.4914479, -3.7260108, -4.5002542, -3.7256556, -0.5202320, 0.5281727
2: 10.2100773, 10.9057083, 10.2083158, 10.9109325, -0.5122732, 0.5129302
3: -3.8933640, -3.0357909, -3.8951073, -3.0276556, -0.6754379, 0.6704321
4: -6.6628566, -5.8372841, -6.6638060, -5.8177929, -0.5192790, 0.5057361
5: -10.9812078, -10.1268578, -10.9830704, -10.1282167, -0.6077228, 0.6072345
6: -13.1984577, -12.0686398, -13.2088013, -12.0745449, -0.6660333, 0.6786265
7: -4.3332911, -3.5432634, -4.3318834, -3.5276103, -0.6244092, 0.6140089
8: -4.2722688, -3.6667628, -4.2793818, -3.6699007, -0.4001355, 0.4081488
9: -10.8564787, -9.7693176, -10.8554621, -9.7507353, -0.7002277, 0.6900315

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222194, upper bound: 0.3224599
time: 8.39 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3222194, upper bound: 0.3224678
time: 6.59 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.5293579, -9.6499138, -10.5244017, -9.6474552, -0.7060955, 0.6923788
1: -4.4978976, -3.7224696, -4.5022564, -3.7252326, -0.5241323, 0.5347357
2: 10.2074833, 10.9128685, 10.2081203, 10.9131527, -0.5179499, 0.5137875
3: -3.8993931, -3.0339909, -3.8969669, -3.0275850, -0.6772661, 0.6744452
4: -6.6651897, -5.8286510, -6.6638303, -5.8151822, -0.5246375, 0.5082834
5: -10.9864864, -10.1253529, -10.9846296, -10.1281900, -0.6093063, 0.6104317
6: -13.1999264, -12.0680456, -13.2090778, -12.0743647, -0.6675644, 0.6799018
7: -4.3372183, -3.5313597, -4.3321466, -3.5239553, -0.6323533, 0.6195359
8: -4.2768631, -3.6652489, -4.2807713, -3.6698239, -0.4030054, 0.4112134
9: -10.8582735, -9.7633572, -10.8555155, -9.7489109, -0.7040925, 0.6924214

Time for backsubstitution: 21.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3222780, upper bound: 0.3248217
time: 5.78 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3222780, upper bound: 0.3248224
time: 4.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.92 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222199, upper bound: 0.3199172
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222199, upper bound: 0.3199165
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222784, upper bound: 0.3222778
NS_A1_B2_A2_B2, status: Status.VERIFIED, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222784, upper bound: 0.3222783
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3201282, upper bound: 0.3248268
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3201282, upper bound: 0.3248273
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222194, upper bound: 0.3224599
NS_A2_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222194, upper bound: 0.3224678
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222780, upper bound: 0.3248217
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.92
Output dim: 2, lower bound: -0.3222780, upper bound: 0.3248224

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5181808, -9.6524010, -0.6978812, 0.6899230
1: -4.5022573, -3.7220972, -4.4963131, -3.7285893, -0.5288210, 0.5296304
2: 10.2010536, 10.9131718, 10.2154713, 10.9070072, -0.5155451, 0.5135360
3: -3.9009418, -3.0273905, -3.8907442, -3.0363779, -0.6766241, 0.6751723
4: -6.6655350, -5.8133445, -6.6535053, -5.8311944, -0.5165334, 0.5146822
5: -10.9918747, -10.1251602, -10.9769993, -10.1304436, -0.6126390, 0.6014762
6: -13.2095709, -12.0677557, -13.1947918, -12.0804615, -0.6724637, 0.6721022
7: -4.3389769, -3.5234466, -4.3252845, -3.5366192, -0.6286945, 0.6300941
8: -4.2808857, -3.6648326, -4.2728310, -3.6727605, -0.4074416, 0.4075385
9: -10.8586035, -9.7452812, -10.8422413, -9.7679768, -0.6981645, 0.6879396

Time for backsubstitution: 20.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3200650, upper bound: 0.3224592
time: 7.16 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248212
time: 5.21 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5261660, -9.6510267, -0.7021055, 0.7017274
1: -4.5022573, -3.7220972, -4.4978828, -3.7253094, -0.5293250, 0.5283494
2: 10.2010536, 10.9131718, 10.2080708, 10.9103661, -0.5189354, 0.5155878
3: -3.9009418, -3.0273905, -3.8953209, -3.0340657, -0.6787262, 0.6768954
4: -6.6655350, -5.8133445, -6.6557350, -5.8290758, -0.5190194, 0.5178926
5: -10.9918747, -10.1251602, -10.9854727, -10.1273384, -0.6167848, 0.6124144
6: -13.2095709, -12.0677557, -13.1991558, -12.0736876, -0.6731899, 0.6697845
7: -4.3389769, -3.5234466, -4.3327417, -3.5317960, -0.6283808, 0.6310728
8: -4.2808857, -3.6648326, -4.2756171, -3.6675589, -0.4078541, 0.4054742
9: -10.8586035, -9.7452812, -10.8465042, -9.7642012, -0.6998682, 0.6945608

Time for backsubstitution: 21.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3200650, upper bound: 0.3224680
time: 3.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248217
time: 4.33 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -10.5293579, -9.6499138, -10.5232925, -9.6474648, -0.7038257, 0.6901510
1: -4.4978976, -3.7224696, -4.5006862, -3.7253766, -0.5244851, 0.5331562
2: 10.2074833, 10.9128685, 10.2084560, 10.9098101, -0.5145779, 0.5134187
3: -3.8993931, -3.0339909, -3.8963876, -3.0297027, -0.6751585, 0.6742773
4: -6.6651897, -5.8286510, -6.6632948, -5.8154626, -0.5239425, 0.5073009
5: -10.9864864, -10.1253529, -10.9834576, -10.1282578, -0.6056054, 0.6079245
6: -13.1999264, -12.0680456, -13.2052021, -12.0745277, -0.6685953, 0.6759980
7: -4.3372183, -3.5313597, -4.3315516, -3.5282660, -0.6282520, 0.6203632
8: -4.2768631, -3.6652489, -4.2780943, -3.6700377, -0.4036684, 0.4085753
9: -10.8582735, -9.7633572, -10.8543186, -9.7490530, -0.7029445, 0.6903040

Time for backsubstitution: 21.08 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 4653

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3226675
time: 6.67 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248217
time: 4.12 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -10.5293579, -9.6499138, -10.5312920, -9.6460924, -0.7061002, 0.7020853
1: -4.4978976, -3.7224696, -4.5022573, -3.7220972, -0.5249889, 0.5319779
2: 10.2074833, 10.9128685, 10.2010536, 10.9131718, -0.5179694, 0.5149791
3: -3.8993931, -3.0339909, -3.9009418, -3.0273905, -0.6775351, 0.6783004
4: -6.6651897, -5.8286510, -6.6655350, -5.8133445, -0.5250461, 0.5101755
5: -10.9864864, -10.1253529, -10.9918747, -10.1251602, -0.6097479, 0.6187823
6: -13.1999264, -12.0680456, -13.2095709, -12.0677557, -0.6685829, 0.6803672
7: -4.3372183, -3.5313597, -4.3389769, -3.5234466, -0.6335642, 0.6218898
8: -4.2768631, -3.6652489, -4.2808857, -3.6648326, -0.4040816, 0.4101665
9: -10.8582735, -9.7633572, -10.8586035, -9.7452812, -0.7043464, 0.6959944

Time for backsubstitution: 21.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4653
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 4653

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3201235, upper bound: 0.3226676
time: 6.40 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201235, upper bound: 0.3248298
time: 6.68 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.88 seconds
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3200650, upper bound: 0.3224592
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248212
NS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3200650, upper bound: 0.3224680
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248217
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3226675
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248217
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3201235, upper bound: 0.3226676
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 34.88
Output dim: 2, lower bound: -0.3201235, upper bound: 0.3248298

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5181808, -9.6524010, -0.7008750, 0.6891766
1: -4.5022559, -3.7220979, -4.4963131, -3.7285893, -0.5258203, 0.5290990
2: 10.2010574, 10.9131689, 10.2154713, 10.9070072, -0.5136173, 0.5070219
3: -3.9009371, -3.0273914, -3.8907442, -3.0363779, -0.6732163, 0.6734688
4: -6.6655350, -5.8133478, -6.6535053, -5.8311944, -0.5165334, 0.5086021
5: -10.9918690, -10.1251602, -10.9769993, -10.1304436, -0.6093416, 0.6014760
6: -13.2095699, -12.0677576, -13.1947918, -12.0804615, -0.6718240, 0.6724517
7: -4.3389754, -3.5234509, -4.3252845, -3.5366192, -0.6262240, 0.6236260
8: -4.2808824, -3.6648331, -4.2728310, -3.6727605, -0.4056222, 0.4062966
9: -10.8586035, -9.7452850, -10.8422413, -9.7679768, -0.6968887, 0.6843405

Time for backsubstitution: 20.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4615

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3247627
time: 8.53 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3248219
time: 4.20 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5261660, -9.6510267, -0.7050998, 0.7008739
1: -4.5022559, -3.7220979, -4.4978828, -3.7253094, -0.5263257, 0.5278177
2: 10.2010574, 10.9131689, 10.2080708, 10.9103661, -0.5170076, 0.5090773
3: -3.9009371, -3.0273914, -3.8953209, -3.0340657, -0.6753221, 0.6751919
4: -6.6655350, -5.8133478, -6.6557350, -5.8290758, -0.5190191, 0.5118126
5: -10.9918690, -10.1251602, -10.9854727, -10.1273384, -0.6134870, 0.6124139
6: -13.2095699, -12.0677576, -13.1991558, -12.0736876, -0.6725502, 0.6706798
7: -4.3389754, -3.5234509, -4.3327417, -3.5317960, -0.6283813, 0.6245990
8: -4.2808824, -3.6648331, -4.2756171, -3.6675589, -0.4060349, 0.4054737
9: -10.8586035, -9.7452850, -10.8465042, -9.7642012, -0.6982741, 0.6909616

Time for backsubstitution: 21.11 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 4615

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177838, upper bound: 0.3247706
time: 5.91 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177838, upper bound: 0.3248293
time: 6.25 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.13 + 553.14 = 611.27 seconds
