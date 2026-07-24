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
execution time: IAR + RelationalAnalysis = 22.00 + 35.99 = 58.00 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.3248499, upper bound: 0.3248496

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4653
type: B, layer: 1, pos: 4653
type: A, layer: 1, pos: 4637
type: B, layer: 1, pos: 4637
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4653

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248440, upper bound: 0.3226902
time: 6.52 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248440, upper bound: 0.3248435
time: 5.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.67 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.67
Output dim: 2, lower bound: -0.3248440, upper bound: 0.3226902
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.67
Output dim: 2, lower bound: -0.3248440, upper bound: 0.3248435

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -10.5193090, -9.6523905, -10.5216465, -9.6515608, -0.6872821, 0.6889911
1: -4.4978857, -3.7284427, -4.4978962, -3.7263455, -0.5269504, 0.5246947
2: 10.2151451, 10.9103556, 10.2147036, 10.9122066, -0.5135100, 0.5114243
3: -3.8913455, -3.0342579, -3.8943524, -3.0342011, -0.6696165, 0.6725719
4: -6.6540518, -5.8309150, -6.6610403, -5.8305941, -0.5057974, 0.5126178
5: -10.9781399, -10.1303730, -10.9789047, -10.1289053, -0.6002328, 0.5995479
6: -13.1986771, -12.0802889, -13.1992474, -12.0761280, -0.6678724, 0.6642971
7: -4.3259020, -3.5323009, -4.3292093, -3.5319710, -0.6210871, 0.6240954
8: -4.2755079, -3.6725445, -4.2764359, -3.6708424, -0.4040990, 0.4028248
9: -10.8434591, -9.7678308, -10.8521585, -9.7672005, -0.6837239, 0.6920581

Time for backsubstitution: 20.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4637
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 4637

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3201287
time: 5.65 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248346, upper bound: 0.3226801
time: 6.38 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -10.5244026, -9.6474533, -10.5224743, -9.6512756, -0.6924622, 0.6946912
1: -4.5022593, -3.7252312, -4.4979000, -3.7256038, -0.5321510, 0.5276308
2: 10.2081203, 10.9131584, 10.2145452, 10.9128609, -0.5212669, 0.5162368
3: -3.8969669, -3.0275803, -3.8954163, -3.0341797, -0.6744597, 0.6801798
4: -6.6638317, -5.8151822, -6.6635113, -5.8304858, -0.5124581, 0.5249640
5: -10.9846315, -10.1281900, -10.9791718, -10.1283875, -0.6073871, 0.6015720
6: -13.2090893, -12.0743647, -13.1994457, -12.0746508, -0.6790662, 0.6684990
7: -4.3321466, -3.5239487, -4.3303776, -3.5318575, -0.6257968, 0.6336670
8: -4.2807746, -3.6698232, -4.2767563, -3.6702404, -0.4097366, 0.4054716
9: -10.8555174, -9.7489080, -10.8552294, -9.7669830, -0.6923141, 0.7018280

Time for backsubstitution: 20.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4637
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4637

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3222826
time: 5.51 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3248346, upper bound: 0.3248336
time: 5.99 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.45 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 32.45
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3201287
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 32.45
Output dim: 2, lower bound: -0.3248346, upper bound: 0.3226801
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 32.45
Output dim: 2, lower bound: -0.3248271, upper bound: 0.3222826
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 32.45
Output dim: 2, lower bound: -0.3248346, upper bound: 0.3248336

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -10.5181808, -9.6524010, -10.5213766, -9.6515636, -0.6848598, 0.6877267
1: -4.4963131, -3.7285893, -4.4975128, -3.7263808, -0.5253403, 0.5241683
2: 10.2154713, 10.9070072, 10.2147827, 10.9113884, -0.5125149, 0.5080180
3: -3.8907442, -3.0363779, -3.8942089, -3.0347185, -0.6688833, 0.6704147
4: -6.6535053, -5.8311944, -6.6609020, -5.8306642, -0.5047638, 0.5121880
5: -10.9769993, -10.1304436, -10.9786329, -10.1289225, -0.5977519, 0.5988936
6: -13.1947918, -12.0804615, -13.1982975, -12.0761709, -0.6639433, 0.6632006
7: -4.3252845, -3.5366192, -4.3290586, -3.5330276, -0.6199207, 0.6199605
8: -4.2728310, -3.6727605, -4.2757807, -3.6708946, -0.4014215, 0.4019740
9: -10.8422413, -9.7679768, -10.8518543, -9.7672358, -0.6815805, 0.6914618

Time for backsubstitution: 21.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A1_A1_B1

### Relational analysis result of NS_A1_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226754, upper bound: 0.3201282
time: 7.07 seconds

## Relational analysis of NS_A1_A1_B2

### Relational analysis result of NS_A1_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226754, upper bound: 0.3201287
time: 4.22 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -10.5261660, -9.6510267, -10.5216446, -9.6515608, -0.6976309, 0.6896539
1: -4.4978828, -3.7253094, -4.4978938, -3.7263460, -0.5264494, 0.5279090
2: 10.2080708, 10.9103661, 10.2147017, 10.9121990, -0.5186312, 0.5104818
3: -3.8953209, -3.0340657, -3.8943534, -3.0342045, -0.6736841, 0.6727836
4: -6.6557350, -5.8290758, -6.6610394, -5.8305960, -0.5089617, 0.5144989
5: -10.9854727, -10.1273384, -10.9789038, -10.1289062, -0.6113095, 0.6025929
6: -13.1991558, -12.0736876, -13.1992397, -12.0761318, -0.6671438, 0.6708210
7: -4.3327417, -3.5317960, -4.3292074, -3.5319805, -0.6280684, 0.6243148
8: -4.2756171, -3.6675589, -4.2764330, -3.6708431, -0.4034524, 0.4079697
9: -10.8465042, -9.7642012, -10.8521538, -9.7672005, -0.6901760, 0.6956966

Time for backsubstitution: 21.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A1_A2_B1

### Relational analysis result of NS_A1_A2_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226829, upper bound: 0.3226802
time: 5.62 seconds

## Relational analysis of NS_A1_A2_B2

### Relational analysis result of NS_A1_A2_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226829, upper bound: 0.3226807
time: 4.14 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -10.5232925, -9.6474648, -10.5222044, -9.6512794, -0.6900568, 0.6934276
1: -4.5006862, -3.7253766, -4.4975166, -3.7256393, -0.5305400, 0.5271015
2: 10.2084560, 10.9098101, 10.2146263, 10.9120445, -0.5199438, 0.5128284
3: -3.8963876, -3.0297027, -3.8952718, -3.0346975, -0.6737328, 0.6780183
4: -6.6632948, -5.8154626, -6.6633730, -5.8305526, -0.5114291, 0.5244558
5: -10.9834576, -10.1282578, -10.9788980, -10.1284008, -0.6048636, 0.6009183
6: -13.2052021, -12.0745277, -13.1984978, -12.0746956, -0.6751268, 0.6674056
7: -4.3315516, -3.5282660, -4.3302274, -3.5329123, -0.6246367, 0.6295276
8: -4.2780943, -3.6700377, -4.2761011, -3.6702917, -0.4070537, 0.4046243
9: -10.8543186, -9.7490530, -10.8549280, -9.7670183, -0.6901755, 0.7010498

Time for backsubstitution: 21.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226732, upper bound: 0.3222824
time: 6.02 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226732, upper bound: 0.3222834
time: 4.53 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5224714, -9.6512756, -0.7028286, 0.6953521
1: -4.5022573, -3.7220972, -4.4978981, -3.7256038, -0.5316503, 0.5308394
2: 10.2010536, 10.9131718, 10.2145462, 10.9128542, -0.5224577, 0.5152941
3: -3.9009418, -3.0273905, -3.8954148, -3.0341835, -0.6785283, 0.6803942
4: -6.6655350, -5.8133445, -6.6635094, -5.8304849, -0.5156338, 0.5253717
5: -10.9918747, -10.1251602, -10.9791698, -10.1283855, -0.6183403, 0.6046162
6: -13.2095709, -12.0677557, -13.1994371, -12.0746527, -0.6783552, 0.6750271
7: -4.3389769, -3.5234466, -4.3303766, -3.5318642, -0.6327748, 0.6338704
8: -4.2808857, -3.6648326, -4.2767525, -3.6702409, -0.4090910, 0.4106201
9: -10.8586035, -9.7452812, -10.8552256, -9.7669821, -0.6987810, 0.7020807

Time for backsubstitution: 22.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4653
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4637
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 4653

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248339
time: 7.61 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248345
time: 4.03 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 33.93 seconds
NS_A1_A1_B1, status: Status.VERIFIED, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226754, upper bound: 0.3201282
NS_A1_A1_B2, status: Status.VERIFIED, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226754, upper bound: 0.3201287
NS_A1_A2_B1, status: Status.VERIFIED, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226829, upper bound: 0.3226802
NS_A1_A2_B2, status: Status.VERIFIED, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226829, upper bound: 0.3226807
NS_A2_A1_B1, status: Status.VERIFIED, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226732, upper bound: 0.3222824
NS_A2_A1_B2, status: Status.VERIFIED, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226732, upper bound: 0.3222834
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248339
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 33.93
Output dim: 2, lower bound: -0.3226808, upper bound: 0.3248345

## BFS NS instance: NS_A2_A2_B1

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

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4615
type: B, layer: 1, pos: 4637
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3226177, upper bound: 0.3224669
time: 6.21 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3226761, upper bound: 0.3248283
time: 5.16 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5244017, -9.6474552, -0.7063472, 0.6973646
1: -4.5022573, -3.7220972, -4.5022564, -3.7252326, -0.5271993, 0.5309079
2: 10.2010536, 10.9131718, 10.2081203, 10.9131527, -0.5233399, 0.5175283
3: -3.9009418, -3.0273905, -3.8969669, -3.0275850, -0.6814644, 0.6788323
4: -6.6655350, -5.8133445, -6.6638303, -5.8151822, -0.5175953, 0.5162988
5: -10.9918747, -10.1251602, -10.9846296, -10.1281900, -0.6164889, 0.6085806
6: -13.2095709, -12.0677557, -13.2090778, -12.0743647, -0.6701336, 0.6773913
7: -4.3389769, -3.5234466, -4.3321466, -3.5239553, -0.6354425, 0.6316624
8: -4.2808857, -3.6648326, -4.2807713, -3.6698239, -0.4086614, 0.4126599
9: -10.8586035, -9.7452812, -10.8555155, -9.7489109, -0.7018166, 0.6990399

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4615

## Relational analysis of NS_A2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3203146, upper bound: 0.3247708
time: 6.26 seconds

## Relational analysis of NS_A2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3226756, upper bound: 0.3248298
time: 5.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 34.02 seconds
NS_A2_A2_B1_A1, status: Status.VERIFIED, split count: 4, time: 34.02
Output dim: 2, lower bound: -0.3226177, upper bound: 0.3224669
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.02
Output dim: 2, lower bound: -0.3226761, upper bound: 0.3248283
NS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 34.02
Output dim: 2, lower bound: -0.3203146, upper bound: 0.3247708
NS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 34.02
Output dim: 2, lower bound: -0.3226756, upper bound: 0.3248298

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5193071, -9.6523914, -0.7050958, 0.6914215
1: -4.5022559, -3.7220979, -4.4978833, -3.7284431, -0.5254626, 0.5306728
2: 10.2010574, 10.9131689, 10.2151442, 10.9103498, -0.5169894, 0.5062480
3: -3.9009371, -3.0273914, -3.8913441, -3.0342607, -0.6753173, 0.6743195
4: -6.6655350, -5.8133478, -6.6540518, -5.8309135, -0.5184209, 0.5095874
5: -10.9918690, -10.1251602, -10.9781380, -10.1303730, -0.6130462, 0.6039412
6: -13.2095699, -12.0677576, -13.1986637, -12.0802937, -0.6720569, 0.6763525
7: -4.3389754, -3.5234509, -4.3259015, -3.5323076, -0.6303267, 0.6228068
8: -4.2808824, -3.6648331, -4.2755046, -3.6725440, -0.4049628, 0.4089314
9: -10.8586035, -9.7452850, -10.8434563, -9.7678318, -0.6980221, 0.6864648

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248212
time: 4.04 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248286
time: 7.76 seconds

## BFS NS instance: NS_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -10.5309620, -9.6462593, -10.5228148, -9.6482849, -0.7044172, 0.6949830
1: -4.5002556, -3.7225223, -4.4958067, -3.7287738, -0.5200057, 0.5240097
2: 10.2012529, 10.9109497, 10.2107210, 10.9059906, -0.5159386, 0.5123992
3: -3.8990812, -3.0274591, -3.8909159, -3.0293932, -0.6758189, 0.6731989
4: -6.6655097, -5.8159580, -6.6614981, -5.8238239, -0.5089905, 0.5114136
5: -10.9903126, -10.1251841, -10.9793673, -10.1296911, -0.6132941, 0.6037223
6: -13.2092991, -12.0679417, -13.2076025, -12.0749598, -0.6692109, 0.6756530
7: -4.3387136, -3.5270996, -4.3282089, -3.5358634, -0.6234031, 0.6246450
8: -4.2794986, -3.6649067, -4.2761726, -3.6713381, -0.4057903, 0.4079697
9: -10.8585510, -9.7471066, -10.8537197, -9.7548780, -0.6958361, 0.6952016

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: A, layer: 1, pos: 6224
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A2_A2_B2_B1_B1

### Relational analysis result of NS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3247638
time: 6.18 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2

### Relational analysis result of NS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3247709
time: 6.62 seconds

## BFS NS instance: NS_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5244017, -9.6474552, -0.7054939, 0.7003577
1: -4.5022573, -3.7220972, -4.5022535, -3.7252336, -0.5266685, 0.5279112
2: 10.2010536, 10.9131718, 10.2081232, 10.9131508, -0.5167928, 0.5174775
3: -3.9009418, -3.0273905, -3.8969636, -3.0275850, -0.6797614, 0.6754260
4: -6.6655350, -5.8133445, -6.6638298, -5.8151860, -0.5115395, 0.5162985
5: -10.9918747, -10.1251602, -10.9846249, -10.1281900, -0.6164894, 0.6053069
6: -13.2095709, -12.0677557, -13.2090807, -12.0743618, -0.6710291, 0.6771915
7: -4.3389769, -3.5234466, -4.3321457, -3.5239611, -0.6289680, 0.6316628
8: -4.2808857, -3.6648326, -4.2807693, -3.6698246, -0.4086609, 0.4108365
9: -10.8586035, -9.7452812, -10.8555155, -9.7489138, -0.6982322, 0.6990395

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4637
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4637

## Relational analysis of NS_A2_A2_B2_B2_B1

### Relational analysis result of NS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201230, upper bound: 0.3248214
time: 6.75 seconds

## Relational analysis of NS_A2_A2_B2_B2_B2

### Relational analysis result of NS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3201230, upper bound: 0.3248300
time: 5.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 34.93 seconds
NS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 34.93
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248212
NS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 34.93
Output dim: 2, lower bound: -0.3201236, upper bound: 0.3248286
NS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 34.93
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3247638
NS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 34.93
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3247709
NS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 34.93
Output dim: 2, lower bound: -0.3201230, upper bound: 0.3248214
NS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 34.93
Output dim: 2, lower bound: -0.3201230, upper bound: 0.3248300

## BFS NS instance: NS_A2_A2_B1_A2_B1

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

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4615

## Relational analysis of NS_A2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3247627
time: 3.78 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3248212
time: 5.52 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2

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

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4615
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 4546
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4615

## Relational analysis of NS_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3247708
time: 4.37 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3247711
time: 4.15 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -10.5309620, -9.6462593, -10.5217056, -9.6482954, -0.7009196, 0.6927488
1: -4.5002556, -3.7225223, -4.4942360, -3.7289176, -0.5203605, 0.5224359
2: 10.2012529, 10.9109497, 10.2110586, 10.9026489, -0.5125668, 0.5120409
3: -3.8990812, -3.0274591, -3.8903408, -3.0315113, -0.6736994, 0.6730440
4: -6.6655097, -5.8159580, -6.6609631, -5.8241038, -0.5071039, 0.5104318
5: -10.9903126, -10.1251841, -10.9781942, -10.1297607, -0.6095917, 0.6012130
6: -13.2092991, -12.0679417, -13.2037220, -12.0751286, -0.6702404, 0.6717541
7: -4.3387136, -3.5270996, -4.3276043, -3.5401745, -0.6193008, 0.6244569
8: -4.2794986, -3.6649067, -4.2734966, -3.6715505, -0.4064507, 0.4053319
9: -10.8585510, -9.7471066, -10.8525229, -9.7550240, -0.6921916, 0.6927457

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A2_A2_B2_B1_B1_A1

### Relational analysis result of NS_A2_A2_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3224599
time: 6.54 seconds

## Relational analysis of NS_A2_A2_B2_B1_B1_A2

### Relational analysis result of NS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3247638
time: 6.02 seconds

## BFS NS instance: NS_A2_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -10.5309620, -9.6462593, -10.5297098, -9.6469202, -0.7044222, 0.7044711
1: -4.5002556, -3.7225223, -4.4958072, -3.7256370, -0.5208607, 0.5211551
2: 10.2012529, 10.9109497, 10.2036533, 10.9060087, -0.5129536, 0.5135891
3: -3.8990812, -3.0274591, -3.8948960, -3.0291934, -0.6760626, 0.6765943
4: -6.6655097, -5.8159580, -6.6632023, -5.8219895, -0.5095830, 0.5133061
5: -10.9903126, -10.1251841, -10.9865942, -10.1266603, -0.6137354, 0.6120510
6: -13.2092991, -12.0679417, -13.2081022, -12.0683546, -0.6702294, 0.6694188
7: -4.3387136, -3.5270996, -4.3350434, -3.5353551, -0.6220212, 0.6251748
8: -4.2794986, -3.6649067, -4.2762871, -3.6663435, -0.4068654, 0.4050474
9: -10.8585510, -9.7471066, -10.8568077, -9.7512512, -0.6964145, 0.6990552

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: A, layer: 1, pos: 4650
type: B, layer: 1, pos: 4650
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A2_A2_B2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3224671
time: 7.13 seconds

## Relational analysis of NS_A2_A2_B2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.3177623, upper bound: 0.3247710
time: 6.51 seconds

## BFS NS instance: NS_A2_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -10.5312920, -9.6460924, -10.5232916, -9.6474648, -0.7021015, 0.6981294
1: -4.5022573, -3.7220972, -4.5006847, -3.7253771, -0.5270226, 0.5263381
2: 10.2010536, 10.9131718, 10.2084551, 10.9098072, -0.5134208, 0.5177076
3: -3.9009418, -3.0273905, -3.8963833, -3.0297041, -0.6776502, 0.6752725
4: -6.6655350, -5.8133445, -6.6632967, -5.8154664, -0.5096486, 0.5153165
5: -10.9918747, -10.1251602, -10.9834518, -10.1282587, -0.6127882, 0.6027925
6: -13.2095709, -12.0677557, -13.2051992, -12.0745316, -0.6720595, 0.6732924
7: -4.3389769, -3.5234466, -4.3315511, -3.5282722, -0.6248672, 0.6323900
8: -4.2808857, -3.6648326, -4.2780933, -3.6700358, -0.4093242, 0.4081986
9: -10.8586035, -9.7452812, -10.8543177, -9.7490559, -0.6945853, 0.6969233

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4615
type: B, layer: 1, pos: 4546
type: A, layer: 1, pos: 4546
type: B, layer: 1, pos: 6224
type: A, layer: 1, pos: 6224
type: B, layer: 1, pos: 4650
type: A, layer: 1, pos: 4650
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 4615

## Relational analysis of NS_A2_A2_B2_B2_B1_A1

### Relational analysis result of NS_A2_A2_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3224599
time: 4.81 seconds

## Relational analysis of NS_A2_A2_B2_B2_B1_A2

### Relational analysis result of NS_A2_A2_B2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 2, lower bound: -0.3177617, upper bound: 0.3224598
time: 6.24 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.00 + 546.61 = 604.61 seconds
