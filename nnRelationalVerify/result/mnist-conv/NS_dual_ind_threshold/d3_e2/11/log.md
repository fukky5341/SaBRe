## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.41043252599999996


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-5.9706378, -4.5664105, -5.9706378, -4.5664105, -0.8283465, 0.8283467)
1: (-7.9303246, -6.6111803, -7.9303246, -6.6111803, -0.7843864, 0.7843866)
2: (-4.4578519, -3.3069019, -4.4578519, -3.3069019, -0.7637737, 0.7637737)
3: (-6.0369263, -4.5567064, -6.0369263, -4.5567064, -0.9626312, 0.9626315)
4: (-12.2408857, -10.3860044, -12.2408857, -10.3860044, -0.9123442, 0.9123440)
5: (-6.6909637, -5.6331291, -6.6909637, -5.6331291, -0.5031065, 0.5031066)
6: (-5.5420136, -4.3639669, -5.5420136, -4.3639669, -0.7319636, 0.7319636)
7: (-11.0493546, -9.8017998, -11.0493546, -9.8017998, -0.8130295, 0.8130298)
8: (9.8868074, 10.8243589, 9.8868074, 10.8243589, -0.6571708, 0.6571705)
9: (-7.5092068, -5.9313560, -7.5092068, -5.9313560, -0.9103527, 0.9103529)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.05 + 34.23 = 58.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4188075, upper bound: 0.4188087

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4611
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4611

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163101, upper bound: 0.4183016
time: 3.34 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188022, upper bound: 0.4188038
time: 3.16 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 6.80 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 6.80
Output dim: 8, lower bound: -0.4163101, upper bound: 0.4183016
NS_A2, status: Status.UNKNOWN, split count: 1, time: 6.80
Output dim: 8, lower bound: -0.4188022, upper bound: 0.4188038

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -5.9572535, -4.5665827, -5.9667625, -4.5664587, -0.8145206, 0.8237352
1: -7.9258809, -6.6168985, -7.9290781, -6.6128144, -0.7701721, 0.7746159
2: -4.4499092, -3.3082843, -4.4555769, -3.3073144, -0.7540004, 0.7595217
3: -6.0330505, -4.5658574, -6.0358515, -4.5593252, -0.9512863, 0.9490108
4: -12.2343588, -10.3876677, -12.2390099, -10.3864689, -0.9042583, 0.9083505
5: -6.6899819, -5.6392822, -6.6906910, -5.6348825, -0.4988369, 0.4951386
6: -5.5368838, -4.3732109, -5.5406055, -4.3666019, -0.7236900, 0.7209809
7: -11.0439234, -9.8036060, -11.0478096, -9.8023033, -0.8046732, 0.8073308
8: 9.8934946, 10.8220596, 9.8887262, 10.8237143, -0.6489055, 0.6522431
9: -7.5079975, -5.9405684, -7.5088615, -5.9339838, -0.9045565, 0.8993320

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163082, upper bound: 0.4172119
time: 3.24 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163082, upper bound: 0.4182996
time: 4.98 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -5.9802866, -4.5620289, -5.9706230, -4.5664110, -0.8462999, 0.8321300
1: -7.9359174, -6.6100378, -7.9303207, -6.6111851, -0.7851286, 0.7844003
2: -4.4597855, -3.3048768, -4.4578457, -3.3069038, -0.7648938, 0.7666106
3: -6.0407534, -4.5536809, -6.0369234, -4.5567169, -0.9627242, 0.9685762
4: -12.2438507, -10.3794670, -12.2408800, -10.3860064, -0.9126031, 0.9187860
5: -6.6980124, -5.6321039, -6.6909647, -5.6331367, -0.5089904, 0.5034107
6: -5.5545154, -4.3627801, -5.5420089, -4.3639770, -0.7418883, 0.7307615
7: -11.0508938, -9.7967606, -11.0493507, -9.8018007, -0.8134208, 0.8165834
8: 9.8847666, 10.8270826, 9.8868141, 10.8243570, -0.6586862, 0.6595950
9: -7.5150871, -5.9291000, -7.5092077, -5.9313622, -0.9157238, 0.9101479

Time for backsubstitution: 22.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188004, upper bound: 0.4177128
time: 5.11 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188004, upper bound: 0.4188006
time: 4.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 32.14 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 32.14
Output dim: 8, lower bound: -0.4163082, upper bound: 0.4172119
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 32.14
Output dim: 8, lower bound: -0.4163082, upper bound: 0.4182996
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 32.14
Output dim: 8, lower bound: -0.4188004, upper bound: 0.4177128
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 32.14
Output dim: 8, lower bound: -0.4188004, upper bound: 0.4188006

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -5.9567032, -4.5666013, -5.9650397, -4.5665188, -0.8135755, 0.8217437
1: -7.9255786, -6.6169977, -7.9281230, -6.6130800, -0.7695811, 0.7732778
2: -4.4495816, -3.3082876, -4.4545555, -3.3073230, -0.7534108, 0.7581363
3: -6.0329118, -4.5659833, -6.0354233, -4.5597219, -0.9508781, 0.9484441
4: -12.2342854, -10.3877916, -12.2387819, -10.3868589, -0.9029021, 0.9069660
5: -6.6896434, -5.6392851, -6.6896224, -5.6348844, -0.4983181, 0.4939387
6: -5.5367413, -4.3737235, -5.5401592, -4.3681989, -0.7220025, 0.7202749
7: -11.0436735, -9.8036146, -11.0470600, -9.8023233, -0.8035846, 0.8055332
8: 9.8941441, 10.8219604, 9.8907585, 10.8234072, -0.6479421, 0.6500945
9: -7.5078139, -5.9406424, -7.5083032, -5.9342165, -0.9040868, 0.8986609

Time for backsubstitution: 22.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172112
time: 4.00 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172101
time: 4.72 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -5.9572506, -4.5665798, -5.9670334, -4.5650334, -0.8156688, 0.8234992
1: -7.9258819, -6.6168976, -7.9293647, -6.6120901, -0.7710094, 0.7748309
2: -4.4499054, -3.3082850, -4.4560680, -3.3063583, -0.7548149, 0.7600496
3: -6.0330515, -4.5658579, -6.0359197, -4.5578952, -0.9530091, 0.9490693
4: -12.2343550, -10.3876696, -12.2394838, -10.3861179, -0.9052577, 0.9081287
5: -6.6899824, -5.6392822, -6.6908164, -5.6338677, -0.4996994, 0.4949856
6: -5.5368810, -4.3732128, -5.5425501, -4.3664045, -0.7235708, 0.7229800
7: -11.0439224, -9.8036070, -11.0484610, -9.8017941, -0.8046730, 0.8086267
8: 9.8934937, 10.8220587, 9.8886070, 10.8263664, -0.6515570, 0.6515450
9: -7.5079947, -5.9405670, -7.5090251, -5.9332757, -0.9052718, 0.8994031

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4182984
time: 4.91 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4182986
time: 5.47 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -5.9797363, -4.5620484, -5.9688973, -4.5664725, -0.8453581, 0.8301401
1: -7.9356122, -6.6101351, -7.9293637, -6.6114535, -0.7845416, 0.7830620
2: -4.4594574, -3.3048801, -4.4568219, -3.3069134, -0.7643032, 0.7652268
3: -6.0406156, -4.5538077, -6.0364943, -4.5571122, -0.9623179, 0.9680066
4: -12.2437801, -10.3795919, -12.2406511, -10.3863964, -0.9112477, 0.9174023
5: -6.6976728, -5.6321044, -6.6898975, -5.6331387, -0.5084716, 0.5022092
6: -5.5543761, -4.3632946, -5.5415659, -4.3655753, -0.7402043, 0.7300558
7: -11.0506420, -9.7967720, -11.0485964, -9.8018236, -0.8123312, 0.8147860
8: 9.8854160, 10.8269854, 9.8888454, 10.8240509, -0.6577239, 0.6574502
9: -7.5149069, -5.9291754, -7.5086465, -5.9315977, -0.9152560, 0.9094753

Time for backsubstitution: 22.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4177133
time: 3.41 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4177135
time: 3.31 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -5.9802842, -4.5620284, -5.9708948, -4.5649862, -0.8474483, 0.8318934
1: -7.9359169, -6.6100349, -7.9306078, -6.6104636, -0.7859664, 0.7846158
2: -4.4597821, -3.3048770, -4.4583364, -3.3059480, -0.7657092, 0.7671418
3: -6.0407529, -4.5536804, -6.0369930, -4.5552855, -0.9644465, 0.9686348
4: -12.2438526, -10.3794699, -12.2413530, -10.3856535, -0.9136026, 0.9185641
5: -6.6980133, -5.6321039, -6.6910896, -5.6321220, -0.5090250, 0.5032573
6: -5.5545168, -4.3627839, -5.5439548, -4.3637791, -0.7417781, 0.7327654
7: -11.0508947, -9.7967644, -11.0500011, -9.8012943, -0.8134203, 0.8178778
8: 9.8847675, 10.8270826, 9.8866949, 10.8270082, -0.6613350, 0.6588967
9: -7.5150867, -5.9291010, -7.5093708, -5.9306545, -0.9164183, 0.9102187

Time for backsubstitution: 22.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4188007
time: 6.16 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4188019
time: 3.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.37 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172112
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172101
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4182984
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4182986
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4177133
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4177135
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4188007
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.37
Output dim: 8, lower bound: -0.4177120, upper bound: 0.4188019

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -5.9555340, -4.5666399, -5.9650397, -4.5665188, -0.8122909, 0.8215113
1: -7.9249263, -6.6171651, -7.9281230, -6.6130800, -0.7687142, 0.7731541
2: -4.4488869, -3.3082948, -4.4545555, -3.3073230, -0.7525001, 0.7580254
3: -6.0326247, -4.5662522, -6.0354233, -4.5597219, -0.9505482, 0.9482713
4: -12.2341299, -10.3880615, -12.2387819, -10.3868589, -0.9021826, 0.9062722
5: -6.6889153, -5.6392851, -6.6896224, -5.6348844, -0.4975339, 0.4938357
6: -5.5364347, -4.3748083, -5.5401592, -4.3681989, -0.7218752, 0.7191665
7: -11.0431757, -9.8036299, -11.0470600, -9.8023233, -0.8024814, 0.8051431
8: 9.8955288, 10.8217468, 9.8907585, 10.8234072, -0.6465459, 0.6498818
9: -7.5074358, -5.9407997, -7.5083032, -5.9342165, -0.9036906, 0.8984718

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4152194
time: 3.17 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172108
time: 4.70 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -5.9575233, -4.5651550, -5.9650397, -4.5665188, -0.8142617, 0.8230040
1: -7.9261661, -6.6161766, -7.9281230, -6.6130800, -0.7701974, 0.7740555
2: -4.4504023, -3.3073304, -4.4545555, -3.3073230, -0.7541194, 0.7590046
3: -6.0331163, -4.5644302, -6.0354233, -4.5597219, -0.9510937, 0.9502470
4: -12.2348328, -10.3873138, -12.2387819, -10.3868589, -0.9030190, 0.9068305
5: -6.6901088, -5.6382689, -6.6896224, -5.6348844, -0.4986564, 0.4948504
6: -5.5388284, -4.3730149, -5.5401592, -4.3681989, -0.7240615, 0.7210855
7: -11.0445719, -9.8030977, -11.0470600, -9.8023233, -0.8041122, 0.8057182
8: 9.8933773, 10.8247147, 9.8907585, 10.8234072, -0.6487374, 0.6528525
9: -7.5081577, -5.9398594, -7.5083032, -5.9342165, -0.9044783, 0.8994668

Time for backsubstitution: 22.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4152194
time: 3.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172119
time: 4.20 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -5.9555340, -4.5666399, -5.9670334, -4.5650334, -0.8137834, 0.8234797
1: -7.9249263, -6.6171651, -7.9293647, -6.6120901, -0.7696157, 0.7746466
2: -4.4488869, -3.3082948, -4.4560680, -3.3063583, -0.7534788, 0.7596543
3: -6.0326247, -4.5662522, -6.0359197, -4.5578952, -0.9525237, 0.9488206
4: -12.2341299, -10.3880615, -12.2394838, -10.3861179, -0.9027381, 0.9071088
5: -6.6889153, -5.6392851, -6.6908164, -5.6338677, -0.4985492, 0.4949603
6: -5.5364347, -4.3748083, -5.5425501, -4.3664045, -0.7237949, 0.7213531
7: -11.0431757, -9.8036299, -11.0484610, -9.8017941, -0.8030565, 0.8067789
8: 9.8955288, 10.8217468, 9.8886070, 10.8263664, -0.6495085, 0.6520755
9: -7.5074358, -5.9407997, -7.5090251, -5.9332757, -0.9046872, 0.8992586

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4163080
time: 3.48 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4182985
time: 4.65 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -5.9575233, -4.5651550, -5.9670334, -4.5650334, -0.8143606, 0.8235726
1: -7.9261661, -6.6161766, -7.9293647, -6.6120901, -0.7712944, 0.7757432
2: -4.4504023, -3.3073304, -4.4560680, -3.3063583, -0.7545192, 0.7600539
3: -6.0331163, -4.5644302, -6.0359197, -4.5578952, -0.9530606, 0.9507878
4: -12.2348328, -10.3873138, -12.2394838, -10.3861179, -0.9058385, 0.9099324
5: -6.6901088, -5.6382689, -6.6908164, -5.6338677, -0.4986836, 0.4949856
6: -5.5388284, -4.3730149, -5.5425501, -4.3664045, -0.7246172, 0.7219079
7: -11.0445719, -9.8030977, -11.0484610, -9.8017941, -0.8059804, 0.8086364
8: 9.8933773, 10.8247147, 9.8886070, 10.8263664, -0.6488502, 0.6521952
9: -7.5081577, -5.9398594, -7.5090251, -5.9332757, -0.9050338, 0.8998120

Time for backsubstitution: 22.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4152175
time: 4.54 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172101
time: 5.47 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9785600, -4.5620890, -5.9688973, -4.5664725, -0.8440812, 0.8299079
1: -7.9349637, -6.6103029, -7.9293637, -6.6114535, -0.7836831, 0.7829382
2: -4.4587622, -3.3048873, -4.4568219, -3.3069134, -0.7633910, 0.7651165
3: -6.0403290, -4.5540791, -6.0364943, -4.5571122, -0.9619908, 0.9678307
4: -12.2436247, -10.3798590, -12.2406511, -10.3863964, -0.9105277, 0.9167109
5: -6.6969471, -5.6321058, -6.6898975, -5.6331387, -0.5076884, 0.5021062
6: -5.5540719, -4.3643794, -5.5415659, -4.3655753, -0.7400944, 0.7289472
7: -11.0501375, -9.7967873, -11.0485964, -9.8018236, -0.8112288, 0.8143957
8: 9.8867979, 10.8267756, 9.8888454, 10.8240509, -0.6563296, 0.6572421
9: -7.5145292, -5.9293370, -7.5086465, -5.9315977, -0.9148612, 0.9092846

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4152194
time: 4.82 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4152194
time: 3.33 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -5.9805532, -4.5606027, -5.9688973, -4.5664725, -0.8460364, 0.8314006
1: -7.9362011, -6.6093140, -7.9293637, -6.6114535, -0.7851551, 0.7838397
2: -4.4602833, -3.3039222, -4.4568219, -3.3069134, -0.7650256, 0.7660956
3: -6.0408211, -4.5522475, -6.0364943, -4.5571122, -0.9625335, 0.9697921
4: -12.2443275, -10.3791208, -12.2406511, -10.3863964, -0.9113641, 0.9172649
5: -6.6981382, -5.6310897, -6.6898975, -5.6331387, -0.5085800, 0.5031211
6: -5.5564580, -4.3625832, -5.5415659, -4.3655753, -0.7406013, 0.7308681
7: -11.0515442, -9.7962561, -11.0485964, -9.8018236, -0.8128660, 0.8149700
8: 9.8846445, 10.8297348, 9.8888454, 10.8240509, -0.6585259, 0.6602025
9: -7.5152535, -5.9283910, -7.5086465, -5.9315977, -0.9156489, 0.9102843

Time for backsubstitution: 22.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4152194
time: 3.53 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4152194
time: 3.36 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9785600, -4.5620890, -5.9708948, -4.5649862, -0.8455741, 0.8318760
1: -7.9349637, -6.6103029, -7.9306078, -6.6104636, -0.7845845, 0.7844316
2: -4.4587622, -3.3048873, -4.4583364, -3.3059480, -0.7643712, 0.7667472
3: -6.0403290, -4.5540791, -6.0369930, -4.5552855, -0.9639673, 0.9683805
4: -12.2436247, -10.3798590, -12.2413530, -10.3856535, -0.9110830, 0.9175479
5: -6.6969471, -5.6321058, -6.6910896, -5.6321220, -0.5078753, 0.5032325
6: -5.5540719, -4.3643794, -5.5439548, -4.3637791, -0.7413344, 0.7311378
7: -11.0501375, -9.7967873, -11.0500011, -9.8012943, -0.8118041, 0.8160334
8: 9.8867979, 10.8267756, 9.8866949, 10.8270082, -0.6592894, 0.6594346
9: -7.5145292, -5.9293370, -7.5093708, -5.9306545, -0.9158275, 0.9100721

Time for backsubstitution: 22.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4163067
time: 4.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4174965
time: 5.41 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -5.9805532, -4.5606027, -5.9708948, -4.5649862, -0.8461325, 0.8319674
1: -7.9362011, -6.6093140, -7.9306078, -6.6104636, -0.7862520, 0.7855285
2: -4.4602833, -3.3039222, -4.4583364, -3.3059480, -0.7654274, 0.7671463
3: -6.0408211, -4.5522475, -6.0369930, -4.5552855, -0.9645019, 0.9703338
4: -12.2443275, -10.3791208, -12.2413530, -10.3856535, -0.9141843, 0.9203677
5: -6.6981382, -5.6310897, -6.6910896, -5.6321220, -0.5088251, 0.5032576
6: -5.5564580, -4.3625832, -5.5439548, -4.3637791, -0.7425282, 0.7316947
7: -11.0515442, -9.7962561, -11.0500011, -9.8012943, -0.8147247, 0.8178871
8: 9.8846445, 10.8297348, 9.8866949, 10.8270082, -0.6586351, 0.6595435
9: -7.5152535, -5.9283910, -7.5093708, -5.9306545, -0.9162028, 0.9106302

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4152188
time: 3.29 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4164098
time: 4.73 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.88 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4152194
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172108
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4152194
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172119
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4163080
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4182985
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4152175
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4152188, upper bound: 0.4172101
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4152194
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4152194
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4152194
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4152194
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4163067
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4174965
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172098, upper bound: 0.4152188
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 8, lower bound: -0.4172101, upper bound: 0.4164098

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9555340, -4.5666399, -5.9555340, -4.5666399, -0.8117042, 0.8117042
1: -7.9249263, -6.6171651, -7.9249263, -6.6171651, -0.7642491, 0.7642492
2: -4.4488869, -3.3082948, -4.4488869, -3.3082948, -0.7514205, 0.7514207
3: -6.0326247, -4.5662522, -6.0326247, -4.5662522, -0.9423971, 0.9423974
4: -12.2341299, -10.3880615, -12.2341299, -10.3880615, -0.9008589, 0.9008586
5: -6.6889153, -5.6392851, -6.6889153, -5.6392851, -0.4922557, 0.4922557
6: -5.5364347, -4.3748083, -5.5364347, -4.3748083, -0.7151754, 0.7151754
7: -11.0431757, -9.8036299, -11.0431757, -9.8036299, -0.7999089, 0.7999089
8: 9.8955288, 10.8217468, 9.8955288, 10.8217468, -0.6445360, 0.6445365
9: -7.5074358, -5.9407997, -7.5074358, -5.9407997, -0.8964119, 0.8964117

Time for backsubstitution: 22.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4146596, upper bound: 0.4152151
time: 5.06 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152161, upper bound: 0.4152164
time: 3.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9555340, -4.5666399, -5.9785600, -4.5620890, -0.8163188, 0.8372238
1: -7.9249263, -6.6171651, -7.9349637, -6.6103029, -0.7708068, 0.7774833
2: -4.4488869, -3.3082948, -4.4587622, -3.3048873, -0.7557266, 0.7618506
3: -6.0326247, -4.5662522, -6.0403290, -4.5540791, -0.9560223, 0.9507143
4: -12.2341299, -10.3880615, -12.2436247, -10.3798590, -0.9091496, 0.9103434
5: -6.6889153, -5.6392851, -6.6969471, -5.6321058, -0.4993737, 0.5003628
6: -5.5364347, -4.3748083, -5.5540719, -4.3643794, -0.7258754, 0.7306225
7: -11.0431757, -9.8036299, -11.0501375, -9.7967873, -0.8070967, 0.8083491
8: 9.8955288, 10.8217468, 9.8867979, 10.8267756, -0.6497629, 0.6538668
9: -7.5074358, -5.9407997, -7.5145292, -5.9293370, -0.9081845, 0.9046812

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4146596, upper bound: 0.4172078
time: 4.26 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152161, upper bound: 0.4172089
time: 4.00 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9575233, -4.5651550, -5.9555340, -4.5666399, -0.8136749, 0.8131969
1: -7.9261661, -6.6161766, -7.9249263, -6.6171651, -0.7657325, 0.7651504
2: -4.4504023, -3.3073304, -4.4488869, -3.3082948, -0.7530403, 0.7523999
3: -6.0331163, -4.5644302, -6.0326247, -4.5662522, -0.9429426, 0.9443731
4: -12.2348328, -10.3873138, -12.2341299, -10.3880615, -0.9016953, 0.9014170
5: -6.6901088, -5.6382689, -6.6889153, -5.6392851, -0.4933782, 0.4932704
6: -5.5388284, -4.3730149, -5.5364347, -4.3748083, -0.7173615, 0.7170944
7: -11.0445719, -9.8030977, -11.0431757, -9.8036299, -0.8015397, 0.8004839
8: 9.8933773, 10.8247147, 9.8955288, 10.8217468, -0.6467280, 0.6475070
9: -7.5081577, -5.9398594, -7.5074358, -5.9407997, -0.8971992, 0.8974066

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4157483, upper bound: 0.4152164
time: 3.48 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163046, upper bound: 0.4152164
time: 3.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9575233, -4.5651550, -5.9785600, -4.5620890, -0.8182895, 0.8387167
1: -7.9261661, -6.6161766, -7.9349637, -6.6103029, -0.7722902, 0.7783846
2: -4.4504023, -3.3073304, -4.4587622, -3.3048873, -0.7573459, 0.7628300
3: -6.0331163, -4.5644302, -6.0403290, -4.5540791, -0.9565678, 0.9526901
4: -12.2348328, -10.3873138, -12.2436247, -10.3798590, -0.9099860, 0.9109018
5: -6.6901088, -5.6382689, -6.6969471, -5.6321058, -0.5004959, 0.5005280
6: -5.5388284, -4.3730149, -5.5540719, -4.3643794, -0.7280614, 0.7318622
7: -11.0445719, -9.8030977, -11.0501375, -9.7967873, -0.8087275, 0.8089242
8: 9.8933773, 10.8247147, 9.8867979, 10.8267756, -0.6519544, 0.6568375
9: -7.5081577, -5.9398594, -7.5145292, -5.9293370, -0.9089718, 0.9056189

Time for backsubstitution: 22.16 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 58.28 + 559.37 = 617.65 seconds
