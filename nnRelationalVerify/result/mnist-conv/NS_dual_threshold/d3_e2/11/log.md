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
execution time: IAR + RelationalAnalysis = 23.79 + 33.31 = 57.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.4188075, upper bound: 0.4188087

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4611
type: A, layer: 1, pos: 4611
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4611

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4183004, upper bound: 0.4163113
time: 3.76 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188026, upper bound: 0.4188034
time: 3.27 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.31 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 7.31
Output dim: 8, lower bound: -0.4183004, upper bound: 0.4163113
NS_B2, status: Status.UNKNOWN, split count: 1, time: 7.31
Output dim: 8, lower bound: -0.4188026, upper bound: 0.4188034

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -5.9667625, -4.5664587, -5.9572535, -4.5665827, -0.8237350, 0.8145208
1: -7.9290781, -6.6128144, -7.9258809, -6.6168985, -0.7746158, 0.7701721
2: -4.4555769, -3.3073144, -4.4499092, -3.3082843, -0.7595215, 0.7540004
3: -6.0358515, -4.5593252, -6.0330505, -4.5658574, -0.9490108, 0.9512861
4: -12.2390099, -10.3864689, -12.2343588, -10.3876677, -0.9083502, 0.9042583
5: -6.6906910, -5.6348825, -6.6899819, -5.6392822, -0.4951386, 0.4988369
6: -5.5406055, -4.3666019, -5.5368838, -4.3732109, -0.7209809, 0.7236900
7: -11.0478096, -9.8023033, -11.0439234, -9.8036060, -0.8073311, 0.8046732
8: 9.8887262, 10.8237143, 9.8934946, 10.8220596, -0.6522431, 0.6489058
9: -7.5088615, -5.9339838, -7.5079975, -5.9405684, -0.8993320, 0.9045565

Time for backsubstitution: 21.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4611
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4611

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163089, upper bound: 0.4163101
time: 3.10 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163089, upper bound: 0.4163101
time: 3.25 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -5.9706230, -4.5664110, -5.9802866, -4.5620289, -0.8321300, 0.8462996
1: -7.9303207, -6.6111851, -7.9359174, -6.6100378, -0.7844002, 0.7851286
2: -4.4578457, -3.3069038, -4.4597855, -3.3048768, -0.7666106, 0.7648940
3: -6.0369234, -4.5567169, -6.0407534, -4.5536809, -0.9685755, 0.9627242
4: -12.2408800, -10.3860064, -12.2438507, -10.3794670, -0.9187860, 0.9126034
5: -6.6909647, -5.6331367, -6.6980124, -5.6321039, -0.5034108, 0.5089904
6: -5.5420089, -4.3639770, -5.5545154, -4.3627801, -0.7307615, 0.7418883
7: -11.0493507, -9.8018007, -11.0508938, -9.7967606, -0.8165834, 0.8134208
8: 9.8868141, 10.8243570, 9.8847666, 10.8270826, -0.6595950, 0.6586864
9: -7.5092077, -5.9313622, -7.5150871, -5.9291000, -0.9101477, 0.9157236

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 4611
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_B2_B1

### Relational analysis result of NS_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188007, upper bound: 0.4177138
time: 3.36 seconds

## Relational analysis of NS_B2_B2

### Relational analysis result of NS_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4188007, upper bound: 0.4188015
time: 3.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.49 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 8, lower bound: -0.4163089, upper bound: 0.4163101
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 8, lower bound: -0.4163089, upper bound: 0.4163101
NS_B2_B1, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 8, lower bound: -0.4188007, upper bound: 0.4177138
NS_B2_B2, status: Status.UNKNOWN, split count: 2, time: 29.49
Output dim: 8, lower bound: -0.4188007, upper bound: 0.4188015

## BFS NS instance: NS_B1_A1

### Backsubstitution after applying NS history:
0: -5.9572535, -4.5665827, -5.9572535, -4.5665827, -0.8139334, 0.8139334
1: -7.9258809, -6.6168985, -7.9258809, -6.6168985, -0.7657073, 0.7657073
2: -4.4499092, -3.3082843, -4.4499092, -3.3082843, -0.7529221, 0.7529218
3: -6.0330505, -4.5658574, -6.0330505, -4.5658574, -0.9431348, 0.9431348
4: -12.2343588, -10.3876677, -12.2343588, -10.3876677, -0.9029362, 0.9029365
5: -6.6899819, -5.6392822, -6.6899819, -5.6392822, -0.4935582, 0.4935582
6: -5.5368838, -4.3732109, -5.5368838, -4.3732109, -0.7169902, 0.7169902
7: -11.0439234, -9.8036060, -11.0439234, -9.8036060, -0.8020999, 0.8021002
8: 9.8934946, 10.8220596, 9.8934946, 10.8220596, -0.6468987, 0.6468990
9: -7.5079975, -5.9405684, -7.5079975, -5.9405684, -0.8972726, 0.8972723

Time for backsubstitution: 22.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152182, upper bound: 0.4163093
time: 4.56 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163069, upper bound: 0.4163094
time: 3.19 seconds

## BFS NS instance: NS_B1_A2

### Backsubstitution after applying NS history:
0: -5.9802866, -4.5620289, -5.9572535, -4.5665827, -0.8394575, 0.8185482
1: -7.9359174, -6.6100378, -7.9258809, -6.6168985, -0.7789290, 0.7722648
2: -4.4597855, -3.3048768, -4.4499092, -3.3082843, -0.7633529, 0.7572281
3: -6.0407534, -4.5536809, -6.0330505, -4.5658574, -0.9514461, 0.9567654
4: -12.2438507, -10.3794670, -12.2343588, -10.3876677, -0.9124210, 0.9112236
5: -6.6980124, -5.6321039, -6.6899819, -5.6392822, -0.5016646, 0.5006762
6: -5.5545154, -4.3627801, -5.5368838, -4.3732109, -0.7324152, 0.7276912
7: -11.0508938, -9.7967606, -11.0439234, -9.8036060, -0.8105402, 0.8092880
8: 9.8847666, 10.8270826, 9.8934946, 10.8220596, -0.6562276, 0.6521182
9: -7.5150871, -5.9291000, -7.5079975, -5.9405684, -0.9055419, 0.9090481

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 4656

## Relational analysis of NS_B1_A2_A1

### Relational analysis result of NS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152182, upper bound: 0.4163094
time: 3.44 seconds

## Relational analysis of NS_B1_A2_A2

### Relational analysis result of NS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163069, upper bound: 0.4163094
time: 3.17 seconds

## BFS NS instance: NS_B2_B1

### Backsubstitution after applying NS history:
0: -5.9700723, -4.5664315, -5.9785600, -4.5620890, -0.8311875, 0.8443139
1: -7.9300156, -6.6112852, -7.9349637, -6.6103029, -0.7838080, 0.7838068
2: -4.4575186, -3.3069060, -4.4587622, -3.3048873, -0.7660232, 0.7635014
3: -6.0367851, -4.5568419, -6.0403290, -4.5540791, -0.9681635, 0.9621634
4: -12.2408028, -10.3861275, -12.2436247, -10.3798590, -0.9174306, 0.9112196
5: -6.6906238, -5.6331377, -6.6969471, -5.6321058, -0.5028915, 0.5077914
6: -5.5418701, -4.3644919, -5.5540719, -4.3643794, -0.7290738, 0.7409847
7: -11.0490999, -9.8018093, -11.0501375, -9.7967873, -0.8154953, 0.8116193
8: 9.8874626, 10.8242607, 9.8867979, 10.8267756, -0.6586363, 0.6565399
9: -7.5090265, -5.9314375, -7.5145292, -5.9293370, -0.9096801, 0.9150507

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4611
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4611

## Relational analysis of NS_B2_B1_A1

### Relational analysis result of NS_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4172116
time: 3.21 seconds

## Relational analysis of NS_B2_B1_A2

### Relational analysis result of NS_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4164120
time: 3.36 seconds

## BFS NS instance: NS_B2_B2

### Backsubstitution after applying NS history:
0: -5.9706216, -4.5664129, -5.9805532, -4.5606027, -0.8332791, 0.8460588
1: -7.9303188, -6.6111856, -7.9362011, -6.6093140, -0.7852385, 0.7853394
2: -4.4578433, -3.3069038, -4.4602833, -3.3039222, -0.7674260, 0.7654231
3: -6.0369225, -4.5567160, -6.0408211, -4.5522475, -0.9702811, 0.9627850
4: -12.2408791, -10.3860044, -12.2443275, -10.3791208, -0.9197855, 0.9123814
5: -6.6909637, -5.6331367, -6.6981382, -5.6310897, -0.5042732, 0.5088246
6: -5.5420084, -4.3639789, -5.5564580, -4.3625832, -0.7306426, 0.7422336
7: -11.0493498, -9.8018017, -11.0515442, -9.7962561, -0.8165824, 0.8147151
8: 9.8868160, 10.8243570, 9.8846445, 10.8297348, -0.6622477, 0.6579912
9: -7.5092072, -5.9313641, -7.5152535, -5.9283910, -0.9108675, 0.9157946

Time for backsubstitution: 22.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4611
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 4656
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 4611

## Relational analysis of NS_B2_B2_A1

### Relational analysis result of NS_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4182994
time: 3.39 seconds

## Relational analysis of NS_B2_B2_A2

### Relational analysis result of NS_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4174983
time: 3.33 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 29.21 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4152182, upper bound: 0.4163093
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4163069, upper bound: 0.4163094
NS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4152182, upper bound: 0.4163094
NS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4163069, upper bound: 0.4163094
NS_B2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4172116
NS_B2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4164120
NS_B2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4182994
NS_B2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 29.21
Output dim: 8, lower bound: -0.4163070, upper bound: 0.4174983

## BFS NS instance: NS_B1_A1_A1

### Backsubstitution after applying NS history:
0: -5.9555340, -4.5666399, -5.9567032, -4.5666013, -0.8119366, 0.8129888
1: -7.9249263, -6.6171651, -7.9255786, -6.6169977, -0.7643728, 0.7651162
2: -4.4488869, -3.3082948, -4.4495816, -3.3082876, -0.7515316, 0.7523315
3: -6.0326247, -4.5662522, -6.0329118, -4.5659833, -0.9425702, 0.9427278
4: -12.2341299, -10.3880615, -12.2342854, -10.3877916, -0.9015527, 0.9015784
5: -6.6889153, -5.6392851, -6.6896434, -5.6392851, -0.4923586, 0.4930397
6: -5.5364347, -4.3748083, -5.5367413, -4.3737235, -0.7162838, 0.7153027
7: -11.0431757, -9.8036299, -11.0436735, -9.8036146, -0.8002989, 0.8010120
8: 9.8955288, 10.8217468, 9.8941441, 10.8219604, -0.6447492, 0.6459327
9: -7.5074358, -5.9407997, -7.5078139, -5.9406424, -0.8966007, 0.8968072

Time for backsubstitution: 21.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5843

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152164, upper bound: 0.4157486
time: 4.09 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152164, upper bound: 0.4163061
time: 3.71 seconds

## BFS NS instance: NS_B1_A1_A2

### Backsubstitution after applying NS history:
0: -5.9575233, -4.5651550, -5.9572506, -4.5665798, -0.8137000, 0.8150821
1: -7.9261661, -6.6161766, -7.9258819, -6.6168976, -0.7659168, 0.7665446
2: -4.4504023, -3.3073304, -4.4499054, -3.3082850, -0.7534359, 0.7537355
3: -6.0331163, -4.5644302, -6.0330515, -4.5658579, -0.9431915, 0.9448586
4: -12.2348328, -10.3873138, -12.2343550, -10.3876696, -0.9027150, 0.9039371
5: -6.6901088, -5.6382689, -6.6899824, -5.6392822, -0.4934049, 0.4944205
6: -5.5388284, -4.3730149, -5.5368810, -4.3732128, -0.7189887, 0.7168703
7: -11.0445719, -9.8030977, -11.0439224, -9.8036070, -0.8033977, 0.8021004
8: 9.8933773, 10.8247147, 9.8934937, 10.8220587, -0.6461973, 0.6495550
9: -7.5081577, -5.9398594, -7.5079947, -5.9405670, -0.8973434, 0.8979905

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5843

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163051, upper bound: 0.4157498
time: 3.29 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163051, upper bound: 0.4163061
time: 3.32 seconds

## BFS NS instance: NS_B1_A2_A1

### Backsubstitution after applying NS history:
0: -5.9785600, -4.5620890, -5.9567032, -4.5666013, -0.8374565, 0.8176033
1: -7.9349637, -6.6103029, -7.9255786, -6.6169977, -0.7776067, 0.7716738
2: -4.4587622, -3.3048873, -4.4495816, -3.3082876, -0.7619619, 0.7566373
3: -6.0403290, -4.5540791, -6.0329118, -4.5659833, -0.9508872, 0.9563529
4: -12.2436247, -10.3798590, -12.2342854, -10.3877916, -0.9110374, 0.9098692
5: -6.6969471, -5.6321058, -6.6896434, -5.6392851, -0.5004658, 0.5001576
6: -5.5540719, -4.3643794, -5.5367413, -4.3737235, -0.7315121, 0.7260027
7: -11.0501375, -9.7967873, -11.0436735, -9.8036146, -0.8087392, 0.8081996
8: 9.8867979, 10.8267756, 9.8941441, 10.8219604, -0.6540794, 0.6511588
9: -7.5145292, -5.9293370, -7.5078139, -5.9406424, -0.9048700, 0.9085798

Time for backsubstitution: 22.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_B1_A2_A1_A1

### Relational analysis result of NS_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4166506, upper bound: 0.4163063
time: 3.35 seconds

## Relational analysis of NS_B1_A2_A1_A2

### Relational analysis result of NS_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4172071, upper bound: 0.4163063
time: 3.42 seconds

## BFS NS instance: NS_B1_A2_A2

### Backsubstitution after applying NS history:
0: -5.9805532, -4.5606027, -5.9572506, -4.5665798, -0.8392220, 0.8196969
1: -7.9362011, -6.6093140, -7.9258819, -6.6168976, -0.7791395, 0.7731022
2: -4.4602833, -3.3039222, -4.4499054, -3.3082850, -0.7638824, 0.7580414
3: -6.0408211, -4.5522475, -6.0330515, -4.5658579, -0.9515061, 0.9584808
4: -12.2443275, -10.3791208, -12.2343550, -10.3876696, -0.9121993, 0.9122236
5: -6.6981382, -5.6310897, -6.6899824, -5.6392822, -0.5014776, 0.5015388
6: -5.5564580, -4.3625832, -5.5368810, -4.3732128, -0.7327602, 0.7275724
7: -11.0515442, -9.7962561, -11.0439224, -9.8036070, -0.8118348, 0.8092873
8: 9.8846445, 10.8297348, 9.8934937, 10.8220587, -0.6555324, 0.6547709
9: -7.5152535, -5.9283910, -7.5079947, -5.9405670, -0.9056132, 0.9097676

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_B1_A2_A2_A1

### Relational analysis result of NS_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4177390, upper bound: 0.4163063
time: 3.14 seconds

## Relational analysis of NS_B1_A2_A2_A2

### Relational analysis result of NS_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4182948, upper bound: 0.4163063
time: 3.65 seconds

## BFS NS instance: NS_B2_B1_A1

### Backsubstitution after applying NS history:
0: -5.9567032, -4.5666013, -5.9785600, -4.5620890, -0.8176033, 0.8374565
1: -7.9255786, -6.6169977, -7.9349637, -6.6103029, -0.7716737, 0.7776068
2: -4.4495816, -3.3082876, -4.4587622, -3.3048873, -0.7566373, 0.7619617
3: -6.0329118, -4.5659833, -6.0403290, -4.5540791, -0.9563532, 0.9508870
4: -12.2342854, -10.3877916, -12.2436247, -10.3798590, -0.9098692, 0.9110374
5: -6.6896434, -5.6392851, -6.6969471, -5.6321058, -0.5001576, 0.5004658
6: -5.5367413, -4.3737235, -5.5540719, -4.3643794, -0.7260027, 0.7315121
7: -11.0436735, -9.8036146, -11.0501375, -9.7967873, -0.8081994, 0.8087394
8: 9.8941441, 10.8219604, 9.8867979, 10.8267756, -0.6511590, 0.6540797
9: -7.5078139, -5.9406424, -7.5145292, -5.9293370, -0.9085798, 0.9048703

Time for backsubstitution: 22.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 5843

## Relational analysis of NS_B2_B1_A1_B1

### Relational analysis result of NS_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4166519
time: 3.51 seconds

## Relational analysis of NS_B2_B1_A1_B2

### Relational analysis result of NS_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4172083
time: 3.14 seconds

## BFS NS instance: NS_B2_B1_A2

### Backsubstitution after applying NS history:
0: -5.9797363, -4.5620484, -5.9785600, -4.5620890, -0.8454978, 0.8444536
1: -7.9356122, -6.6101351, -7.9349637, -6.6103029, -0.7860420, 0.7852960
2: -4.4594574, -3.3048801, -4.4587622, -3.3048873, -0.7674828, 0.7666805
3: -6.0406156, -4.5538077, -6.0403290, -4.5540791, -0.9691119, 0.9689534
4: -12.2437801, -10.3795919, -12.2436247, -10.3798590, -0.9120517, 0.9120228
5: -6.6976728, -5.6321044, -6.6969471, -5.6321058, -0.5033722, 0.5026895
6: -5.5543761, -4.3632946, -5.5540719, -4.3643794, -0.7333252, 0.7343109
7: -11.0506420, -9.7967720, -11.0501375, -9.7967873, -0.8148122, 0.8140998
8: 9.8854160, 10.8269854, 9.8867979, 10.8267756, -0.6590571, 0.6578708
9: -7.5149069, -5.9291754, -7.5145292, -5.9293370, -0.9108417, 0.9106374

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_B2_B1_A2_A1

### Relational analysis result of NS_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4157473, upper bound: 0.4164089
time: 3.43 seconds

## Relational analysis of NS_B2_B1_A2_A2

### Relational analysis result of NS_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163036, upper bound: 0.4152164
time: 3.78 seconds

## BFS NS instance: NS_B2_B2_A1

### Backsubstitution after applying NS history:
0: -5.9572506, -4.5665798, -5.9805532, -4.5606027, -0.8196969, 0.8392217
1: -7.9258819, -6.6168976, -7.9362011, -6.6093140, -0.7731023, 0.7791396
2: -4.4499054, -3.3082850, -4.4602833, -3.3039222, -0.7580414, 0.7638824
3: -6.0330515, -4.5658579, -6.0408211, -4.5522475, -0.9584808, 0.9515064
4: -12.2343550, -10.3876696, -12.2443275, -10.3791208, -0.9122233, 0.9121993
5: -6.6899824, -5.6392822, -6.6981382, -5.6310897, -0.5015386, 0.5014776
6: -5.5368810, -4.3732128, -5.5564580, -4.3625832, -0.7275724, 0.7327602
7: -11.0439224, -9.8036070, -11.0515442, -9.7962561, -0.8092875, 0.8118348
8: 9.8934937, 10.8220587, 9.8846445, 10.8297348, -0.6547709, 0.6555326
9: -7.5079947, -5.9405670, -7.5152535, -5.9283910, -0.9097676, 0.9056129

Time for backsubstitution: 21.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 1, pos: 5843

## Relational analysis of NS_B2_B2_A1_B1

### Relational analysis result of NS_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4177402
time: 3.22 seconds

## Relational analysis of NS_B2_B2_A1_B2

### Relational analysis result of NS_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4182960
time: 3.24 seconds

## BFS NS instance: NS_B2_B2_A2

### Backsubstitution after applying NS history:
0: -5.9802842, -4.5620284, -5.9805532, -4.5606027, -0.8475881, 0.8461983
1: -7.9359169, -6.6100349, -7.9362011, -6.6093140, -0.7874727, 0.7868531
2: -4.4597821, -3.3048770, -4.4602833, -3.3039222, -0.7688870, 0.7686012
3: -6.0407529, -4.5536804, -6.0408211, -4.5522475, -0.9712286, 0.9695821
4: -12.2438526, -10.3794699, -12.2443275, -10.3791208, -0.9144058, 0.9131844
5: -6.6980133, -5.6321039, -6.6981382, -5.6310897, -0.5047543, 0.5037389
6: -5.5545168, -4.3627839, -5.5564580, -4.3625832, -0.7348928, 0.7370148
7: -11.0508947, -9.7967644, -11.0515442, -9.7962561, -0.8159013, 0.8171959
8: 9.8847675, 10.8270826, 9.8846445, 10.8297348, -0.6626689, 0.6593218
9: -7.5150867, -5.9291010, -7.5152535, -5.9283910, -0.9120309, 0.9113817

Time for backsubstitution: 22.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 5843
type: A, layer: 1, pos: 4656
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_B2_B2_A2_A1

### Relational analysis result of NS_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4157473, upper bound: 0.4163049
time: 3.16 seconds

## Relational analysis of NS_B2_B2_A2_A2

### Relational analysis result of NS_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163036, upper bound: 0.4174952
time: 3.15 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 28.69 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4152164, upper bound: 0.4157486
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4152164, upper bound: 0.4163061
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163051, upper bound: 0.4157498
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163051, upper bound: 0.4163061
NS_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4166506, upper bound: 0.4163063
NS_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4172071, upper bound: 0.4163063
NS_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4177390, upper bound: 0.4163063
NS_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4182948, upper bound: 0.4163063
NS_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4166519
NS_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4172083
NS_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4157473, upper bound: 0.4164089
NS_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163036, upper bound: 0.4152164
NS_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4177402
NS_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163039, upper bound: 0.4182960
NS_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4157473, upper bound: 0.4163049
NS_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 28.69
Output dim: 8, lower bound: -0.4163036, upper bound: 0.4174952

## BFS NS instance: NS_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -5.9555340, -4.5666399, -5.9552593, -4.5666127, -0.8118403, 0.8114936
1: -7.9249263, -6.6171651, -7.9254122, -6.6175256, -0.7637973, 0.7647654
2: -4.4488869, -3.3082948, -4.4493451, -3.3085506, -0.7512603, 0.7521114
3: -6.0326247, -4.5662522, -6.0328484, -4.5679774, -0.9406171, 0.9426558
4: -12.2341299, -10.3880615, -12.2340746, -10.3879232, -0.9014802, 0.9014034
5: -6.6889153, -5.6392851, -6.6895952, -5.6397772, -0.4918629, 0.4929955
6: -5.5364347, -4.3748083, -5.5352163, -4.3737884, -0.7162397, 0.7137783
7: -11.0431757, -9.8036299, -11.0435553, -9.8037415, -0.8000751, 0.8009257
8: 9.8955288, 10.8217468, 9.8948212, 10.8219185, -0.6446733, 0.6451807
9: -7.5074358, -5.9407997, -7.5068617, -5.9407387, -0.8964825, 0.8957572

Time for backsubstitution: 22.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.32 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4146594, upper bound: 0.4157495
time: 3.38 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4146594, upper bound: 0.4157495
time: 3.18 seconds

## BFS NS instance: NS_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -5.9555345, -4.5666409, -5.9582968, -4.5482922, -0.8232578, 0.8204410
1: -7.9249272, -6.6171641, -7.9333825, -6.6167426, -0.7673070, 0.7726350
2: -4.4488878, -3.3082950, -4.4546280, -3.3080280, -0.7522595, 0.7578244
3: -6.0326238, -4.5662565, -6.0620384, -4.5629759, -0.9530330, 0.9579792
4: -12.2341309, -10.3880596, -12.2365808, -10.3776007, -0.9137752, 0.9051194
5: -6.6889157, -5.6392870, -6.6981354, -5.6392231, -0.4938194, 0.5010396
6: -5.5364318, -4.3748078, -5.5371928, -4.3506546, -0.7248707, 0.7199302
7: -11.0431757, -9.8036308, -11.0490303, -9.7994957, -0.8109381, 0.8074057
8: 9.8955307, 10.8217468, 9.8935966, 10.8316488, -0.6544971, 0.6493089
9: -7.5074339, -5.9408007, -7.5094185, -5.9227829, -0.9048157, 0.9032655

Time for backsubstitution: 22.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_B1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152164, upper bound: 0.4152166
time: 3.57 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4152164, upper bound: 0.4163061
time: 3.59 seconds

## BFS NS instance: NS_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -5.9575233, -4.5651550, -5.9558072, -4.5665951, -0.8136041, 0.8135872
1: -7.9261661, -6.6161766, -7.9257154, -6.6174245, -0.7653413, 0.7661939
2: -4.4504023, -3.3073304, -4.4496698, -3.3085480, -0.7531643, 0.7535167
3: -6.0331163, -4.5644302, -6.0329852, -4.5678520, -0.9412398, 0.9447865
4: -12.2348328, -10.3873138, -12.2341461, -10.3877993, -0.9026432, 0.9037623
5: -6.6901088, -5.6382689, -6.6899357, -5.6397762, -0.4929090, 0.4943768
6: -5.5388284, -4.3730149, -5.5353575, -4.3732781, -0.7189448, 0.7153459
7: -11.0445719, -9.8030977, -11.0438061, -9.8037310, -0.8031733, 0.8020139
8: 9.8933773, 10.8247147, 9.8941727, 10.8220167, -0.6461215, 0.6488032
9: -7.5081577, -5.9398594, -7.5070410, -5.9406629, -0.8972249, 0.8969400

Time for backsubstitution: 22.14 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 4656
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 1, pos: 5843

## Relational analysis of NS_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4157481, upper bound: 0.4157494
time: 3.43 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4157481, upper bound: 0.4157494
time: 3.35 seconds

## BFS NS instance: NS_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -5.9575210, -4.5651546, -5.9588485, -4.5482764, -0.8248683, 0.8225346
1: -7.9261656, -6.6161785, -7.9336896, -6.6166420, -0.7688513, 0.7740626
2: -4.4504023, -3.3073313, -4.4549522, -3.3080246, -0.7541647, 0.7592297
3: -6.0331173, -4.5644317, -6.0621738, -4.5628486, -0.9536557, 0.9593868
4: -12.2348309, -10.3873148, -12.2366524, -10.3774757, -0.9149389, 0.9074750
5: -6.6901102, -5.6382709, -6.6984739, -5.6392226, -0.4948657, 0.5015715
6: -5.5388260, -4.3730135, -5.5373335, -4.3501449, -0.7261112, 0.7214983
7: -11.0445709, -9.8030987, -11.0492764, -9.7994900, -0.8140373, 0.8084934
8: 9.8933792, 10.8247147, 9.8929472, 10.8317471, -0.6559441, 0.6529315
9: -7.5081558, -5.9398603, -7.5095997, -5.9227076, -0.9055645, 0.9044468

Time for backsubstitution: 22.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4656
type: A, layer: 1, pos: 5843
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 4656

## Relational analysis of NS_B1_A1_A2_B2_B1

### Relational analysis result of NS_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163050, upper bound: 0.4152166
time: 3.38 seconds

## Relational analysis of NS_B1_A1_A2_B2_B2

### Relational analysis result of NS_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.4163052, upper bound: 0.4152166
time: 3.37 seconds

## BFS NS instance: NS_B1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -5.9771214, -4.5620999, -5.9567032, -4.5666013, -0.8359618, 0.8175073
1: -7.9348035, -6.6108317, -7.9255786, -6.6169977, -0.7772579, 0.7710981
2: -4.4585218, -3.3051505, -4.4495816, -3.3082876, -0.7617283, 0.7563665
3: -6.0402670, -4.5560803, -6.0329118, -4.5659833, -0.9508166, 0.9544086
4: -12.2434120, -10.3799887, -12.2342854, -10.3877916, -0.9108629, 0.9097993
5: -6.6969013, -5.6325994, -6.6896434, -5.6392851, -0.5004231, 0.4996620
6: -5.5525537, -4.3644438, -5.5367413, -4.3737235, -0.7299826, 0.7259591
7: -11.0500202, -9.7969112, -11.0436735, -9.8036146, -0.8086524, 0.8079760
8: 9.8874865, 10.8267355, 9.8941441, 10.8219604, -0.6533189, 0.6510844
9: -7.5135756, -5.9294353, -7.5078139, -5.9406424, -0.9038200, 0.9084601

Time for backsubstitution: 22.15 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.10 + 554.92 = 612.02 seconds
