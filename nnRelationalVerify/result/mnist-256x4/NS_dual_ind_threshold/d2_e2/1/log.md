## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 4.241187818999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-2.4096687, 2.0148206, -2.4096687, 2.0148206, -4.4244890, 4.4244890)
1: (-1.9410727, 1.8341657, -1.9410727, 1.8341657, -3.7752380, 3.7752385)
2: (-2.4231946, 1.8821208, -2.4231946, 1.8821208, -4.3053155, 4.3053155)
3: (-2.6752136, 1.5974345, -2.6752136, 1.5974345, -4.2726479, 4.2726479)
4: (-2.8320472, 1.9358228, -2.8320472, 1.9358228, -4.7678690, 4.7678695)
5: (-2.2663326, 2.0560665, -2.2663326, 2.0560665, -4.3223991, 4.3223987)
6: (-2.0454202, 2.3011079, -2.0454202, 2.3011079, -4.3465281, 4.3465276)
7: (-2.3598943, 2.3240304, -2.3598943, 2.3240304, -4.6839247, 4.6839247)
8: (-3.3761089, 1.7121029, -3.3761089, 1.7121029, -5.0882120, 5.0882120)
9: (-2.1206036, 2.2545609, -2.1206036, 2.2545609, -4.3751640, 4.3751645)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 3.57 = 5.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -4.2840282, upper bound: 4.2840282

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 89
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 89

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2829174, upper bound: 4.2830134
time: 4.86 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2828628, upper bound: 4.2828628
time: 2.47 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.53 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 8, lower bound: -4.2829174, upper bound: 4.2830134
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 8, lower bound: -4.2828628, upper bound: 4.2828628

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -2.2406449, 1.8774600, -2.4096687, 2.0148206, -4.2554655, 4.2871284
1: -1.8108708, 1.7158930, -1.9410727, 1.8341657, -3.6450365, 3.6569657
2: -2.2227380, 1.7685162, -2.4231946, 1.8821208, -4.1048589, 4.1917109
3: -2.4834917, 1.5000969, -2.6752136, 1.5974345, -4.0809259, 4.1753106
4: -2.6360934, 1.8023062, -2.8320472, 1.9358228, -4.5719161, 4.6343532
5: -2.0870905, 1.9165885, -2.2663326, 2.0560665, -4.1431570, 4.1829214
6: -1.8853948, 2.1440406, -2.0454202, 2.3011079, -4.1865025, 4.1894608
7: -2.1964598, 2.1638086, -2.3598943, 2.3240304, -4.5204902, 4.5237026
8: -3.1112311, 1.5845833, -3.3761089, 1.7121029, -4.8233337, 4.9606924
9: -1.9620469, 2.1019683, -2.1206036, 2.2545609, -4.2166080, 4.2225718

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2828617, upper bound: 4.2828617
time: 2.01 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2828617, upper bound: 4.2828628
time: 1.42 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -2.5081155, 2.0808802, -2.3669100, 1.9797350, -4.4878507, 4.4477901
1: -2.0209517, 1.9009715, -1.9079409, 1.8047125, -3.8256643, 3.8089123
2: -2.5483272, 1.9342301, -2.3727069, 1.8530551, -4.4013824, 4.3069372
3: -2.7958884, 1.6495739, -2.6271243, 1.5725070, -4.3683953, 4.2766981
4: -2.9415290, 2.0162282, -2.7832031, 1.9012948, -4.8428230, 4.7994313
5: -2.3683844, 2.0932498, -2.2218282, 2.0212359, -4.3896203, 4.3150778
6: -2.1282659, 2.3664422, -2.0052643, 2.2611723, -4.3894382, 4.3717065
7: -2.4692125, 2.4275060, -2.3186667, 2.2841616, -4.7533741, 4.7461729
8: -3.5112345, 1.7121710, -3.3083200, 1.6777759, -5.1890106, 5.0204911
9: -2.2055407, 2.3056865, -2.0809774, 2.2150795, -4.4206200, 4.3866639

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811795
time: 1.61 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
time: 1.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 8, lower bound: -4.2828617, upper bound: 4.2828617
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 8, lower bound: -4.2828617, upper bound: 4.2828628
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811795
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.24
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -2.2406449, 1.8774600, -2.2406449, 1.8774600, -4.1181049, 4.1181049
1: -1.8108708, 1.7158930, -1.8108708, 1.7158930, -3.5267639, 3.5267639
2: -2.2227380, 1.7685162, -2.2227380, 1.7685162, -3.9912543, 3.9912543
3: -2.4834917, 1.5000969, -2.4834917, 1.5000969, -3.9835887, 3.9835887
4: -2.6360934, 1.8023062, -2.6360934, 1.8023062, -4.4383993, 4.4383993
5: -2.0870905, 1.9165885, -2.0870905, 1.9165885, -4.0036793, 4.0036793
6: -1.8853948, 2.1440406, -1.8853948, 2.1440406, -4.0294352, 4.0294352
7: -2.1964598, 2.1638086, -2.1964598, 2.1638086, -4.3602686, 4.3602686
8: -3.1112311, 1.5845833, -3.1112311, 1.5845833, -4.6958141, 4.6958141
9: -1.9620469, 2.1019683, -1.9620469, 2.1019683, -4.0640154, 4.0640154

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2812104, upper bound: 4.2817478
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
time: 2.06 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -2.2406449, 1.8774600, -2.5081155, 2.0808802, -4.3215251, 4.3855753
1: -1.8108708, 1.7158930, -2.0209517, 1.9009715, -3.7118423, 3.7368448
2: -2.2227380, 1.7685162, -2.5483272, 1.9342301, -4.1569681, 4.3168435
3: -2.4834917, 1.5000969, -2.7958884, 1.6495739, -4.1330652, 4.2959852
4: -2.6360934, 1.8023062, -2.9415290, 2.0162282, -4.6523218, 4.7438345
5: -2.0870905, 1.9165885, -2.3683844, 2.0932498, -4.1803398, 4.2849731
6: -1.8853948, 2.1440406, -2.1282659, 2.3664422, -4.2518368, 4.2723064
7: -2.1964598, 2.1638086, -2.4692125, 2.4275060, -4.6239657, 4.6330214
8: -3.1112311, 1.5845833, -3.5112345, 1.7121710, -4.8234019, 5.0958176
9: -1.9620469, 2.1019683, -2.2055407, 2.3056865, -4.2677336, 4.3075089

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2812104, upper bound: 4.2817478
time: 5.39 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
time: 3.64 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -2.5081155, 2.0808802, -2.1980474, 1.8333367, -4.3414521, 4.2789278
1: -2.0209517, 1.9009715, -1.7727579, 1.6854051, -3.7063570, 3.6737294
2: -2.5483272, 1.9342301, -2.1616793, 1.7373395, -4.2856665, 4.0959091
3: -2.7958884, 1.6495739, -2.4298658, 1.4703407, -4.2662292, 4.0794396
4: -2.9415290, 2.0162282, -2.5873692, 1.7596258, -4.7011547, 4.6035976
5: -2.3683844, 2.0932498, -2.0453029, 1.8900293, -4.2584138, 4.1385522
6: -2.1282659, 2.3664422, -1.8426377, 2.0929298, -4.2211957, 4.2090797
7: -2.4692125, 2.4275060, -2.1488485, 2.1226454, -4.5918579, 4.5763545
8: -3.5112345, 1.7121710, -3.0182483, 1.5337818, -5.0450163, 4.7304187
9: -2.2055407, 2.3056865, -1.9216177, 2.0531349, -4.2586756, 4.2273045

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 242

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
time: 8.47 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
time: 1.67 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -2.4547141, 2.0361223, -2.6439631, 2.1545300, -4.6092443, 4.6800852
1: -1.9785997, 1.8632408, -2.1317694, 2.0013943, -3.9799938, 3.9950104
2: -2.4822431, 1.8978560, -2.7182810, 2.0229354, -4.5051785, 4.6161370
3: -2.7338409, 1.6172525, -2.9624031, 1.7238531, -4.4576941, 4.5796556
4: -2.8801756, 1.9712511, -3.1114035, 2.1265557, -5.0067310, 5.0826545
5: -2.3130271, 2.0515361, -2.5210524, 2.1904237, -4.5034509, 4.5725884
6: -2.0776491, 2.3153532, -2.2560430, 2.4719207, -4.5495701, 4.5713959
7: -2.4152641, 2.3766313, -2.6142220, 2.5757518, -4.9910159, 4.9908533
8: -3.4209237, 1.6711793, -3.7146823, 1.7528028, -5.1737266, 5.3858614
9: -2.1555927, 2.2556434, -2.3409052, 2.4129767, -4.5685692, 4.5965486

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
time: 2.54 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
time: 1.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 6.47 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2812104, upper bound: 4.2817478
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2812104, upper bound: 4.2817478
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 6.47
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2811020

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -2.0833631, 1.7416940, -2.2406449, 1.8774600, -3.9608231, 3.9823389
1: -1.6855249, 1.6037705, -1.8108708, 1.7158930, -3.4014180, 3.4146414
2: -2.0269756, 1.6622633, -2.2227380, 1.7685162, -3.7954917, 3.8850012
3: -2.3013508, 1.4037338, -2.4834917, 1.5000969, -3.8014479, 3.8872256
4: -2.4549296, 1.6684861, -2.6360934, 1.8023062, -4.2572355, 4.3045797
5: -1.9207706, 1.7953770, -2.0870905, 1.9165885, -3.8373592, 3.8824675
6: -1.7350885, 1.9856529, -1.8853948, 2.1440406, -3.8791289, 3.8710477
7: -2.0391674, 2.0143762, -2.1964598, 2.1638086, -4.2029762, 4.2108359
8: -2.8381224, 1.4534273, -3.1112311, 1.5845833, -4.4227057, 4.5646582
9: -1.8155354, 1.9517577, -1.9620469, 2.1019683, -3.9175038, 3.9138045

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
time: 9.48 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
time: 1.92 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.5353227, 2.0650425, -2.1925833, 1.8363951, -4.3717179, 4.2576256
1: -2.0477524, 1.9247468, -1.7728690, 1.6815726, -3.7293248, 3.6976156
2: -2.5890172, 1.9487718, -2.1633863, 1.7356799, -4.3246970, 4.1121583
3: -2.8394561, 1.6611673, -2.4280965, 1.4706440, -4.3101001, 4.0892639
4: -2.9864419, 2.0390973, -2.5805106, 1.7623138, -4.7487555, 4.6196079
5: -2.4047542, 2.0988798, -2.0365386, 1.8789510, -4.2837052, 4.1354184
6: -2.1533420, 2.3689156, -1.8393116, 2.0979171, -4.2512589, 4.2082272
7: -2.5078759, 2.4724133, -2.1482964, 2.1183105, -4.6261864, 4.6207094
8: -3.5472908, 1.6696390, -3.0273857, 1.5456331, -5.0929241, 4.6970248
9: -2.2390659, 2.3143539, -1.9170835, 2.0563293, -4.2953949, 4.2314377

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
time: 1.47 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -2.0833631, 1.7416940, -2.5081155, 2.0808802, -4.1642432, 4.2498093
1: -1.6855249, 1.6037705, -2.0209517, 1.9009715, -3.5864964, 3.6247222
2: -2.0269756, 1.6622633, -2.5483272, 1.9342301, -3.9612057, 4.2105904
3: -2.3013508, 1.4037338, -2.7958884, 1.6495739, -3.9509249, 4.1996222
4: -2.4549296, 1.6684861, -2.9415290, 2.0162282, -4.4711580, 4.6100144
5: -1.9207706, 1.7953770, -2.3683844, 2.0932498, -4.0140204, 4.1637611
6: -1.7350885, 1.9856529, -2.1282659, 2.3664422, -4.1015306, 4.1139183
7: -2.0391674, 2.0143762, -2.4692125, 2.4275060, -4.4666734, 4.4835887
8: -2.8381224, 1.4534273, -3.5112345, 1.7121710, -4.5502934, 4.9646616
9: -1.8155354, 1.9517577, -2.2055407, 2.3056865, -4.1212215, 4.1572981

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 242

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
time: 1.88 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
time: 2.04 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.5353227, 2.0650425, -2.4547141, 2.0361223, -4.5714450, 4.5197563
1: -2.0477524, 1.9247468, -1.9785997, 1.8632408, -3.9109931, 3.9033465
2: -2.5890172, 1.9487718, -2.4822431, 1.8978560, -4.4868731, 4.4310150
3: -2.8394561, 1.6611673, -2.7338409, 1.6172525, -4.4567084, 4.3950081
4: -2.9864419, 2.0390973, -2.8801756, 1.9712511, -4.9576931, 4.9192729
5: -2.4047542, 2.0988798, -2.3130271, 2.0515361, -4.4562902, 4.4119072
6: -2.1533420, 2.3689156, -2.0776491, 2.3153532, -4.4686952, 4.4465647
7: -2.5078759, 2.4724133, -2.4152641, 2.3766313, -4.8845072, 4.8876772
8: -3.5472908, 1.6696390, -3.4209237, 1.6711793, -5.2184701, 5.0905628
9: -2.2390659, 2.3143539, -2.1555927, 2.2556434, -4.4947090, 4.4699469

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
time: 3.40 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.3383520, 1.9373052, -2.1980474, 1.8333367, -4.1716886, 4.1353526
1: -1.8850859, 1.7810353, -1.7727579, 1.6854051, -3.5704911, 3.5537932
2: -2.3362648, 1.8184144, -2.1616793, 1.7373395, -4.0736046, 3.9800937
3: -2.5977340, 1.5467988, -2.4298658, 1.4703407, -4.0680747, 3.9766645
4: -2.7463758, 1.8715136, -2.5873692, 1.7596258, -4.5060015, 4.4588828
5: -2.1912882, 1.9615949, -2.0453029, 1.8900293, -4.0813174, 4.0068979
6: -1.9660486, 2.1979883, -1.8426377, 2.0929298, -4.0589786, 4.0406260
7: -2.2969217, 2.2653391, -2.1488485, 2.1226454, -4.4195671, 4.4141874
8: -3.2221632, 1.5760597, -3.0182483, 1.5337818, -4.7559452, 4.5943079
9: -2.0473509, 2.1442001, -1.9216177, 2.0531349, -4.1004858, 4.0658178

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
time: 1.53 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.7924316, 2.2534561, -2.1980474, 1.8333367, -4.6257682, 4.4515038
1: -2.2482793, 2.1008081, -1.7727579, 1.6854051, -3.9336843, 3.8735662
2: -2.9004233, 2.1074500, -2.1616793, 1.7373395, -4.6377630, 4.2691293
3: -3.1394947, 1.8035789, -2.4298658, 1.4703407, -4.6098356, 4.2334447
4: -3.2741139, 2.2465856, -2.5873692, 1.7596258, -5.0337396, 4.8339548
5: -2.6698141, 2.2723768, -2.0453029, 1.8900293, -4.5598435, 4.3176794
6: -2.3863821, 2.5809729, -1.8426377, 2.0929298, -4.4793119, 4.4236107
7: -2.7691746, 2.7219460, -2.1488485, 2.1226454, -4.8918200, 4.8707943
8: -3.9220977, 1.7971760, -3.0182483, 1.5337818, -5.4558792, 4.8154244
9: -2.4696856, 2.5105207, -1.9216177, 2.0531349, -4.5228205, 4.4321384

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
time: 1.55 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.3383520, 1.9373052, -2.6439631, 2.1545300, -4.4928818, 4.5812683
1: -1.8850859, 1.7810353, -2.1317694, 2.0013943, -3.8864803, 3.9128046
2: -2.3362648, 1.8184144, -2.7182810, 2.0229354, -4.3592005, 4.5366955
3: -2.5977340, 1.5467988, -2.9624031, 1.7238531, -4.3215871, 4.5092020
4: -2.7463758, 1.8715136, -3.1114035, 2.1265557, -4.8729315, 4.9829168
5: -2.1912882, 1.9615949, -2.5210524, 2.1904237, -4.3817120, 4.4826474
6: -1.9660486, 2.1979883, -2.2560430, 2.4719207, -4.4379692, 4.4540310
7: -2.2969217, 2.2653391, -2.6142220, 2.5757518, -4.8726735, 4.8795614
8: -3.2221632, 1.5760597, -3.7146823, 1.7528028, -4.9749660, 5.2907419
9: -2.0473509, 2.1442001, -2.3409052, 2.4129767, -4.4603276, 4.4851055

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 1.47 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.7920773, 2.2534561, -2.6439631, 2.1545300, -4.9466076, 4.8974190
1: -2.2482793, 2.0991929, -2.1317694, 2.0013943, -4.2496738, 4.2309623
2: -2.9004233, 2.1066322, -2.7182810, 2.0229354, -4.9233589, 4.8249130
3: -3.1394947, 1.8029028, -2.9624031, 1.7238531, -4.8633480, 4.7653060
4: -3.2741139, 2.2432344, -3.1114035, 2.1265557, -5.4006696, 5.3546381
5: -2.6697662, 2.2723768, -2.5210524, 2.1904237, -4.8601899, 4.7934294
6: -2.3863821, 2.5790639, -2.2560430, 2.4719207, -4.8583031, 4.8351068
7: -2.7691746, 2.7217488, -2.6142220, 2.5757518, -5.3449264, 5.3359709
8: -3.9214466, 1.7971760, -3.7146823, 1.7528028, -5.6742496, 5.5118585
9: -2.4696856, 2.5095091, -2.3409052, 2.4129767, -4.8826623, 4.8504143

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 89
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 89

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 1.81 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
time: 1.51 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.92 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2817699, upper bound: 4.2817699
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811604, upper bound: 4.2813999
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2815467, upper bound: 4.2811674
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.92
Output dim: 8, lower bound: -4.2811020, upper bound: 4.2810962

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.0833631, 1.7416940, -2.0833631, 1.7416940, -3.8250570, 3.8250570
1: -1.6855249, 1.6037705, -1.6855249, 1.6037705, -3.2892954, 3.2892954
2: -2.0269756, 1.6622633, -2.0269756, 1.6622633, -3.6892390, 3.6892390
3: -2.3013508, 1.4037338, -2.3013508, 1.4037338, -3.7050848, 3.7050848
4: -2.4549296, 1.6684861, -2.4549296, 1.6684861, -4.1234159, 4.1234159
5: -1.9207706, 1.7953770, -1.9207706, 1.7953770, -3.7161477, 3.7161477
6: -1.7350885, 1.9856529, -1.7350885, 1.9856529, -3.7207413, 3.7207413
7: -2.0391674, 2.0143762, -2.0391674, 2.0143762, -4.0535436, 4.0535436
8: -2.8381224, 1.4534273, -2.8381224, 1.4534273, -4.2915497, 4.2915497
9: -1.8155354, 1.9517577, -1.8155354, 1.9517577, -3.7672930, 3.7672930

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2813745
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2813745
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.0833631, 1.7416940, -2.5353227, 2.0650425, -4.1484056, 4.2770166
1: -1.6855249, 1.6037705, -2.0477524, 1.9247468, -3.6102717, 3.6515229
2: -2.0269756, 1.6622633, -2.5890172, 1.9487718, -3.9757476, 4.2512803
3: -2.3013508, 1.4037338, -2.8394561, 1.6611673, -3.9625182, 4.2431898
4: -2.4549296, 1.6684861, -2.9864419, 2.0390973, -4.4940271, 4.6549282
5: -1.9207706, 1.7953770, -2.4047542, 2.0988798, -4.0196505, 4.2001314
6: -1.7350885, 1.9856529, -2.1533420, 2.3689156, -4.1040039, 4.1389952
7: -2.0391674, 2.0143762, -2.5078759, 2.4724133, -4.5115805, 4.5222521
8: -2.8381224, 1.4534273, -3.5472908, 1.6696390, -4.5077615, 5.0007181
9: -1.8155354, 1.9517577, -2.2390659, 2.3143539, -4.1298895, 4.1908236

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2813745
time: 1.41 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2813745
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.5353227, 2.0650425, -2.0833631, 1.7416940, -4.2770166, 4.1484056
1: -2.0477524, 1.9247468, -1.6855249, 1.6037705, -3.6515229, 3.6102717
2: -2.5890172, 1.9487718, -2.0269756, 1.6622633, -4.2512803, 3.9757476
3: -2.8394561, 1.6611673, -2.3013508, 1.4037338, -4.2431898, 3.9625182
4: -2.9864419, 2.0390973, -2.4549296, 1.6684861, -4.6549282, 4.4940271
5: -2.4047542, 2.0988798, -1.9207706, 1.7953770, -4.2001314, 4.0196505
6: -2.1533420, 2.3689156, -1.7350885, 1.9856529, -4.1389952, 4.1040039
7: -2.5078759, 2.4724133, -2.0391674, 2.0143762, -4.5222521, 4.5115805
8: -3.5472908, 1.6696390, -2.8381224, 1.4534273, -5.0007181, 4.5077615
9: -2.2390659, 2.3143539, -1.8155354, 1.9517577, -4.1908236, 4.1298895

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2810406
time: 2.18 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2810406
time: 8.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.5353227, 2.0650425, -2.5324423, 2.0650425, -4.6003652, 4.5974846
1: -2.0477524, 1.9247468, -2.0455713, 1.9247468, -3.9724991, 3.9703181
2: -2.5890172, 1.9487718, -2.5890172, 1.9447163, -4.5337334, 4.5377889
3: -2.8394561, 1.6611673, -2.8364882, 1.6611673, -4.5006232, 4.4976554
4: -2.9864419, 2.0390973, -2.9829094, 2.0384564, -5.0248985, 5.0220070
5: -2.4047542, 2.0988798, -2.4047542, 2.0980911, -4.5028453, 4.5036340
6: -2.1533420, 2.3689156, -2.1503782, 2.3689156, -4.5222578, 4.5192938
7: -2.5078759, 2.4724133, -2.5078759, 2.4698517, -4.9777279, 4.9802895
8: -3.5472908, 1.6696390, -3.5451155, 1.6694697, -5.2167606, 5.2147546
9: -2.2390659, 2.3143539, -2.2390490, 2.3127527, -4.5518188, 4.5534029

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2810406
time: 2.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2810406
time: 5.19 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.0833631, 1.7416940, -2.3383520, 1.9373052, -4.0206680, 4.0800457
1: -1.6855249, 1.6037705, -1.8850859, 1.7810353, -3.4665604, 3.4888563
2: -2.0269756, 1.6622633, -2.3362648, 1.8184144, -3.8453901, 3.9985280
3: -2.3013508, 1.4037338, -2.5977340, 1.5467988, -3.8481498, 4.0014677
4: -2.4549296, 1.6684861, -2.7463758, 1.8715136, -4.3264432, 4.4148617
5: -1.9207706, 1.7953770, -2.1912882, 1.9615949, -3.8823657, 3.9866652
6: -1.7350885, 1.9856529, -1.9660486, 2.1979883, -3.9330769, 3.9517016
7: -2.0391674, 2.0143762, -2.2969217, 2.2653391, -4.3045063, 4.3112979
8: -2.8381224, 1.4534273, -3.2221632, 1.5760597, -4.4141822, 4.6755905
9: -1.8155354, 1.9517577, -2.0473509, 2.1442001, -3.9597354, 3.9991086

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2810408
time: 2.78 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804495, upper bound: 4.2810408
time: 2.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.0833631, 1.7416940, -2.7924316, 2.2534561, -4.3368192, 4.5341253
1: -1.6855249, 1.6037705, -2.2482793, 2.1008081, -3.7863331, 3.8520498
2: -2.0269756, 1.6622633, -2.9004233, 2.1074500, -4.1344256, 4.5626864
3: -2.3013508, 1.4037338, -3.1394947, 1.8035789, -4.1049299, 4.5432286
4: -2.4549296, 1.6684861, -3.2741139, 2.2465856, -4.7015152, 4.9426003
5: -1.9207706, 1.7953770, -2.6698141, 2.2723768, -4.1931477, 4.4651909
6: -1.7350885, 1.9856529, -2.3863821, 2.5809729, -4.3160615, 4.3720350
7: -2.0391674, 2.0143762, -2.7691746, 2.7219460, -4.7611132, 4.7835507
8: -2.8381224, 1.4534273, -3.9220977, 1.7971760, -4.6352983, 5.3755250
9: -1.8155354, 1.9517577, -2.4696856, 2.5105207, -4.3260560, 4.4214430

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2810408
time: 7.50 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804495, upper bound: 4.2810408
time: 2.46 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.5353227, 2.0650425, -2.3383520, 1.9373052, -4.4726276, 4.4033947
1: -2.0477524, 1.9247468, -1.8850859, 1.7810353, -3.8287878, 3.8098326
2: -2.5890172, 1.9487718, -2.3362648, 1.8184144, -4.4074316, 4.2850366
3: -2.8394561, 1.6611673, -2.5977340, 1.5467988, -4.3862548, 4.2589011
4: -2.9864419, 2.0390973, -2.7463758, 1.8715136, -4.8579555, 4.7854729
5: -2.4047542, 2.0988798, -2.1912882, 1.9615949, -4.3663492, 4.2901678
6: -2.1533420, 2.3689156, -1.9660486, 2.1979883, -4.3513303, 4.3349643
7: -2.5078759, 2.4724133, -2.2969217, 2.2653391, -4.7732153, 4.7693348
8: -3.5472908, 1.6696390, -3.2221632, 1.5760597, -5.1233506, 4.8918023
9: -2.2390659, 2.3143539, -2.0473509, 2.1442001, -4.3832660, 4.3617048

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2806894
time: 1.54 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803971, upper bound: 4.2806894
time: 2.15 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.5353227, 2.0650425, -2.7920773, 2.2534561, -4.7887788, 4.8571196
1: -2.0477524, 1.9247468, -2.2482793, 2.0991929, -4.1469450, 4.1730261
2: -2.5890172, 1.9487718, -2.9004233, 2.1066322, -4.6956491, 4.8491950
3: -2.8394561, 1.6611673, -3.1394947, 1.8029028, -4.6423588, 4.8006620
4: -2.9864419, 2.0390973, -3.2741139, 2.2432344, -5.2296762, 5.3132114
5: -2.4047542, 2.0988798, -2.6697662, 2.2723768, -4.6771307, 4.7686462
6: -2.1533420, 2.3689156, -2.3863821, 2.5790639, -4.7324057, 4.7552977
7: -2.5078759, 2.4724133, -2.7691746, 2.7217488, -5.2296247, 5.2415876
8: -3.5472908, 1.6696390, -3.9214466, 1.7971760, -5.3444667, 5.5910854
9: -2.2390659, 2.3143539, -2.4696856, 2.5095091, -4.7485752, 4.7840395

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2806894
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803971, upper bound: 4.2806894
time: 1.41 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.3383520, 1.9373052, -2.0833631, 1.7416940, -4.0800457, 4.0206680
1: -1.8850859, 1.7810353, -1.6855249, 1.6037705, -3.4888563, 3.4665604
2: -2.3362648, 1.8184144, -2.0269756, 1.6622633, -3.9985280, 3.8453901
3: -2.5977340, 1.5467988, -2.3013508, 1.4037338, -4.0014677, 3.8481498
4: -2.7463758, 1.8715136, -2.4549296, 1.6684861, -4.4148617, 4.3264432
5: -2.1912882, 1.9615949, -1.9207706, 1.7953770, -3.9866652, 3.8823657
6: -1.9660486, 2.1979883, -1.7350885, 1.9856529, -3.9517016, 3.9330769
7: -2.2969217, 2.2653391, -2.0391674, 2.0143762, -4.3112979, 4.3045063
8: -3.2221632, 1.5760597, -2.8381224, 1.4534273, -4.6755905, 4.4141822
9: -2.0473509, 2.1442001, -1.8155354, 1.9517577, -3.9991086, 3.9597354

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2813241
time: 3.05 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2813241
time: 1.75 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.3383520, 1.9373052, -2.3370433, 1.9372864, -4.2756386, 4.2743483
1: -1.8850859, 1.7810353, -1.8840588, 1.7807646, -3.6658506, 3.6650941
2: -2.3362648, 1.8184144, -2.3358686, 1.8162715, -4.1525364, 4.1542830
3: -2.5977340, 1.5467988, -2.5962441, 1.5466145, -4.1443486, 4.1430430
4: -2.7463758, 1.8715136, -2.7450030, 1.8712034, -4.6175795, 4.6165166
5: -2.1912882, 1.9615949, -2.1912735, 1.9602768, -4.1515651, 4.1528683
6: -1.9660486, 2.1979883, -1.9646537, 2.1978512, -4.1638999, 4.1626420
7: -2.2969217, 2.2653391, -2.2949510, 2.2643619, -4.5612836, 4.5602903
8: -3.2221632, 1.5760597, -3.2210562, 1.5754025, -4.7975655, 4.7971158
9: -2.0473509, 2.1442001, -2.0464134, 2.1435452, -4.1908960, 4.1906137

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2813241
time: 2.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2813241
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.7924316, 2.2534561, -2.0833631, 1.7416940, -4.5341253, 4.3368192
1: -2.2482793, 2.1008081, -1.6855249, 1.6037705, -3.8520498, 3.7863331
2: -2.9004233, 2.1074500, -2.0269756, 1.6622633, -4.5626864, 4.1344256
3: -3.1394947, 1.8035789, -2.3013508, 1.4037338, -4.5432286, 4.1049299
4: -3.2741139, 2.2465856, -2.4549296, 1.6684861, -4.9426003, 4.7015152
5: -2.6698141, 2.2723768, -1.9207706, 1.7953770, -4.4651909, 4.1931477
6: -2.3863821, 2.5809729, -1.7350885, 1.9856529, -4.3720350, 4.3160615
7: -2.7691746, 2.7219460, -2.0391674, 2.0143762, -4.7835507, 4.7611132
8: -3.9220977, 1.7971760, -2.8381224, 1.4534273, -5.3755250, 4.6352983
9: -2.4696856, 2.5105207, -1.8155354, 1.9517577, -4.4214430, 4.3260560

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2804110
time: 1.66 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2804183
time: 2.80 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.7924316, 2.2534561, -2.3370433, 1.9372864, -4.7297182, 4.5904994
1: -2.2482793, 2.1008081, -1.8840588, 1.7807646, -4.0290442, 3.9848671
2: -2.9004233, 2.1074500, -2.3358686, 1.8162715, -4.7166948, 4.4433184
3: -3.1394947, 1.8035789, -2.5962441, 1.5466145, -4.6861091, 4.3998232
4: -3.2741139, 2.2465856, -2.7450030, 1.8712034, -5.1453171, 4.9915886
5: -2.6698141, 2.2723768, -2.1912735, 1.9602768, -4.6300907, 4.4636502
6: -2.3863821, 2.5809729, -1.9646537, 2.1978512, -4.5842333, 4.5456266
7: -2.7691746, 2.7219460, -2.2949510, 2.2643619, -5.0335364, 5.0168972
8: -3.9220977, 1.7971760, -3.2210562, 1.5754025, -5.4975004, 5.0182323
9: -2.4696856, 2.5105207, -2.0464134, 2.1435452, -4.6132307, 4.5569344

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2804110
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2804183
time: 2.08 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.3383520, 1.9373052, -2.5361462, 2.0662789, -4.4046307, 4.4734516
1: -1.8850859, 1.7810353, -2.0492849, 1.9250532, -3.8101392, 3.8303204
2: -2.3362648, 1.8184144, -2.5908852, 1.9513563, -4.2876210, 4.4092999
3: -2.5977340, 1.5467988, -2.8413615, 1.6620383, -4.2597723, 4.3881602
4: -2.7463758, 1.8715136, -2.9868734, 2.0407581, -4.7871342, 4.8583870
5: -2.1912882, 1.9615949, -2.4049091, 2.1016920, -4.2929802, 4.3665042
6: -1.9660486, 2.1979883, -2.1545725, 2.3704898, -4.3365383, 4.3525610
7: -2.2969217, 2.2653391, -2.5102975, 2.4734833, -4.7704048, 4.7756367
8: -3.2221632, 1.5760597, -3.5472908, 1.6748905, -4.8970537, 5.1233506
9: -2.0473509, 2.1442001, -2.2410514, 2.3170941, -4.3644447, 4.3852515

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2808330
time: 1.89 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804238, upper bound: 4.2808330
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.3383520, 1.9373052, -2.7867386, 2.2534561, -4.5918083, 4.7240438
1: -1.8850859, 1.7810353, -2.2466912, 2.0991929, -3.9842787, 4.0277267
2: -2.3362648, 1.8184144, -2.8959122, 2.1066322, -4.4428968, 4.7143269
3: -2.5977340, 1.5467988, -3.1337678, 1.8029028, -4.4006367, 4.6805668
4: -2.7463758, 1.8715136, -3.2741139, 2.2414691, -4.9878449, 5.1456275
5: -2.1912882, 1.9615949, -2.6697662, 2.2673578, -4.4586458, 4.6313610
6: -1.9660486, 2.1979883, -2.3820064, 2.5790639, -4.5451126, 4.5799947
7: -2.2969217, 2.2653391, -2.7657773, 2.7217488, -5.0186706, 5.0311165
8: -3.2221632, 1.5760597, -3.9204443, 1.7971760, -5.0193391, 5.4965038
9: -2.0473509, 2.1442001, -2.4696856, 2.5079243, -4.5552750, 4.6138859

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2808330
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804238, upper bound: 4.2808330
time: 1.85 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.7920773, 2.2534561, -2.5361462, 2.0662789, -4.8583565, 4.7896023
1: -2.2482793, 2.0991929, -2.0492849, 1.9250532, -4.1733327, 4.1484776
2: -2.9004233, 2.1066322, -2.5908852, 1.9513563, -4.8517795, 4.6975174
3: -3.1394947, 1.8029028, -2.8413615, 1.6620383, -4.8015327, 4.6442642
4: -3.2741139, 2.2432344, -2.9868734, 2.0407581, -5.3148718, 5.2301078
5: -2.6697662, 2.2723768, -2.4049091, 2.1016920, -4.7714581, 4.6772861
6: -2.3863821, 2.5790639, -2.1545725, 2.3704898, -4.7568722, 4.7336364
7: -2.7691746, 2.7217488, -2.5102975, 2.4734833, -5.2426577, 5.2320461
8: -3.9214466, 1.7971760, -3.5472908, 1.6748905, -5.5963373, 5.3444667
9: -2.4696856, 2.5095091, -2.2410514, 2.3170941, -4.7867794, 4.7505608

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2803347
time: 3.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803456, upper bound: 4.2803428
time: 1.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.7920773, 2.2534561, -2.7867386, 2.2534561, -5.0455332, 5.0401945
1: -2.2482793, 2.0991929, -2.2466912, 2.0991929, -4.3474722, 4.3458843
2: -2.9004233, 2.1066322, -2.8959122, 2.1066322, -5.0070553, 5.0025444
3: -3.1394947, 1.8029028, -3.1337678, 1.8029028, -4.9423976, 4.9366708
4: -3.2741139, 2.2432344, -3.2741139, 2.2414691, -5.5155830, 5.5173483
5: -2.6697662, 2.2723768, -2.6697662, 2.2673578, -4.9371243, 4.9421430
6: -2.3863821, 2.5790639, -2.3820064, 2.5790639, -4.9654460, 4.9610701
7: -2.7691746, 2.7217488, -2.7657773, 2.7217488, -5.4909234, 5.4875259
8: -3.9214466, 1.7971760, -3.9204443, 1.7971760, -5.7186227, 5.7176204
9: -2.4696856, 2.5095091, -2.4696856, 2.5079243, -4.9776096, 4.9791946

Time for backsubstitution: 1.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2803347
time: 3.24 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803456, upper bound: 4.2803428
time: 2.13 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 7.57 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2813745
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2813745
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2813745
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2813745
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2810406
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2810406
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2810406
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2810406
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2810408
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2804495, upper bound: 4.2810408
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2810408
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2804495, upper bound: 4.2810408
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2806894
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2803971, upper bound: 4.2806894
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2806894
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2803971, upper bound: 4.2806894
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2813241
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2813241
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2813241
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2813241
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2804110
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2804183
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2804110
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2804183
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2808330
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2804238, upper bound: 4.2808330
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2808330
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2804238, upper bound: 4.2808330
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2803347
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2803456, upper bound: 4.2803428
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2803347
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 7.57
Output dim: 8, lower bound: -4.2803456, upper bound: 4.2803428

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.0130851, 1.6872475, -3.3018303, 3.3923271
1: -1.3168527, 1.2810533, -1.6298018, 1.5546049, -2.8714576, 2.9108551
2: -1.4648219, 1.3692248, -1.9418502, 1.6179225, -3.0827446, 3.3110750
3: -1.7454548, 1.1351854, -2.2191486, 1.3636807, -3.1091356, 3.3543339
4: -1.9133011, 1.2971265, -2.3733096, 1.6121153, -3.5254164, 3.6704361
5: -1.4184097, 1.4522150, -1.8448559, 1.7453510, -3.1637607, 3.2970710
6: -1.3162330, 1.5626514, -1.6696838, 1.9223180, -3.2385511, 3.2323351
7: -1.5742366, 1.5566748, -1.9691199, 1.9452224, -3.5194590, 3.5257947
8: -2.0757601, 1.2396940, -2.7285669, 1.4158745, -3.4916346, 3.9682608
9: -1.3722337, 1.5473963, -1.7497044, 1.8911940, -3.2634277, 3.2971005

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2812689, upper bound: 4.2812689
time: 1.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2812689, upper bound: 4.2817331
time: 1.95 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.0526538, 1.7183356, -3.5615640, 3.6115804
1: -1.4958531, 1.4391533, -1.6611110, 1.5821961, -3.0780492, 3.1002643
2: -1.7368103, 1.5125700, -1.9896408, 1.6427830, -3.3795934, 3.5022109
3: -2.0219579, 1.2669443, -2.2653165, 1.3861984, -3.4081564, 3.5322609
4: -2.1765027, 1.4790530, -2.4190600, 1.6437674, -3.8202701, 3.8981130
5: -1.6626878, 1.6230266, -1.8875427, 1.7734259, -3.4361138, 3.5105693
6: -1.5174071, 1.7687372, -1.7064879, 1.9581484, -3.4755554, 3.4752250
7: -1.8011651, 1.7800044, -2.0083580, 1.9840245, -3.7851896, 3.7883625
8: -2.4602904, 1.3307571, -2.7898638, 1.4369512, -3.8972416, 4.1206207
9: -1.5911342, 1.7429529, -1.7866650, 1.9250674, -3.5162015, 3.5296178

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2817331, upper bound: 4.2812689
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2817331, upper bound: 4.2817331
time: 3.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.4662311, 2.0109935, -3.6255760, 3.8454731
1: -1.3168527, 1.2810533, -1.9928600, 1.8762439, -3.1930966, 3.2739134
2: -1.4648219, 1.3692248, -2.5050306, 1.9048541, -3.3696761, 3.8742554
3: -1.7454548, 1.1351854, -2.7587161, 1.6219858, -3.3674407, 3.8939013
4: -1.9133011, 1.2971265, -2.9062030, 1.9833999, -3.8967009, 4.2033296
5: -1.4184097, 1.4522150, -2.3304517, 2.0497375, -3.4681473, 3.7826667
6: -1.3162330, 1.5626514, -2.0887394, 2.3062692, -3.6225023, 3.6513908
7: -1.5742366, 1.5566748, -2.4385178, 2.4043698, -3.9786065, 3.9951925
8: -2.0757601, 1.2396940, -3.4403281, 1.6306739, -3.7064340, 4.6800222
9: -1.3722337, 1.5473963, -2.1748543, 2.2546911, -3.6269250, 3.7222505

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2807852
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2813745
time: 2.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.5053649, 2.0418644, -3.8850927, 4.0642915
1: -1.4958531, 1.4391533, -2.0238945, 1.9036586, -3.3995118, 3.4630480
2: -1.7368103, 1.5125700, -2.5524821, 1.9296215, -3.6664319, 4.0650520
3: -2.0219579, 1.2669443, -2.8043442, 1.6441460, -3.6661038, 4.0712886
4: -2.1765027, 1.4790530, -2.9514761, 2.0148625, -4.1913652, 4.4305291
5: -1.6626878, 1.6230266, -2.3725250, 2.0775080, -3.7401958, 3.9955516
6: -1.5174071, 1.7687372, -2.1253297, 2.3418822, -3.8592892, 3.8940668
7: -1.8011651, 1.7800044, -2.4776349, 2.4428124, -4.2439775, 4.2576394
8: -2.4602904, 1.3307571, -3.5006123, 1.6527987, -4.1130891, 4.8313694
9: -1.5911342, 1.7429529, -2.2111189, 2.2882407, -3.8793750, 3.9540720

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2807852
time: 1.78 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2813745
time: 1.56 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -2.0130851, 1.6872475, -3.6909952, 3.6649480
1: -1.6264559, 1.5585181, -1.6298018, 1.5546049, -3.1810608, 3.1883197
2: -1.9452751, 1.6141748, -1.9418502, 1.6179225, -3.5631976, 3.5560250
3: -2.2200139, 1.3563321, -2.2191486, 1.3636807, -3.5836945, 3.5754807
4: -2.3756607, 1.6093513, -2.3733096, 1.6121153, -3.9877758, 3.9826608
5: -1.8359622, 1.7206392, -1.8448559, 1.7453510, -3.5813131, 3.5654950
6: -1.6592586, 1.8914390, -1.6696838, 1.9223180, -3.5815766, 3.5611229
7: -1.9775292, 1.9471791, -1.9691199, 1.9452224, -3.9227514, 3.9162989
8: -2.7113912, 1.3773797, -2.7285669, 1.4158745, -4.1272659, 4.1059465
9: -1.7459316, 1.8566196, -1.7497044, 1.8911940, -3.6371255, 3.6063240

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2807852, upper bound: 4.2806202
time: 2.24 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2807852, upper bound: 4.2810987
time: 1.48 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -2.0526538, 1.7183356, -3.9835260, 3.9085281
1: -1.8323848, 1.7356169, -1.6611110, 1.5821961, -3.4145808, 3.3967280
2: -2.2596996, 1.7757981, -1.9896408, 1.6427830, -3.9024825, 3.7654390
3: -2.5229864, 1.5075209, -2.2653165, 1.3861984, -3.9091849, 3.7728374
4: -2.6710010, 1.8203437, -2.4190600, 1.6437674, -4.3147683, 4.2394037
5: -2.1143703, 1.9067429, -1.8875427, 1.7734259, -3.8877964, 3.7942858
6: -1.9004462, 2.1250553, -1.7064879, 1.9581484, -3.8585944, 3.8315432
7: -2.2354999, 2.2050855, -2.0083580, 1.9840245, -4.2195244, 4.2134438
8: -3.1254687, 1.5177147, -2.7898638, 1.4369512, -4.5624199, 4.3075786
9: -1.9864624, 2.0786352, -1.7866650, 1.9250674, -3.9115298, 3.8653002

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813745, upper bound: 4.2806202
time: 2.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813745, upper bound: 4.2810987
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -2.4639292, 2.0109935, -4.0147409, 4.1157918
1: -1.6264559, 1.5585181, -1.9911249, 1.8762439, -3.5026999, 3.5496430
2: -1.9452751, 1.6141748, -2.5050306, 1.9016126, -3.8468876, 4.1192055
3: -2.2200139, 1.3563321, -2.7563460, 1.6219858, -3.8419995, 4.1126781
4: -2.3756607, 1.6093513, -2.9033670, 1.9829023, -4.3585629, 4.5127182
5: -1.8359622, 1.7206392, -2.3304517, 2.0491028, -3.8850651, 4.0510912
6: -1.6592586, 1.8914390, -2.0863752, 2.3062692, -3.9655278, 3.9778142
7: -1.9775292, 1.9471791, -2.4385178, 2.4023345, -4.3798637, 4.3856969
8: -2.7113912, 1.3773797, -3.4385791, 1.6305426, -4.3419337, 4.8159590
9: -1.7459316, 1.8566196, -2.1748395, 2.2534423, -3.9993739, 4.0314589

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2804938
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2810406
time: 1.55 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -2.5027354, 2.0418644, -4.3070545, 4.3586097
1: -1.8323848, 1.7356169, -2.0219073, 1.9036586, -3.7360435, 3.7575243
2: -2.2596996, 1.7757981, -2.5524821, 1.9259192, -4.1856189, 4.3282804
3: -2.5229864, 1.5075209, -2.8016360, 1.6441460, -4.1671324, 4.3091569
4: -2.6710010, 1.8203437, -2.9482486, 2.0142832, -4.6852841, 4.7685924
5: -2.1143703, 1.9067429, -2.3725250, 2.0767868, -4.1911573, 4.2792678
6: -1.9004462, 2.1250553, -2.1226268, 2.3418822, -4.2423286, 4.2476821
7: -2.2354999, 2.2050855, -2.4776349, 2.4404781, -4.6759777, 4.6827202
8: -3.1254687, 1.5177147, -3.4986246, 1.6526455, -4.7781143, 5.0163393
9: -1.9864624, 2.0786352, -2.2111032, 2.2867901, -4.2732525, 4.2897387

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2804938
time: 1.48 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2810406
time: 2.68 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.2673373, 1.8820553, -3.4966378, 3.6465793
1: -1.3168527, 1.2810533, -1.8285353, 1.7312907, -3.0481434, 3.1095886
2: -1.4648219, 1.3692248, -2.2500448, 1.7725872, -3.2374091, 3.6192696
3: -1.7454548, 1.1351854, -2.5144083, 1.5065919, -3.2520466, 3.6495938
4: -1.9133011, 1.2971265, -2.6638618, 1.8143563, -3.7276573, 3.9609883
5: -1.4184097, 1.4522150, -2.1149516, 1.9105868, -3.3289967, 3.5671666
6: -1.3162330, 1.5626514, -1.8994002, 2.1338038, -3.4500370, 3.4620516
7: -1.5742366, 1.5566748, -2.2254615, 2.1953452, -3.7695818, 3.7821364
8: -2.0757601, 1.2396940, -3.1121616, 1.5359800, -3.6117401, 4.3518558
9: -1.3722337, 1.5473963, -1.9809750, 2.0827076, -3.4549413, 3.5283713

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2809428, upper bound: 4.2809772
time: 2.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2809428, upper bound: 4.2815140
time: 1.50 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.3063719, 1.9128499, -3.7560782, 3.8652987
1: -1.4958531, 1.4391533, -1.8595669, 1.7585385, -3.2543917, 3.2987204
2: -1.7368103, 1.5125700, -2.2972462, 1.7977023, -3.5345125, 3.8098164
3: -2.0219579, 1.2669443, -2.5600564, 1.5286677, -3.5506256, 3.8270006
4: -2.1765027, 1.4790530, -2.7090247, 1.8456672, -4.0221701, 4.1880779
5: -1.6626878, 1.6230266, -2.1569529, 1.9386665, -3.6013541, 3.7799795
6: -1.5174071, 1.7687372, -1.9360446, 2.1692541, -3.6866612, 3.7047818
7: -1.8011651, 1.7800044, -2.2645044, 2.2336624, -4.0348272, 4.0445089
8: -2.4602904, 1.3307571, -3.1722569, 1.5581231, -4.0184135, 4.5030141
9: -1.5911342, 1.7429529, -2.0173056, 2.1164427, -3.7075768, 3.7602587

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813755, upper bound: 4.2809772
time: 1.99 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813755, upper bound: 4.2815140
time: 1.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.7241497, 2.2005155, -3.8150983, 4.1033916
1: -1.3168527, 1.2810533, -2.1940517, 2.0531697, -3.3700223, 3.4751050
2: -1.4648219, 1.3692248, -2.8171077, 2.0638189, -3.5286407, 4.1863327
3: -1.7454548, 1.1351854, -3.0593657, 1.7647923, -3.5102472, 4.1945510
4: -1.9133011, 1.2971265, -3.1956828, 2.1906197, -4.1039209, 4.4928093
5: -1.4184097, 1.4522150, -2.5973604, 2.2232342, -3.6416440, 4.0495753
6: -1.3162330, 1.5626514, -2.3223600, 2.5189800, -3.8352132, 3.8850114
7: -1.5742366, 1.5566748, -2.7002928, 2.6550069, -4.2292433, 4.2569675
8: -2.0757601, 1.2396940, -3.8174546, 1.7576821, -3.8334422, 5.0571485
9: -1.3722337, 1.5473963, -2.4063938, 2.4510696, -3.8233032, 3.9537902

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2802009
time: 1.86 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2810408
time: 1.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.7606716, 2.2292750, -4.0725036, 4.3195982
1: -1.4958531, 1.4391533, -2.2230182, 2.0786042, -3.5744572, 3.6621714
2: -1.7368103, 1.5125700, -2.8615317, 2.0871043, -3.8239145, 4.3741016
3: -2.0219579, 1.2669443, -3.1020985, 1.7855471, -3.8075051, 4.3690429
4: -2.1765027, 1.4790530, -3.2374682, 2.2204823, -4.3969851, 4.7165213
5: -1.6626878, 1.6230266, -2.6362038, 2.2496858, -3.9123735, 4.2592306
6: -1.5174071, 1.7687372, -2.3567095, 2.5526044, -4.0700116, 4.1254468
7: -1.8011651, 1.7800044, -2.7369266, 2.6907096, -4.4918747, 4.5169311
8: -2.4602904, 1.3307571, -3.8731475, 1.7791513, -4.2394419, 5.2039046
9: -1.5911342, 1.7429529, -2.4401627, 2.4830203, -4.0741544, 4.1831155

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804438, upper bound: 4.2802009
time: 2.03 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804438, upper bound: 4.2810408
time: 1.94 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -2.2673373, 1.8820553, -3.8858030, 3.9191999
1: -1.6264559, 1.5585181, -1.8285353, 1.7312907, -3.3577466, 3.3870535
2: -1.9452751, 1.6141748, -2.2500448, 1.7725872, -3.7178621, 3.8642197
3: -2.2200139, 1.3563321, -2.5144083, 1.5065919, -3.7266059, 3.8707404
4: -2.3756607, 1.6093513, -2.6638618, 1.8143563, -4.1900167, 4.2732129
5: -1.8359622, 1.7206392, -2.1149516, 1.9105868, -3.7465491, 3.8355908
6: -1.6592586, 1.8914390, -1.8994002, 2.1338038, -3.7930624, 3.7908392
7: -1.9775292, 1.9471791, -2.2254615, 2.1953452, -4.1728745, 4.1726408
8: -2.7113912, 1.3773797, -3.1121616, 1.5359800, -4.2473712, 4.4895411
9: -1.7459316, 1.8566196, -1.9809750, 2.0827076, -3.8286393, 3.8375945

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803228, upper bound: 4.2801701
time: 2.03 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803228, upper bound: 4.2807760
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -2.3063719, 1.9128499, -4.1780405, 4.1622462
1: -1.8323848, 1.7356169, -1.8595669, 1.7585385, -3.5909233, 3.5951838
2: -2.2596996, 1.7757981, -2.2972462, 1.7977023, -4.0574017, 4.0730443
3: -2.5229864, 1.5075209, -2.5600564, 1.5286677, -4.0516539, 4.0675774
4: -2.6710010, 1.8203437, -2.7090247, 1.8456672, -4.5166683, 4.5293684
5: -2.1143703, 1.9067429, -2.1569529, 1.9386665, -4.0530367, 4.0636959
6: -1.9004462, 2.1250553, -1.9360446, 2.1692541, -4.0697002, 4.0611000
7: -2.2354999, 2.2050855, -2.2645044, 2.2336624, -4.4691620, 4.4695902
8: -3.1254687, 1.5177147, -3.1722569, 1.5581231, -4.6835918, 4.6899719
9: -1.9864624, 2.0786352, -2.0173056, 2.1164427, -4.1029053, 4.0959406

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2809117, upper bound: 4.2801701
time: 1.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2809117, upper bound: 4.2807760
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -2.7238591, 2.2005155, -4.2042632, 4.3757219
1: -1.6264559, 1.5585181, -2.1940517, 2.0518489, -3.6783047, 3.7525697
2: -1.9452751, 1.6141748, -2.8171077, 2.0631495, -4.0084248, 4.4312825
3: -2.2200139, 1.3563321, -3.0593657, 1.7642397, -3.9842534, 4.4156981
4: -2.3756607, 1.6093513, -3.1956828, 2.1878791, -4.5635395, 4.8050342
5: -1.8359622, 1.7206392, -2.5973217, 2.2232342, -4.0591965, 4.3179607
6: -1.6592586, 1.8914390, -2.3223600, 2.5174184, -4.1766768, 4.2137990
7: -1.9775292, 1.9471791, -2.7002928, 2.6548438, -4.6323729, 4.6474719
8: -2.7113912, 1.3773797, -3.8169200, 1.7576821, -4.4690733, 5.1942997
9: -1.7459316, 1.8566196, -2.4063938, 2.4502387, -4.1961703, 4.2630134

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2799326
time: 2.23 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2806894
time: 3.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -2.7603467, 2.2292750, -4.4944654, 4.6162210
1: -1.8323848, 1.7356169, -2.2230182, 2.0771291, -3.9095140, 3.9586351
2: -2.2596996, 1.7757981, -2.8615317, 2.0863564, -4.3460560, 4.6373301
3: -2.5229864, 1.5075209, -3.1020985, 1.7849278, -4.3079143, 4.6096191
4: -2.6710010, 1.8203437, -3.2374682, 2.2174199, -4.8884211, 5.0578117
5: -2.1143703, 1.9067429, -2.6361611, 2.2496858, -4.3640561, 4.5429039
6: -1.9004462, 2.1250553, -2.3567095, 2.5508597, -4.4513059, 4.4817648
7: -2.2354999, 2.2050855, -2.7369266, 2.6905286, -4.9260283, 4.9420118
8: -3.1254687, 1.5177147, -3.8725517, 1.7791513, -4.9046202, 5.3902664
9: -1.9864624, 2.0786352, -2.4401627, 2.4820931, -4.4685555, 4.5187979

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803913, upper bound: 4.2799326
time: 1.58 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803913, upper bound: 4.2806894
time: 1.78 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.8458550, 1.5601518, -2.0130851, 1.6872475, -3.5331025, 3.5732369
1: -1.4973698, 1.4441805, -1.6298018, 1.5546049, -3.0519748, 3.0739822
2: -1.7430093, 1.5131910, -1.9418502, 1.6179225, -3.3609319, 3.4550412
3: -2.0262935, 1.2658768, -2.2191486, 1.3636807, -3.3899741, 3.4850254
4: -2.1838734, 1.4798396, -2.3733096, 1.6121153, -3.7959886, 3.8531492
5: -1.6644652, 1.6105781, -1.8448559, 1.7453510, -3.4098163, 3.4554338
6: -1.5194894, 1.7553332, -1.6696838, 1.9223180, -3.4418073, 3.4250169
7: -1.8082078, 1.7851741, -1.9691199, 1.9452224, -3.7534301, 3.7542939
8: -2.4493060, 1.3118429, -2.7285669, 1.4158745, -3.8651805, 4.0404100
9: -1.5935162, 1.7208647, -1.7497044, 1.8911940, -3.4847102, 3.4705691

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2809772, upper bound: 4.2809428
time: 2.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2809772, upper bound: 4.2813755
time: 2.77 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.0812759, 1.7388462, -2.0526538, 1.7183356, -3.7996116, 3.7915001
1: -1.6805236, 1.6002891, -1.6611110, 1.5821961, -3.2627196, 3.2614002
2: -2.0233788, 1.6545870, -1.9896408, 1.6427830, -3.6661620, 3.6442280
3: -2.2960305, 1.4003358, -2.2653165, 1.3861984, -3.6822290, 3.6656523
4: -2.4466107, 1.6642282, -2.4190600, 1.6437674, -4.0903778, 4.0832882
5: -1.9141870, 1.7769457, -1.8875427, 1.7734259, -3.6876130, 3.6644883
6: -1.7262683, 1.9657857, -1.7064879, 1.9581484, -3.6844168, 3.6722736
7: -2.0387635, 2.0116763, -2.0083580, 1.9840245, -4.0227880, 4.0200343
8: -2.8200552, 1.4329078, -2.7898638, 1.4369512, -4.2570066, 4.2227716
9: -1.8069404, 1.9201001, -1.7866650, 1.9250674, -3.7320080, 3.7067652

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815140, upper bound: 4.2809428
time: 1.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2815140, upper bound: 4.2813755
time: 1.60 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.8458550, 1.5601518, -2.2668567, 1.8820553, -3.7279103, 3.8270085
1: -1.4973698, 1.4441805, -1.8281972, 1.7311931, -3.2285628, 3.2723777
2: -1.7430093, 1.5131910, -2.2498987, 1.7719265, -3.5149360, 3.7630897
3: -2.0262935, 1.2658768, -2.5139344, 1.5065165, -3.5328100, 3.7798111
4: -2.1838734, 1.4798396, -2.6632805, 1.8142456, -3.9981189, 4.1431198
5: -1.6644652, 1.6105781, -2.1149516, 1.9102659, -3.5747311, 3.7255297
6: -1.5194894, 1.7553332, -1.8989302, 2.1337605, -3.6532497, 3.6542635
7: -1.8082078, 1.7851741, -2.2247384, 2.1949542, -4.0031619, 4.0099125
8: -2.4493060, 1.3118429, -3.1116753, 1.5359468, -3.9852529, 4.4235182
9: -1.5935162, 1.7208647, -1.9807469, 2.0825043, -3.6760206, 3.7016115

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2808198
time: 2.51 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2813241
time: 10.34 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.0812759, 1.7388462, -2.3054376, 1.9128499, -3.9941258, 4.0442839
1: -1.6805236, 1.6002891, -1.8588532, 1.7583623, -3.4388859, 3.4591422
2: -2.0233788, 1.6545870, -2.2969868, 1.7962288, -3.8196077, 3.9515738
3: -2.2960305, 1.4003358, -2.5590541, 1.5285331, -3.8245635, 3.9593899
4: -2.4466107, 1.6642282, -2.7080207, 1.8454505, -4.2920613, 4.3722486
5: -1.9141870, 1.7769457, -2.1569529, 1.9378042, -3.8519912, 3.9338984
6: -1.7262683, 1.9657857, -1.9350677, 2.1691766, -3.8954449, 3.9008534
7: -2.0387635, 2.0116763, -2.2631321, 2.2329767, -4.2717400, 4.2748084
8: -2.8200552, 1.4329078, -3.1714318, 1.5578244, -4.3778796, 4.6043396
9: -1.8069404, 1.9201001, -2.0167232, 2.1160355, -3.9229760, 3.9368234

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2808198
time: 1.80 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2813241
time: 2.25 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.2618871, 1.8464744, -2.0130851, 1.6872475, -3.9491346, 3.8595595
1: -1.8277041, 1.7360274, -1.6298018, 1.5546049, -3.3823090, 3.3658290
2: -2.2569242, 1.7718315, -1.9418502, 1.6179225, -3.8748467, 3.7136817
3: -2.5204260, 1.5026162, -2.2191486, 1.3636807, -3.8841066, 3.7217648
4: -2.6700065, 1.8147115, -2.3733096, 1.6121153, -4.2821217, 4.1880212
5: -2.1106806, 1.8917195, -1.8448559, 1.7453510, -3.8560314, 3.7365754
6: -1.8933653, 2.1022105, -1.6696838, 1.9223180, -3.8156834, 3.7718945
7: -2.2367978, 2.2034998, -1.9691199, 1.9452224, -4.1820202, 4.1726198
8: -3.1042314, 1.4917414, -2.7285669, 1.4158745, -4.5201058, 4.2203083
9: -1.9823278, 2.0509677, -1.7497044, 1.8911940, -3.8735218, 3.8006721

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2802009, upper bound: 4.2799724
time: 2.29 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2802009, upper bound: 4.2804438
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.5087495, 2.0368962, -2.0526538, 1.7183356, -4.2270851, 4.0895500
1: -2.0226388, 1.9026828, -1.6611110, 1.5821961, -3.6048350, 3.5637937
2: -2.5533326, 1.9254339, -1.9896408, 1.6427830, -4.1961155, 3.9150748
3: -2.8057382, 1.6424177, -2.2653165, 1.3861984, -4.1919365, 3.9077342
4: -2.9470456, 2.0133429, -2.4190600, 1.6437674, -4.5908127, 4.4324026
5: -2.3690577, 2.0694890, -1.8875427, 1.7734259, -4.1424837, 3.9570317
6: -2.1209738, 2.3267791, -1.7064879, 1.9581484, -4.0791221, 4.0332670
7: -2.4814527, 2.4430037, -2.0083580, 1.9840245, -4.4654770, 4.4513617
8: -3.4850137, 1.6349620, -2.7898638, 1.4369512, -4.9219646, 4.4248257
9: -2.2063260, 2.2638803, -1.7866650, 1.9250674, -4.1313934, 4.0505452

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810408, upper bound: 4.2799724
time: 1.93 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2810408, upper bound: 4.2804495
time: 1.62 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.2618871, 1.8464744, -2.2668567, 1.8820553, -4.1439424, 4.1133308
1: -1.8277041, 1.7360274, -1.8281972, 1.7311931, -3.5588970, 3.5642247
2: -2.2569242, 1.7718315, -2.2498987, 1.7719265, -4.0288506, 4.0217304
3: -2.5204260, 1.5026162, -2.5139344, 1.5065165, -4.0269423, 4.0165505
4: -2.6700065, 1.8147115, -2.6632805, 1.8142456, -4.4842520, 4.4779921
5: -2.1106806, 1.8917195, -2.1149516, 1.9102659, -4.0209465, 4.0066710
6: -1.8933653, 2.1022105, -1.8989302, 2.1337605, -4.0271258, 4.0011406
7: -2.2367978, 2.2034998, -2.2247384, 2.1949542, -4.4317522, 4.4282379
8: -3.1042314, 1.4917414, -3.1116753, 1.5359468, -4.6401782, 4.6034164
9: -1.9823278, 2.0509677, -1.9807469, 2.0825043, -4.0648322, 4.0317144

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2798548
time: 1.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2804110
time: 1.69 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.5087495, 2.0368962, -2.3054376, 1.9128499, -4.4215994, 4.3423338
1: -2.0226388, 1.9026828, -1.8588532, 1.7583623, -3.7810011, 3.7615361
2: -2.5533326, 1.9254339, -2.2969868, 1.7962288, -4.3495612, 4.2224207
3: -2.8057382, 1.6424177, -2.5590541, 1.5285331, -4.3342714, 4.2014718
4: -2.9470456, 2.0133429, -2.7080207, 1.8454505, -4.7924962, 4.7213635
5: -2.3690577, 2.0694890, -2.1569529, 1.9378042, -4.3068619, 4.2264419
6: -2.1209738, 2.3267791, -1.9350677, 2.1691766, -4.2901506, 4.2618465
7: -2.4814527, 2.4430037, -2.2631321, 2.2329767, -4.7144294, 4.7061357
8: -3.4850137, 1.6349620, -3.1714318, 1.5578244, -5.0428381, 4.8063936
9: -2.2063260, 2.2638803, -2.0167232, 2.1160355, -4.3223615, 4.2806034

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2798548
time: 2.00 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2804183
time: 1.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.8458550, 1.5601518, -2.4668553, 2.0117130, -3.8575680, 4.0270071
1: -1.4973698, 1.4441805, -1.9938930, 1.8764622, -3.3738320, 3.4380736
2: -1.7430093, 1.5131910, -2.5061800, 1.9068102, -3.6498194, 4.0193710
3: -2.0262935, 1.2658768, -2.7601337, 1.6224308, -3.6487243, 4.0260105
4: -2.1838734, 1.4798396, -2.9065208, 1.9842879, -4.1681614, 4.3863602
5: -1.6644652, 1.6105781, -2.3305614, 2.0518069, -3.7162721, 3.9411395
6: -1.5194894, 1.7553332, -2.0896711, 2.3072011, -3.8266907, 3.8450043
7: -1.8082078, 1.7851741, -2.4402728, 2.4050071, -4.2132149, 4.2254467
8: -2.4493060, 1.3118429, -3.4403281, 1.6342632, -4.0835690, 4.7521710
9: -1.5935162, 1.7208647, -2.1763244, 2.2565827, -3.8500991, 3.8971891

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2801701, upper bound: 4.2803228
time: 1.72 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2801701, upper bound: 4.2809117
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -2.0812759, 1.7388462, -2.5061016, 2.0428770, -4.1241531, 4.2449479
1: -1.6805236, 1.6002891, -2.0252109, 1.9039276, -3.5844512, 3.6255000
2: -2.0233788, 1.6545870, -2.5540395, 1.9319321, -3.9553108, 4.2086267
3: -2.2960305, 1.4003358, -2.8060374, 1.6448342, -3.9408646, 4.2063732
4: -2.4466107, 1.6642282, -2.9518602, 2.0161910, -4.4628019, 4.6160884
5: -1.9141870, 1.7769457, -2.3726606, 2.0799973, -3.9941843, 4.1496062
6: -1.7262683, 1.9657857, -2.1264291, 2.3431785, -4.0694466, 4.0922146
7: -2.0387635, 2.0116763, -2.4797633, 2.4436944, -4.4824581, 4.4914398
8: -2.8200552, 1.4329078, -3.5006123, 1.6573261, -4.4773812, 4.9335203
9: -1.8069404, 1.9201001, -2.2128787, 2.2906084, -4.0975490, 4.1329789

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2807760, upper bound: 4.2803228
time: 1.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2807760, upper bound: 4.2809117
time: 1.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.8458550, 1.5601518, -2.7192976, 2.2005155, -4.0463705, 4.2794495
1: -1.4973698, 1.4441805, -2.1926937, 2.0518489, -3.5492187, 3.6368742
2: -1.7430093, 1.5131910, -2.8132589, 2.0631495, -3.8061588, 4.3264499
3: -2.0262935, 1.2658768, -3.0544732, 1.7642397, -3.7905331, 4.3203497
4: -2.1838734, 1.4798396, -3.1956828, 2.1863675, -4.3702412, 4.6755223
5: -1.6644652, 1.6105781, -2.5973217, 2.2189379, -3.8834031, 4.2079000
6: -1.5194894, 1.7553332, -2.3186181, 2.5174184, -4.0369077, 4.0739512
7: -1.8082078, 1.7851741, -2.6973901, 2.6548438, -4.4630518, 4.4825640
8: -2.4493060, 1.3118429, -3.8160651, 1.7576821, -4.2069883, 5.1279078
9: -1.5935162, 1.7208647, -2.4063938, 2.4488807, -4.0423970, 4.1272583

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2800580
time: 2.02 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2808330
time: 1.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -2.0812759, 1.7388462, -2.7553742, 2.2292750, -4.3105507, 4.4942203
1: -1.6805236, 1.6002891, -2.2215383, 2.0771291, -3.7576528, 3.8218274
2: -2.0233788, 1.6545870, -2.8573298, 2.0863564, -4.1097355, 4.5119171
3: -2.2960305, 1.4003358, -3.0967674, 1.7849278, -4.0809584, 4.4971032
4: -2.4466107, 1.6642282, -3.2374682, 2.2157748, -4.6623855, 4.9016962
5: -1.9141870, 1.7769457, -2.6361611, 2.2450097, -4.1591969, 4.4131069
6: -1.7262683, 1.9657857, -2.3526342, 2.5508597, -4.2771282, 4.3184199
7: -2.0387635, 2.0116763, -2.7337642, 2.6905286, -4.7292919, 4.7454405
8: -2.8200552, 1.4329078, -3.8716187, 1.7791513, -4.5992064, 5.3045263
9: -1.8069404, 1.9201001, -2.4401627, 2.4806170, -4.2875576, 4.3602629

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804181, upper bound: 4.2800580
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2804181, upper bound: 4.2808330
time: 1.67 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -2.2618871, 1.8464744, -2.4668553, 2.0117130, -4.2736001, 4.3133297
1: -1.8277041, 1.7360274, -1.9938930, 1.8764622, -3.7041664, 3.7299204
2: -2.2569242, 1.7718315, -2.5061800, 1.9068102, -4.1637344, 4.2780113
3: -2.5204260, 1.5026162, -2.7601337, 1.6224308, -4.1428566, 4.2627497
4: -2.6700065, 1.8147115, -2.9065208, 1.9842879, -4.6542945, 4.7212324
5: -2.1106806, 1.8917195, -2.3305614, 2.0518069, -4.1624875, 4.2222810
6: -1.8933653, 2.1022105, -2.0896711, 2.3072011, -4.2005663, 4.1918817
7: -2.2367978, 2.2034998, -2.4402728, 2.4050071, -4.6418047, 4.6437726
8: -3.1042314, 1.4917414, -3.4403281, 1.6342632, -4.7384944, 4.9320698
9: -1.9823278, 2.0509677, -2.1763244, 2.2565827, -4.2389107, 4.2272921

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799326, upper bound: 4.2798604
time: 1.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799326, upper bound: 4.2803913
time: 1.77 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -2.5086644, 2.0368962, -2.5061016, 2.0428770, -4.5515413, 4.5429978
1: -2.0226388, 1.9023004, -2.0252109, 1.9039276, -3.9265664, 3.9275112
2: -2.5533326, 1.9252393, -2.5540395, 1.9319321, -4.4852648, 4.4792786
3: -2.8057382, 1.6422582, -2.8060374, 1.6448342, -4.4505725, 4.4482956
4: -2.9470456, 2.0125513, -2.9518602, 2.0161910, -4.9632368, 4.9644117
5: -2.3690464, 2.0694890, -2.3726606, 2.0799973, -4.4490438, 4.4421496
6: -2.1209738, 2.3263254, -2.1264291, 2.3431785, -4.4641523, 4.4527545
7: -2.4814527, 2.4429564, -2.4797633, 2.4436944, -4.9251471, 4.9227200
8: -3.4848585, 1.6349620, -3.5006123, 1.6573261, -5.1421847, 5.1355743
9: -2.2063260, 2.2636371, -2.2128787, 2.2906084, -4.4969344, 4.4765158

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806894, upper bound: 4.2798604
time: 1.63 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2806894, upper bound: 4.2803971
time: 1.75 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -2.2618871, 1.8464744, -2.7192976, 2.2005155, -4.4624023, 4.5657721
1: -1.8277041, 1.7360274, -2.1926937, 2.0518489, -3.8795528, 3.9287210
2: -2.2569242, 1.7718315, -2.8132589, 2.0631495, -4.3200736, 4.5850906
3: -2.5204260, 1.5026162, -3.0544732, 1.7642397, -4.2846656, 4.5570893
4: -2.6700065, 1.8147115, -3.1956828, 2.1863675, -4.8563738, 5.0103941
5: -2.1106806, 1.8917195, -2.5973217, 2.2189379, -4.3296185, 4.4890413
6: -1.8933653, 2.1022105, -2.3186181, 2.5174184, -4.4107838, 4.4208288
7: -2.2367978, 2.2034998, -2.6973901, 2.6548438, -4.8916416, 4.9008899
8: -3.1042314, 1.4917414, -3.8160651, 1.7576821, -4.8619137, 5.3078065
9: -1.9823278, 2.0509677, -2.4063938, 2.4488807, -4.4312086, 4.4573612

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2796482
time: 1.83 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2803347
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -2.5086644, 2.0368962, -2.7553742, 2.2292750, -4.7379394, 4.7922707
1: -2.0226388, 1.9023004, -2.2215383, 2.0771291, -4.0997677, 4.1238384
2: -2.5533326, 1.9252393, -2.8573298, 2.0863564, -4.6396890, 4.7825689
3: -2.8057382, 1.6422582, -3.0967674, 1.7849278, -4.5906658, 4.7390256
4: -2.9470456, 2.0125513, -3.2374682, 2.2157748, -5.1628203, 5.2500196
5: -2.3690464, 2.0694890, -2.6361611, 2.2450097, -4.6140561, 4.7056503
6: -2.1209738, 2.3263254, -2.3526342, 2.5508597, -4.6718335, 4.6789598
7: -2.4814527, 2.4429564, -2.7337642, 2.6905286, -5.1719813, 5.1767206
8: -3.4848585, 1.6349620, -3.8716187, 1.7791513, -5.2640100, 5.5065808
9: -2.2063260, 2.2636371, -2.4401627, 2.4806170, -4.6869431, 4.7037997

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 245
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 159

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803368, upper bound: 4.2796482
time: 2.33 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2803368, upper bound: 4.2803428
time: 1.95 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 6.19 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2812689, upper bound: 4.2812689
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2812689, upper bound: 4.2817331
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2817331, upper bound: 4.2812689
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2817331, upper bound: 4.2817331
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2807852
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2806202, upper bound: 4.2813745
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2807852
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2810987, upper bound: 4.2813745
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2807852, upper bound: 4.2806202
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2807852, upper bound: 4.2810987
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2813745, upper bound: 4.2806202
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2813745, upper bound: 4.2810987
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2804938
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2804938, upper bound: 4.2810406
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2804938
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2810406, upper bound: 4.2810406
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2809428, upper bound: 4.2809772
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2809428, upper bound: 4.2815140
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2813755, upper bound: 4.2809772
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2813755, upper bound: 4.2815140
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2802009
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2799724, upper bound: 4.2810408
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2804438, upper bound: 4.2802009
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2804438, upper bound: 4.2810408
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2803228, upper bound: 4.2801701
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2803228, upper bound: 4.2807760
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2809117, upper bound: 4.2801701
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2809117, upper bound: 4.2807760
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2799326
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2798604, upper bound: 4.2806894
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2803913, upper bound: 4.2799326
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2803913, upper bound: 4.2806894
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2809772, upper bound: 4.2809428
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2809772, upper bound: 4.2813755
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2815140, upper bound: 4.2809428
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2815140, upper bound: 4.2813755
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2808198
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2808198, upper bound: 4.2813241
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2808198
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2813252, upper bound: 4.2813241
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2802009, upper bound: 4.2799724
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2802009, upper bound: 4.2804438
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2810408, upper bound: 4.2799724
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2810408, upper bound: 4.2804495
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2798548
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2800580, upper bound: 4.2804110
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2798548
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2808337, upper bound: 4.2804183
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2801701, upper bound: 4.2803228
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2801701, upper bound: 4.2809117
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2807760, upper bound: 4.2803228
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2807760, upper bound: 4.2809117
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2800580
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2798548, upper bound: 4.2808330
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2804181, upper bound: 4.2800580
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2804181, upper bound: 4.2808330
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2799326, upper bound: 4.2798604
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2799326, upper bound: 4.2803913
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2806894, upper bound: 4.2798604
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2806894, upper bound: 4.2803971
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2796482
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2796482, upper bound: 4.2803347
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2803368, upper bound: 4.2796482
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 6.19
Output dim: 8, lower bound: -4.2803368, upper bound: 4.2803428

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -1.6145827, 1.3792419, -2.9938245, 2.9938245
1: -1.3168527, 1.2810533, -1.3168527, 1.2810533, -2.5979061, 2.5979061
2: -1.4648219, 1.3692248, -1.4648219, 1.3692248, -2.8340468, 2.8340468
3: -1.7454548, 1.1351854, -1.7454548, 1.1351854, -2.8806400, 2.8806400
4: -1.9133011, 1.2971265, -1.9133011, 1.2971265, -3.2104278, 3.2104278
5: -1.4184097, 1.4522150, -1.4184097, 1.4522150, -2.8706245, 2.8706245
6: -1.3162330, 1.5626514, -1.3162330, 1.5626514, -2.8788843, 2.8788843
7: -1.5742366, 1.5566748, -1.5742366, 1.5566748, -3.1309114, 3.1309114
8: -2.0757601, 1.2396940, -2.0757601, 1.2396940, -3.3154540, 3.3154540
9: -1.3722337, 1.5473963, -1.3722337, 1.5473963, -2.9196301, 2.9196301

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2742791, upper bound: 4.2752380
time: 1.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797657, upper bound: 4.2797677
time: 1.52 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -1.8432285, 1.5589267, -3.1735094, 3.2224703
1: -1.3168527, 1.2810533, -1.4958531, 1.4391533, -2.7560060, 2.7769065
2: -1.4648219, 1.3692248, -1.7368103, 1.5125700, -2.9773920, 3.1060352
3: -1.7454548, 1.1351854, -2.0219579, 1.2669443, -3.0123992, 3.1571431
4: -1.9133011, 1.2971265, -2.1765027, 1.4790530, -3.3923540, 3.4736292
5: -1.4184097, 1.4522150, -1.6626878, 1.6230266, -3.0414362, 3.1149027
6: -1.3162330, 1.5626514, -1.5174071, 1.7687372, -3.0849702, 3.0800586
7: -1.5742366, 1.5566748, -1.8011651, 1.7800044, -3.3542409, 3.3578401
8: -2.0757601, 1.2396940, -2.4602904, 1.3307571, -3.4065173, 3.6999846
9: -1.3722337, 1.5473963, -1.5911342, 1.7429529, -3.1151867, 3.1385305

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2742791, upper bound: 4.2758038
time: 1.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797657, upper bound: 4.2802472
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -1.6145827, 1.3792419, -3.2224703, 3.1735094
1: -1.4958531, 1.4391533, -1.3168527, 1.2810533, -2.7769065, 2.7560060
2: -1.7368103, 1.5125700, -1.4648219, 1.3692248, -3.1060352, 2.9773920
3: -2.0219579, 1.2669443, -1.7454548, 1.1351854, -3.1571431, 3.0123992
4: -2.1765027, 1.4790530, -1.9133011, 1.2971265, -3.4736292, 3.3923540
5: -1.6626878, 1.6230266, -1.4184097, 1.4522150, -3.1149027, 3.0414362
6: -1.5174071, 1.7687372, -1.3162330, 1.5626514, -3.0800586, 3.0849702
7: -1.8011651, 1.7800044, -1.5742366, 1.5566748, -3.3578401, 3.3542409
8: -2.4602904, 1.3307571, -2.0757601, 1.2396940, -3.6999846, 3.4065173
9: -1.5911342, 1.7429529, -1.3722337, 1.5473963, -3.1385305, 3.1151867

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2749220, upper bound: 4.2752171
time: 2.09 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2802472, upper bound: 4.2797657
time: 3.02 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -1.8432285, 1.5589267, -3.4021552, 3.4021552
1: -1.4958531, 1.4391533, -1.4958531, 1.4391533, -2.9350064, 2.9350064
2: -1.7368103, 1.5125700, -1.7368103, 1.5125700, -3.2493804, 3.2493804
3: -2.0219579, 1.2669443, -2.0219579, 1.2669443, -3.2889023, 3.2889023
4: -2.1765027, 1.4790530, -2.1765027, 1.4790530, -3.6555557, 3.6555557
5: -1.6626878, 1.6230266, -1.6626878, 1.6230266, -3.2857144, 3.2857144
6: -1.5174071, 1.7687372, -1.5174071, 1.7687372, -3.2861443, 3.2861443
7: -1.8011651, 1.7800044, -1.8011651, 1.7800044, -3.5811696, 3.5811696
8: -2.4602904, 1.3307571, -2.4602904, 1.3307571, -3.7910476, 3.7910476
9: -1.5911342, 1.7429529, -1.5911342, 1.7429529, -3.3340871, 3.3340871

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2749220, upper bound: 4.2753599
time: 1.63 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2802472, upper bound: 4.2798384
time: 2.97 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.0037477, 1.6518627, -3.2664454, 3.3829896
1: -1.3168527, 1.2810533, -1.6264559, 1.5585181, -2.8753707, 2.9075093
2: -1.4648219, 1.3692248, -1.9452751, 1.6141748, -3.0789967, 3.3144999
3: -1.7454548, 1.1351854, -2.2200139, 1.3563321, -3.1017869, 3.3551993
4: -1.9133011, 1.2971265, -2.3756607, 1.6093513, -3.5226524, 3.6727872
5: -1.4184097, 1.4522150, -1.8359622, 1.7206392, -3.1390491, 3.2881770
6: -1.3162330, 1.5626514, -1.6592586, 1.8914390, -3.2076721, 3.2219100
7: -1.5742366, 1.5566748, -1.9775292, 1.9471791, -3.5214157, 3.5342040
8: -2.0757601, 1.2396940, -2.7113912, 1.3773797, -3.4531398, 3.9510851
9: -1.3722337, 1.5473963, -1.7459316, 1.8566196, -3.2288532, 3.2933278

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2730489, upper bound: 4.2742954
time: 3.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791913, upper bound: 4.2793266
time: 1.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.2651904, 1.8558743, -3.4704571, 3.6444323
1: -1.3168527, 1.2810533, -1.8323848, 1.7356169, -3.0524697, 3.1134381
2: -1.4648219, 1.3692248, -2.2596996, 1.7757981, -3.2406201, 3.6289244
3: -1.7454548, 1.1351854, -2.5229864, 1.5075209, -3.2529757, 3.6581717
4: -1.9133011, 1.2971265, -2.6710010, 1.8203437, -3.7336450, 3.9681275
5: -1.4184097, 1.4522150, -2.1143703, 1.9067429, -3.3251526, 3.5665853
6: -1.3162330, 1.5626514, -1.9004462, 2.1250553, -3.4412885, 3.4630976
7: -1.5742366, 1.5566748, -2.2354999, 2.2050855, -3.7793221, 3.7921748
8: -2.0757601, 1.2396940, -3.1254687, 1.5177147, -3.5934749, 4.3651628
9: -1.3722337, 1.5473963, -1.9864624, 2.0786352, -3.4508691, 3.5338588

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2730489, upper bound: 4.2749973
time: 1.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2791913, upper bound: 4.2799501
time: 1.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.0037477, 1.6518627, -3.4950912, 3.5626745
1: -1.4958531, 1.4391533, -1.6264559, 1.5585181, -3.0543711, 3.0656092
2: -1.7368103, 1.5125700, -1.9452751, 1.6141748, -3.3509851, 3.4578452
3: -2.0219579, 1.2669443, -2.2200139, 1.3563321, -3.3782899, 3.4869580
4: -2.1765027, 1.4790530, -2.3756607, 1.6093513, -3.7858539, 3.8547137
5: -1.6626878, 1.6230266, -1.8359622, 1.7206392, -3.3833270, 3.4589887
6: -1.5174071, 1.7687372, -1.6592586, 1.8914390, -3.4088459, 3.4279957
7: -1.8011651, 1.7800044, -1.9775292, 1.9471791, -3.7483442, 3.7575336
8: -2.4602904, 1.3307571, -2.7113912, 1.3773797, -3.8376701, 4.0421486
9: -1.5911342, 1.7429529, -1.7459316, 1.8566196, -3.4477539, 3.4888844

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2735160, upper bound: 4.2742771
time: 3.16 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797118, upper bound: 4.2793255
time: 1.75 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.2651904, 1.8558743, -3.6991029, 3.8241172
1: -1.4958531, 1.4391533, -1.8323848, 1.7356169, -3.2314701, 3.2715383
2: -1.7368103, 1.5125700, -2.2596996, 1.7757981, -3.5126085, 3.7722697
3: -2.0219579, 1.2669443, -2.5229864, 1.5075209, -3.5294788, 3.7899308
4: -2.1765027, 1.4790530, -2.6710010, 1.8203437, -3.9968464, 4.1500540
5: -1.6626878, 1.6230266, -2.1143703, 1.9067429, -3.5694308, 3.7373970
6: -1.5174071, 1.7687372, -1.9004462, 2.1250553, -3.6424623, 3.6691833
7: -1.8011651, 1.7800044, -2.2354999, 2.2050855, -4.0062504, 4.0155044
8: -2.4602904, 1.3307571, -3.1254687, 1.5177147, -3.9780052, 4.4562259
9: -1.5911342, 1.7429529, -1.9864624, 2.0786352, -3.6697693, 3.7294154

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2735160, upper bound: 4.2745500
time: 1.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2797118, upper bound: 4.2794867
time: 1.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -1.6145827, 1.3792419, -3.3829896, 3.2664454
1: -1.6264559, 1.5585181, -1.3168527, 1.2810533, -2.9075093, 2.8753707
2: -1.9452751, 1.6141748, -1.4648219, 1.3692248, -3.3144999, 3.0789967
3: -2.2200139, 1.3563321, -1.7454548, 1.1351854, -3.3551993, 3.1017869
4: -2.3756607, 1.6093513, -1.9133011, 1.2971265, -3.6727872, 3.5226524
5: -1.8359622, 1.7206392, -1.4184097, 1.4522150, -3.2881770, 3.1390491
6: -1.6592586, 1.8914390, -1.3162330, 1.5626514, -3.2219100, 3.2076721
7: -1.9775292, 1.9471791, -1.5742366, 1.5566748, -3.5342040, 3.5214157
8: -2.7113912, 1.3773797, -2.0757601, 1.2396940, -3.9510851, 3.4531398
9: -1.7459316, 1.8566196, -1.3722337, 1.5473963, -3.2933278, 3.2288532

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718441, upper bound: 4.2730302
time: 1.61 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793255, upper bound: 4.2791913
time: 1.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -1.8432285, 1.5589267, -3.5626745, 3.4950912
1: -1.6264559, 1.5585181, -1.4958531, 1.4391533, -3.0656092, 3.0543711
2: -1.9452751, 1.6141748, -1.7368103, 1.5125700, -3.4578452, 3.3509851
3: -2.2200139, 1.3563321, -2.0219579, 1.2669443, -3.4869580, 3.3782899
4: -2.3756607, 1.6093513, -2.1765027, 1.4790530, -3.8547137, 3.7858539
5: -1.8359622, 1.7206392, -1.6626878, 1.6230266, -3.4589887, 3.3833270
6: -1.6592586, 1.8914390, -1.5174071, 1.7687372, -3.4279957, 3.4088459
7: -1.9775292, 1.9471791, -1.8011651, 1.7800044, -3.7575336, 3.7483442
8: -2.7113912, 1.3773797, -2.4602904, 1.3307571, -4.0421486, 3.8376701
9: -1.7459316, 1.8566196, -1.5911342, 1.7429529, -3.4888844, 3.4477539

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718441, upper bound: 4.2736455
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2793255, upper bound: 4.2797118
time: 2.36 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -1.6145827, 1.3792419, -3.6444323, 3.4704571
1: -1.8323848, 1.7356169, -1.3168527, 1.2810533, -3.1134381, 3.0524697
2: -2.2596996, 1.7757981, -1.4648219, 1.3692248, -3.6289244, 3.2406201
3: -2.5229864, 1.5075209, -1.7454548, 1.1351854, -3.6581717, 3.2529757
4: -2.6710010, 1.8203437, -1.9133011, 1.2971265, -3.9681275, 3.7336450
5: -2.1143703, 1.9067429, -1.4184097, 1.4522150, -3.5665853, 3.3251526
6: -1.9004462, 2.1250553, -1.3162330, 1.5626514, -3.4630976, 3.4412885
7: -2.2354999, 2.2050855, -1.5742366, 1.5566748, -3.7921748, 3.7793221
8: -3.1254687, 1.5177147, -2.0757601, 1.2396940, -4.3651628, 3.5934749
9: -1.9864624, 2.0786352, -1.3722337, 1.5473963, -3.5338588, 3.4508691

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724411, upper bound: 4.2730302
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799501, upper bound: 4.2791913
time: 4.46 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -1.8432285, 1.5589267, -3.8241172, 3.6991029
1: -1.8323848, 1.7356169, -1.4958531, 1.4391533, -3.2715383, 3.2314701
2: -2.2596996, 1.7757981, -1.7368103, 1.5125700, -3.7722697, 3.5126085
3: -2.5229864, 1.5075209, -2.0219579, 1.2669443, -3.7899308, 3.5294788
4: -2.6710010, 1.8203437, -2.1765027, 1.4790530, -4.1500540, 3.9968464
5: -2.1143703, 1.9067429, -1.6626878, 1.6230266, -3.7373970, 3.5694308
6: -1.9004462, 2.1250553, -1.5174071, 1.7687372, -3.6691833, 3.6424623
7: -2.2354999, 2.2050855, -1.8011651, 1.7800044, -4.0155044, 4.0062504
8: -3.1254687, 1.5177147, -2.4602904, 1.3307571, -4.4562259, 3.9780052
9: -1.9864624, 2.0786352, -1.5911342, 1.7429529, -3.7294154, 3.6697693

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724411, upper bound: 4.2733132
time: 3.59 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799501, upper bound: 4.2793495
time: 3.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -2.0037477, 1.6518627, -3.6556106, 3.6556106
1: -1.6264559, 1.5585181, -1.6264559, 1.5585181, -3.1849740, 3.1849740
2: -1.9452751, 1.6141748, -1.9452751, 1.6141748, -3.5594499, 3.5594499
3: -2.2200139, 1.3563321, -2.2200139, 1.3563321, -3.5763459, 3.5763459
4: -2.3756607, 1.6093513, -2.3756607, 1.6093513, -3.9850121, 3.9850121
5: -1.8359622, 1.7206392, -1.8359622, 1.7206392, -3.5566015, 3.5566015
6: -1.6592586, 1.8914390, -1.6592586, 1.8914390, -3.5506976, 3.5506976
7: -1.9775292, 1.9471791, -1.9775292, 1.9471791, -3.9247084, 3.9247084
8: -2.7113912, 1.3773797, -2.7113912, 1.3773797, -4.0887709, 4.0887709
9: -1.7459316, 1.8566196, -1.7459316, 1.8566196, -3.6025512, 3.6025512

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716720, upper bound: 4.2729357
time: 2.49 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790684, upper bound: 4.2790684
time: 1.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -2.2645829, 1.8558743, -3.8596220, 3.9164457
1: -1.6264559, 1.5585181, -1.8319528, 1.7356169, -3.3620729, 3.3904710
2: -1.9452751, 1.6141748, -2.2596996, 1.7749428, -3.7202177, 3.8738744
3: -2.2200139, 1.3563321, -2.5223713, 1.5075209, -3.7275348, 3.8787034
4: -2.3756607, 1.6093513, -2.6702142, 1.8202523, -4.1959128, 4.2795653
5: -1.8359622, 1.7206392, -2.1143703, 1.9065698, -3.7425320, 3.8350096
6: -1.6592586, 1.8914390, -1.8998392, 2.1250553, -3.7843139, 3.7912781
7: -1.9775292, 1.9471791, -2.2354999, 2.2045813, -4.1821103, 4.1826792
8: -2.7113912, 1.3773797, -3.1249814, 1.5176911, -4.2290821, 4.5023613
9: -1.7459316, 1.8566196, -1.9864597, 2.0784044, -3.8243361, 3.8430793

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2716720, upper bound: 4.2736008
time: 7.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790684, upper bound: 4.2796523
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -2.0037477, 1.6518627, -3.9170532, 3.8596220
1: -1.8323848, 1.7356169, -1.6264559, 1.5585181, -3.3909030, 3.3620729
2: -2.2596996, 1.7757981, -1.9452751, 1.6141748, -3.8738744, 3.7210732
3: -2.5229864, 1.5075209, -2.2200139, 1.3563321, -3.8793185, 3.7275348
4: -2.6710010, 1.8203437, -2.3756607, 1.6093513, -4.2803521, 4.1960044
5: -2.1143703, 1.9067429, -1.8359622, 1.7206392, -3.8350096, 3.7427051
6: -1.9004462, 2.1250553, -1.6592586, 1.8914390, -3.7918851, 3.7843139
7: -2.2354999, 2.2050855, -1.9775292, 1.9471791, -4.1826792, 4.1826148
8: -3.1254687, 1.5177147, -2.7113912, 1.3773797, -4.5028486, 4.2291059
9: -1.9864624, 2.0786352, -1.7459316, 1.8566196, -3.8430820, 3.8245668

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2722267, upper bound: 4.2729357
time: 1.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796523, upper bound: 4.2790684
time: 2.39 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -2.2645829, 1.8558743, -4.1210647, 4.1204572
1: -1.8323848, 1.7356169, -1.8319528, 1.7356169, -3.5680017, 3.5675697
2: -2.2596996, 1.7757981, -2.2596996, 1.7749428, -4.0346422, 4.0354977
3: -2.5229864, 1.5075209, -2.5223713, 1.5075209, -4.0305071, 4.0298920
4: -2.6710010, 1.8203437, -2.6702142, 1.8202523, -4.4912534, 4.4905577
5: -2.1143703, 1.9067429, -2.1143703, 1.9065698, -4.0209403, 4.0211134
6: -1.9004462, 2.1250553, -1.8998392, 2.1250553, -4.0255013, 4.0248947
7: -2.2354999, 2.2050855, -2.2354999, 2.2045813, -4.4400811, 4.4405851
8: -3.1254687, 1.5177147, -3.1249814, 1.5176911, -4.6431599, 4.6426964
9: -1.9864624, 2.0786352, -1.9864597, 2.0784044, -4.0648670, 4.0650949

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2722267, upper bound: 4.2732390
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2796523, upper bound: 4.2792499
time: 1.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -1.8458550, 1.5601518, -3.1747346, 3.2250969
1: -1.3168527, 1.2810533, -1.4973698, 1.4441805, -2.7610331, 2.7784231
2: -1.4648219, 1.3692248, -1.7430093, 1.5131910, -2.9780130, 3.1122341
3: -1.7454548, 1.1351854, -2.0262935, 1.2658768, -3.0113316, 3.1614790
4: -1.9133011, 1.2971265, -2.1838734, 1.4798396, -3.3931408, 3.4809999
5: -1.4184097, 1.4522150, -1.6644652, 1.6105781, -3.0289879, 3.1166801
6: -1.3162330, 1.5626514, -1.5194894, 1.7553332, -3.0715661, 3.0821409
7: -1.5742366, 1.5566748, -1.8082078, 1.7851741, -3.3594108, 3.3648825
8: -2.0757601, 1.2396940, -2.4493060, 1.3118429, -3.3876030, 3.6890001
9: -1.3722337, 1.5473963, -1.5935162, 1.7208647, -3.0930984, 3.1409125

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2736635, upper bound: 4.2747252
time: 1.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794401, upper bound: 4.2794666
time: 5.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.0812759, 1.7388462, -3.3534288, 3.4605179
1: -1.3168527, 1.2810533, -1.6805236, 1.6002891, -2.9171419, 2.9615769
2: -1.4648219, 1.3692248, -2.0233788, 1.6545870, -3.1194091, 3.3926036
3: -1.7454548, 1.1351854, -2.2960305, 1.4003358, -3.1457906, 3.4312158
4: -1.9133011, 1.2971265, -2.4466107, 1.6642282, -3.5775294, 3.7437372
5: -1.4184097, 1.4522150, -1.9141870, 1.7769457, -3.1953554, 3.3664019
6: -1.3162330, 1.5626514, -1.7262683, 1.9657857, -3.2820187, 3.2889197
7: -1.5742366, 1.5566748, -2.0387635, 2.0116763, -3.5859129, 3.5954385
8: -2.0757601, 1.2396940, -2.8200552, 1.4329078, -3.5086679, 4.0597491
9: -1.3722337, 1.5473963, -1.8069404, 1.9201001, -3.2923338, 3.3543367

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2736635, upper bound: 4.2754328
time: 1.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2794401, upper bound: 4.2800401
time: 2.33 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -1.8458550, 1.5601518, -3.4033804, 3.4047818
1: -1.4958531, 1.4391533, -1.4973698, 1.4441805, -2.9400334, 2.9365230
2: -1.7368103, 1.5125700, -1.7430093, 1.5131910, -3.2500014, 3.2555795
3: -2.0219579, 1.2669443, -2.0262935, 1.2658768, -3.2878346, 3.2932377
4: -2.1765027, 1.4790530, -2.1838734, 1.4798396, -3.6563423, 3.6629264
5: -1.6626878, 1.6230266, -1.6644652, 1.6105781, -3.2732658, 3.2874918
6: -1.5174071, 1.7687372, -1.5194894, 1.7553332, -3.2727404, 3.2882266
7: -1.8011651, 1.7800044, -1.8082078, 1.7851741, -3.5863392, 3.5882120
8: -2.4602904, 1.3307571, -2.4493060, 1.3118429, -3.7721334, 3.7800632
9: -1.5911342, 1.7429529, -1.5935162, 1.7208647, -3.3119988, 3.3364692

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2743240, upper bound: 4.2747089
time: 5.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799208, upper bound: 4.2794654
time: 1.65 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.0812759, 1.7388462, -3.5820746, 3.6402025
1: -1.4958531, 1.4391533, -1.6805236, 1.6002891, -3.0961423, 3.1196771
2: -1.7368103, 1.5125700, -2.0233788, 1.6545870, -3.3913975, 3.5359488
3: -2.0219579, 1.2669443, -2.2960305, 1.4003358, -3.4222937, 3.5629749
4: -2.1765027, 1.4790530, -2.4466107, 1.6642282, -3.8407309, 3.9256637
5: -1.6626878, 1.6230266, -1.9141870, 1.7769457, -3.4396334, 3.5372136
6: -1.5174071, 1.7687372, -1.7262683, 1.9657857, -3.4831929, 3.4950056
7: -1.8011651, 1.7800044, -2.0387635, 2.0116763, -3.8128414, 3.8187680
8: -2.4602904, 1.3307571, -2.8200552, 1.4329078, -3.8931983, 4.1508121
9: -1.5911342, 1.7429529, -1.8069404, 1.9201001, -3.5112343, 3.5498934

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2743240, upper bound: 4.2749831
time: 2.02 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2799208, upper bound: 4.2795822
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.2618871, 1.8464744, -3.4610572, 3.6411290
1: -1.3168527, 1.2810533, -1.8277041, 1.7360274, -3.0528800, 3.1087575
2: -1.4648219, 1.3692248, -2.2569242, 1.7718315, -3.2366533, 3.6261489
3: -1.7454548, 1.1351854, -2.5204260, 1.5026162, -3.2480710, 3.6556115
4: -1.9133011, 1.2971265, -2.6700065, 1.8147115, -3.7280126, 3.9671330
5: -1.4184097, 1.4522150, -2.1106806, 1.8917195, -3.3101292, 3.5628955
6: -1.3162330, 1.5626514, -1.8933653, 2.1022105, -3.4184437, 3.4560165
7: -1.5742366, 1.5566748, -2.2367978, 2.2034998, -3.7777364, 3.7934728
8: -2.0757601, 1.2396940, -3.1042314, 1.4917414, -3.5675015, 4.3439255
9: -1.3722337, 1.5473963, -1.9823278, 2.0509677, -3.4232016, 3.5297241

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718439, upper bound: 4.2733267
time: 4.16 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785916, upper bound: 4.2787933
time: 4.93 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.6145827, 1.3792419, -2.5087495, 2.0368962, -3.6514788, 3.8879914
1: -1.3168527, 1.2810533, -2.0226388, 1.9026828, -3.2195354, 3.3036921
2: -1.4648219, 1.3692248, -2.5533326, 1.9254339, -3.3902559, 3.9225574
3: -1.7454548, 1.1351854, -2.8057382, 1.6424177, -3.3878725, 3.9409237
4: -1.9133011, 1.2971265, -2.9470456, 2.0133429, -3.9266438, 4.2441721
5: -1.4184097, 1.4522150, -2.3690577, 2.0694890, -3.4878988, 3.8212726
6: -1.3162330, 1.5626514, -2.1209738, 2.3267791, -3.6430120, 3.6836252
7: -1.5742366, 1.5566748, -2.4814527, 2.4430037, -4.0172405, 4.0381274
8: -2.0757601, 1.2396940, -3.4850137, 1.6349620, -3.7107220, 4.7247076
9: -1.3722337, 1.5473963, -2.2063260, 2.2638803, -3.6361141, 3.7537222

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2718439, upper bound: 4.2744415
time: 4.04 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2785916, upper bound: 4.2796366
time: 1.66 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.2618871, 1.8464744, -3.6897030, 3.8208137
1: -1.4958531, 1.4391533, -1.8277041, 1.7360274, -3.2318804, 3.2668574
2: -1.7368103, 1.5125700, -2.2569242, 1.7718315, -3.5086417, 3.7694941
3: -2.0219579, 1.2669443, -2.5204260, 1.5026162, -3.5245740, 3.7873702
4: -2.1765027, 1.4790530, -2.6700065, 1.8147115, -3.9912143, 4.1490593
5: -1.6626878, 1.6230266, -2.1106806, 1.8917195, -3.5544071, 3.7337072
6: -1.5174071, 1.7687372, -1.8933653, 2.1022105, -3.6196175, 3.6621025
7: -1.8011651, 1.7800044, -2.2367978, 2.2034998, -4.0046649, 4.0168023
8: -2.4602904, 1.3307571, -3.1042314, 1.4917414, -3.9520319, 4.4349885
9: -1.5911342, 1.7429529, -1.9823278, 2.0509677, -3.6421018, 3.7252808

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724019, upper bound: 4.2733181
time: 1.57 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790935, upper bound: 4.2787916
time: 1.52 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.8432285, 1.5589267, -2.5087495, 2.0368962, -3.8801246, 4.0676761
1: -1.4958531, 1.4391533, -2.0226388, 1.9026828, -3.3985357, 3.4617920
2: -1.7368103, 1.5125700, -2.5533326, 1.9254339, -3.6622443, 4.0659027
3: -2.0219579, 1.2669443, -2.8057382, 1.6424177, -3.6643755, 4.0726824
4: -2.1765027, 1.4790530, -2.9470456, 2.0133429, -4.1898456, 4.4260988
5: -1.6626878, 1.6230266, -2.3690577, 2.0694890, -3.7321768, 3.9920843
6: -1.5174071, 1.7687372, -2.1209738, 2.3267791, -3.8441863, 3.8897109
7: -1.8011651, 1.7800044, -2.4814527, 2.4430037, -4.2441688, 4.2614570
8: -2.4602904, 1.3307571, -3.4850137, 1.6349620, -4.0952525, 4.8157711
9: -1.5911342, 1.7429529, -2.2063260, 2.2638803, -3.8550143, 3.9492788

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2724019, upper bound: 4.2739821
time: 1.90 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2790935, upper bound: 4.2790601
time: 2.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -1.8458550, 1.5601518, -3.5638995, 3.4977179
1: -1.6264559, 1.5585181, -1.4973698, 1.4441805, -3.0706363, 3.0558877
2: -1.9452751, 1.6141748, -1.7430093, 1.5131910, -3.4584661, 3.3571842
3: -2.2200139, 1.3563321, -2.0262935, 1.2658768, -3.4858906, 3.3826256
4: -2.3756607, 1.6093513, -2.1838734, 1.4798396, -3.8555002, 3.7932248
5: -1.8359622, 1.7206392, -1.6644652, 1.6105781, -3.4465404, 3.3851044
6: -1.6592586, 1.8914390, -1.5194894, 1.7553332, -3.4145918, 3.4109282
7: -1.9775292, 1.9471791, -1.8082078, 1.7851741, -3.7627034, 3.7553868
8: -2.7113912, 1.3773797, -2.4493060, 1.3118429, -4.0232344, 3.8266857
9: -1.7459316, 1.8566196, -1.5935162, 1.7208647, -3.4667964, 3.4501357

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2708990, upper bound: 4.2720860
time: 7.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2788996, upper bound: 4.2787720
time: 1.56 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -2.0037477, 1.6518627, -2.0812759, 1.7388462, -3.7425938, 3.7331386
1: -1.6264559, 1.5585181, -1.6805236, 1.6002891, -3.2267451, 3.2390418
2: -1.9452751, 1.6141748, -2.0233788, 1.6545870, -3.5998621, 3.6375537
3: -2.2200139, 1.3563321, -2.2960305, 1.4003358, -3.6203496, 3.6523626
4: -2.3756607, 1.6093513, -2.4466107, 1.6642282, -4.0398889, 4.0559621
5: -1.8359622, 1.7206392, -1.9141870, 1.7769457, -3.6129079, 3.6348262
6: -1.6592586, 1.8914390, -1.7262683, 1.9657857, -3.6250443, 3.6177073
7: -1.9775292, 1.9471791, -2.0387635, 2.0116763, -3.9892054, 3.9859426
8: -2.7113912, 1.3773797, -2.8200552, 1.4329078, -4.1442990, 4.1974349
9: -1.7459316, 1.8566196, -1.8069404, 1.9201001, -3.6660318, 3.6635599

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2708990, upper bound: 4.2729731
time: 5.52 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2788996, upper bound: 4.2793868
time: 1.59 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -1.8458550, 1.5601518, -3.8253422, 3.7017293
1: -1.8323848, 1.7356169, -1.4973698, 1.4441805, -3.2765653, 3.2329867
2: -2.2596996, 1.7757981, -1.7430093, 1.5131910, -3.7728906, 3.5188074
3: -2.5229864, 1.5075209, -2.0262935, 1.2658768, -3.7888632, 3.5338144
4: -2.6710010, 1.8203437, -2.1838734, 1.4798396, -4.1508408, 4.0042171
5: -2.1143703, 1.9067429, -1.6644652, 1.6105781, -3.7249484, 3.5712080
6: -1.9004462, 2.1250553, -1.5194894, 1.7553332, -3.6557794, 3.6445446
7: -2.2354999, 2.2050855, -1.8082078, 1.7851741, -4.0206738, 4.0132933
8: -3.1254687, 1.5177147, -2.4493060, 1.3118429, -4.4373116, 3.9670208
9: -1.9864624, 2.0786352, -1.5935162, 1.7208647, -3.7073269, 3.6721516

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 245
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2715887, upper bound: 4.2720860
time: 1.66 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -4.2795189, upper bound: 4.2787720
time: 2.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -2.2651904, 1.8558743, -2.0812759, 1.7388462, -4.0040364, 3.9371502
1: -1.8323848, 1.7356169, -1.6805236, 1.6002891, -3.4326739, 3.4161406
2: -2.2596996, 1.7757981, -2.0233788, 1.6545870, -3.9142866, 3.7991769
3: -2.5229864, 1.5075209, -2.2960305, 1.4003358, -3.9233222, 3.8035514
4: -2.6710010, 1.8203437, -2.4466107, 1.6642282, -4.3352289, 4.2669544
5: -2.1143703, 1.9067429, -1.9141870, 1.7769457, -3.8913159, 3.8209300
6: -1.9004462, 2.1250553, -1.7262683, 1.9657857, -3.8662319, 3.8513236
7: -2.2354999, 2.2050855, -2.0387635, 2.0116763, -4.2471762, 4.2438488
8: -3.1254687, 1.5177147, -2.8200552, 1.4329078, -4.5583763, 4.3377700
9: -1.9864624, 2.0786352, -1.8069404, 1.9201001, -3.9065623, 3.8855758

Time for backsubstitution: 1.68 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 5.24 + 595.56 = 600.80 seconds
