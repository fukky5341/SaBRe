## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.287744645


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.3012581, -5.2145228, -6.3012581, -5.2145228, -0.6356344, 0.6356342)
1: (-7.0748386, -6.2827544, -7.0748386, -6.2827544, -0.5325336, 0.5325336)
2: (-7.7188220, -6.7522135, -7.7188220, -6.7522135, -0.5109730, 0.5109732)
3: (-5.6355677, -4.5174036, -5.6355677, -4.5174036, -0.7120657, 0.7120652)
4: (-7.8559666, -6.7313633, -7.8559666, -6.7313633, -0.6489444, 0.6489446)
5: (-0.6390738, 0.2880228, -0.6390738, 0.2880228, -0.5138854, 0.5138855)
6: (-2.6789899, -1.7529851, -2.6789899, -1.7529851, -0.6067991, 0.6067991)
7: (-10.3231449, -9.3116732, -10.3231449, -9.3116732, -0.6091228, 0.6091225)
8: (7.6439333, 8.2786579, 7.6439333, 8.2786579, -0.4724457, 0.4724457)
9: (-5.9656425, -4.8498135, -5.9656425, -4.8498135, -0.7989490, 0.7989488)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.76 + 33.99 = 56.75 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3028891, upper bound: 0.3028899

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4596
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 4596

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3028824
time: 3.67 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028866, upper bound: 0.3028876
time: 4.09 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.98 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.98
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3028824
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.98
Output dim: 8, lower bound: -0.3028866, upper bound: 0.3028876

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.2955713, -5.2166767, -6.2995672, -5.2146235, -0.6300993, 0.6316321
1: -7.0692120, -6.2877288, -7.0728297, -6.2829561, -0.5265889, 0.5255992
2: -7.7165623, -6.7542505, -7.7180443, -6.7522674, -0.5086243, 0.5076890
3: -5.6310658, -4.5208149, -5.6340637, -4.5174656, -0.7072434, 0.7074161
4: -7.8534293, -6.7369132, -7.8559632, -6.7332067, -0.6446133, 0.6432931
5: -0.6370356, 0.2843680, -0.6389737, 0.2867427, -0.5102606, 0.5100639
6: -2.6769042, -1.7538207, -2.6784763, -1.7530367, -0.6047671, 0.6052699
7: -10.3206367, -9.3132687, -10.3224125, -9.3118477, -0.6062346, 0.6059129
8: 7.6477642, 8.2772732, 7.6452265, 8.2785740, -0.4685044, 0.4695172
9: -5.9617362, -4.8537521, -5.9653687, -4.8512201, -0.7937632, 0.7945576

Time for backsubstitution: 21.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4596
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 4596

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3020143
time: 5.37 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3028824
time: 3.99 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.3012533, -5.2145228, -6.3012581, -5.2145243, -0.6318319, 0.6356320
1: -7.0748320, -6.2827568, -7.0748358, -6.2827549, -0.5263782, 0.5324409
2: -7.7188196, -6.7522140, -7.7188230, -6.7522120, -0.5096536, 0.5105991
3: -5.6355600, -4.5174046, -5.6355658, -4.5174041, -0.7082853, 0.7119360
4: -7.8559661, -6.7313695, -7.8559666, -6.7313643, -0.6489429, 0.6441641
5: -0.6390729, 0.2880161, -0.6390731, 0.2880220, -0.5136383, 0.5106354
6: -2.6789882, -1.7529861, -2.6789908, -1.7529849, -0.6067934, 0.6068461
7: -10.3231421, -9.3116751, -10.3231449, -9.3116760, -0.6082339, 0.6086016
8: 7.6439390, 8.2786560, 7.6439342, 8.2786570, -0.4702854, 0.4722594
9: -5.9656420, -4.8498163, -5.9656434, -4.8498139, -0.7987926, 0.7948396

Time for backsubstitution: 21.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4596
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 4596

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028816, upper bound: 0.3020144
time: 5.24 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028816, upper bound: 0.3020145
time: 4.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.29 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3020143
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3028824
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 8, lower bound: -0.3028816, upper bound: 0.3020144
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.29
Output dim: 8, lower bound: -0.3028816, upper bound: 0.3020145

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.2955713, -5.2166767, -6.2955713, -5.2166767, -0.6280265, 0.6280267
1: -7.0692120, -6.2877288, -7.0692120, -6.2877288, -0.5220451, 0.5220451
2: -7.7165623, -6.7542505, -7.7165623, -6.7542505, -0.5064738, 0.5064740
3: -5.6310658, -4.5208149, -5.6310658, -4.5208149, -0.7043629, 0.7043629
4: -7.8534293, -6.7369132, -7.8534293, -6.7369132, -0.6408415, 0.6408415
5: -0.6370356, 0.2843680, -0.6370356, 0.2843680, -0.5078588, 0.5078588
6: -2.6769042, -1.7538207, -2.6769042, -1.7538207, -0.6038775, 0.6038773
7: -10.3206367, -9.3132687, -10.3206367, -9.3132687, -0.6042504, 0.6042502
8: 7.6477642, 8.2772732, 7.6477642, 8.2772732, -0.4669740, 0.4669739
9: -5.9617362, -4.8537521, -5.9617362, -4.8537521, -0.7911630, 0.7911634

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3020142
time: 5.03 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3020151
time: 4.19 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.2955713, -5.2166767, -6.3012533, -5.2145228, -0.6302178, 0.6334369
1: -7.0692120, -6.2877288, -7.0748320, -6.2827568, -0.5267787, 0.5277071
2: -7.7165623, -6.7542505, -7.7188196, -6.7522140, -0.5083215, 0.5087495
3: -5.6310658, -4.5208149, -5.6355600, -4.5174046, -0.7072353, 0.7090588
4: -7.8534293, -6.7369132, -7.8559661, -6.7313695, -0.6464739, 0.6433041
5: -0.6370356, 0.2843680, -0.6390729, 0.2880161, -0.5115175, 0.5099764
6: -2.6769042, -1.7538207, -2.6789882, -1.7529861, -0.6048036, 0.6058517
7: -10.3206367, -9.3132687, -10.3231421, -9.3116751, -0.6059308, 0.6069183
8: 7.6477642, 8.2772732, 7.6439390, 8.2786560, -0.4684460, 0.4707830
9: -5.9617362, -4.8537521, -5.9656420, -4.8498163, -0.7951953, 0.7947578

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3028822
time: 4.19 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3028822
time: 3.69 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.3012533, -5.2145228, -6.2955713, -5.2166767, -0.6334372, 0.6302176
1: -7.0748320, -6.2827568, -7.0692120, -6.2877288, -0.5277076, 0.5267787
2: -7.7188196, -6.7522140, -7.7165623, -6.7542505, -0.5087492, 0.5083213
3: -5.6355600, -4.5174046, -5.6310658, -4.5208149, -0.7090592, 0.7072349
4: -7.8559661, -6.7313695, -7.8534293, -6.7369132, -0.6433039, 0.6464741
5: -0.6390729, 0.2880161, -0.6370356, 0.2843680, -0.5099764, 0.5115175
6: -2.6789882, -1.7529861, -2.6769042, -1.7538207, -0.6058517, 0.6048036
7: -10.3231421, -9.3116751, -10.3206367, -9.3132687, -0.6069183, 0.6059310
8: 7.6439390, 8.2786560, 7.6477642, 8.2772732, -0.4707830, 0.4684463
9: -5.9656420, -4.8498163, -5.9617362, -4.8537521, -0.7947578, 0.7951956

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027461, upper bound: 0.3020143
time: 4.95 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028812, upper bound: 0.3020143
time: 3.82 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.3012533, -5.2145228, -6.3012533, -5.2145228, -0.6318314, 0.6318316
1: -7.0748320, -6.2827568, -7.0748320, -6.2827568, -0.5263777, 0.5263774
2: -7.7188196, -6.7522140, -7.7188196, -6.7522140, -0.5096526, 0.5096526
3: -5.6355600, -4.5174046, -5.6355600, -4.5174046, -0.7082849, 0.7082849
4: -7.8559661, -6.7313695, -7.8559661, -6.7313695, -0.6441641, 0.6441643
5: -0.6390729, 0.2880161, -0.6390729, 0.2880161, -0.5106345, 0.5106344
6: -2.6789882, -1.7529861, -2.6789882, -1.7529861, -0.6068439, 0.6068439
7: -10.3231421, -9.3116751, -10.3231421, -9.3116751, -0.6082320, 0.6082323
8: 7.6439390, 8.2786560, 7.6439390, 8.2786560, -0.4702847, 0.4702847
9: -5.9656420, -4.8498163, -5.9656420, -4.8498163, -0.7948387, 0.7948384

Time for backsubstitution: 21.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027464, upper bound: 0.3020207
time: 5.02 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028815, upper bound: 0.3020143
time: 3.99 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 31.15 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3020142
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3020151
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3028822
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3028822
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3027461, upper bound: 0.3020143
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3028812, upper bound: 0.3020143
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3027464, upper bound: 0.3020207
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 31.15
Output dim: 8, lower bound: -0.3028815, upper bound: 0.3020143

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -6.2907219, -5.2168140, -6.2945943, -5.2167039, -0.6215353, 0.6250246
1: -7.0688410, -6.2913294, -7.0691390, -6.2884545, -0.5211425, 0.5185585
2: -7.7145720, -6.7546062, -7.7161622, -6.7543182, -0.5043871, 0.5057633
3: -5.6302543, -4.5252872, -5.6309032, -4.5217171, -0.7028093, 0.6997781
4: -7.8532720, -6.7422695, -7.8533955, -6.7379923, -0.6395297, 0.6354170
5: -0.6362884, 0.2836895, -0.6368828, 0.2842262, -0.5053811, 0.5058084
6: -2.6708522, -1.7544346, -2.6756830, -1.7539409, -0.5970621, 0.6020174
7: -10.3203335, -9.3184814, -10.3205795, -9.3143167, -0.6028013, 0.5984337
8: 7.6483288, 8.2771826, 7.6478748, 8.2772541, -0.4660738, 0.4666301
9: -5.9613199, -4.8557110, -5.9616528, -4.8541460, -0.7901773, 0.7882996

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3018789
time: 4.69 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3020145
time: 4.89 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -6.2964373, -5.2081637, -6.2955694, -5.2166777, -0.6301086, 0.6338184
1: -7.0767188, -6.2855148, -7.0692129, -6.2877288, -0.5297790, 0.5241117
2: -7.7172379, -6.7459521, -7.7165623, -6.7542515, -0.5064971, 0.5143180
3: -5.6419692, -4.5196428, -5.6310654, -4.5208182, -0.7160883, 0.7048707
4: -7.8643961, -6.7340708, -7.8534265, -6.7369165, -0.6469684, 0.6425304
5: -0.6389948, 0.2861705, -0.6370370, 0.2843673, -0.5083774, 0.5129986
6: -2.6780343, -1.7326282, -2.6768987, -1.7538205, -0.6041141, 0.6187637
7: -10.3340549, -9.3126698, -10.3206358, -9.3132706, -0.6120234, 0.6048160
8: 7.6446695, 8.2785454, 7.6477637, 8.2772732, -0.4716711, 0.4682994
9: -5.9713607, -4.8529534, -5.9617333, -4.8537550, -0.8006992, 0.7934585

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3018796
time: 3.84 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020151
time: 3.77 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -6.2907219, -5.2168140, -6.3002758, -5.2145529, -0.6237268, 0.6304388
1: -7.0688410, -6.2913294, -7.0747571, -6.2834835, -0.5258756, 0.5242200
2: -7.7145720, -6.7546062, -7.7184181, -6.7522831, -0.5062404, 0.5080390
3: -5.6302543, -4.5252872, -5.6353989, -4.5183029, -0.7056828, 0.7044740
4: -7.8532720, -6.7422695, -7.8559370, -6.7324505, -0.6451635, 0.6378796
5: -0.6362884, 0.2836895, -0.6389225, 0.2878778, -0.5090404, 0.5079269
6: -2.6708522, -1.7544346, -2.6777694, -1.7531054, -0.5979867, 0.6039934
7: -10.3203335, -9.3184814, -10.3230829, -9.3127260, -0.6044803, 0.6011002
8: 7.6483288, 8.2771826, 7.6440535, 8.2786388, -0.4675465, 0.4704370
9: -5.9613199, -4.8557110, -5.9655581, -4.8502131, -0.7942078, 0.7918925

Time for backsubstitution: 22.08 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3027468
time: 4.82 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3028811
time: 5.76 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -6.2964373, -5.2081637, -6.3012505, -5.2145247, -0.6323001, 0.6366930
1: -7.0767188, -6.2855148, -7.0748301, -6.2827582, -0.5345111, 0.5297737
2: -7.7172379, -6.7459521, -7.7188158, -6.7522140, -0.5083439, 0.5153469
3: -5.6419692, -4.5196428, -5.6355605, -4.5174065, -0.7180135, 0.7095671
4: -7.8643961, -6.7340708, -7.8559680, -6.7313733, -0.6497622, 0.6449931
5: -0.6389948, 0.2861705, -0.6390734, 0.2880173, -0.5120363, 0.5151155
6: -2.6780343, -1.7326282, -2.6789851, -1.7529845, -0.6050401, 0.6201779
7: -10.3340549, -9.3126698, -10.3231411, -9.3116789, -0.6128292, 0.6074843
8: 7.6446695, 8.2785454, 7.6439390, 8.2786560, -0.4731441, 0.4721076
9: -5.9713607, -4.8529534, -5.9656420, -4.8498192, -0.8047299, 0.7970519

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3027468
time: 3.89 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3028820
time: 3.99 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -6.2964015, -5.2146597, -6.2945943, -5.2167039, -0.6269641, 0.6272163
1: -7.0744586, -6.2863560, -7.0691390, -6.2884545, -0.5268044, 0.5232942
2: -7.7168264, -6.7525687, -7.7161622, -6.7543182, -0.5066633, 0.5076234
3: -5.6347475, -4.5218687, -5.6309032, -4.5217171, -0.7075052, 0.7026539
4: -7.8558125, -6.7367268, -7.8533955, -6.7379923, -0.6419930, 0.6410525
5: -0.6383338, 0.2873418, -0.6368828, 0.2842262, -0.5075049, 0.5094688
6: -2.6729426, -1.7536023, -2.6756830, -1.7539409, -0.5990372, 0.6029403
7: -10.3228350, -9.3168888, -10.3205795, -9.3143167, -0.6054645, 0.6001122
8: 7.6445141, 8.2785664, 7.6478748, 8.2772541, -0.4698737, 0.4681028
9: -5.9652209, -4.8517737, -5.9616528, -4.8541460, -0.7937660, 0.7923303

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3018787
time: 3.66 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3020143
time: 4.43 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -6.3021374, -5.2060070, -6.2955694, -5.2166777, -0.6355455, 0.6342177
1: -7.0823383, -6.2804770, -7.0692129, -6.2877288, -0.5354381, 0.5288997
2: -7.7194943, -6.7438407, -7.7165623, -6.7542515, -0.5087750, 0.5150735
3: -5.6464744, -4.5161667, -5.6310654, -4.5208182, -0.7194729, 0.7077899
4: -7.8669467, -6.7285452, -7.8534265, -6.7369165, -0.6471300, 0.6481352
5: -0.6410499, 0.2898231, -0.6370370, 0.2843673, -0.5105182, 0.5166662
6: -2.6801329, -1.7317413, -2.6768987, -1.7538205, -0.6061006, 0.6193929
7: -10.3365517, -9.3110790, -10.3206358, -9.3132706, -0.6135318, 0.6064966
8: 7.6408663, 8.2799339, 7.6477637, 8.2772732, -0.4754407, 0.4697767
9: -5.9753113, -4.8490191, -5.9617333, -4.8537550, -0.8043323, 0.7974911

Time for backsubstitution: 22.18 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028812, upper bound: 0.3018787
time: 4.26 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028812, upper bound: 0.3020143
time: 4.13 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -6.2964015, -5.2146597, -6.3002758, -5.2145529, -0.6253591, 0.6288333
1: -7.0744586, -6.2863560, -7.0747571, -6.2834835, -0.5254736, 0.5228925
2: -7.7168264, -6.7525687, -7.7184181, -6.7522831, -0.5075693, 0.5089445
3: -5.6347475, -4.5218687, -5.6353989, -4.5183029, -0.7067323, 0.7037001
4: -7.8558125, -6.7367268, -7.8559370, -6.7324505, -0.6428523, 0.6387427
5: -0.6383338, 0.2873418, -0.6389225, 0.2878778, -0.5081596, 0.5085864
6: -2.6729426, -1.7536023, -2.6777694, -1.7531054, -0.6000273, 0.6049829
7: -10.3228350, -9.3168888, -10.3230829, -9.3127260, -0.6067822, 0.6024134
8: 7.6445141, 8.2785664, 7.6440535, 8.2786388, -0.4693761, 0.4699398
9: -5.9652209, -4.8517737, -5.9655581, -4.8502131, -0.7938452, 0.7919707

Time for backsubstitution: 22.27 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3018825
time: 4.57 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3020211
time: 4.30 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -6.3021374, -5.2060070, -6.3012505, -5.2145247, -0.6339412, 0.6386609
1: -7.0823383, -6.2804770, -7.0748301, -6.2827582, -0.5341101, 0.5284464
2: -7.7194943, -6.7438407, -7.7188158, -6.7522140, -0.5096760, 0.5183654
3: -5.6464744, -4.5161667, -5.6355605, -4.5174065, -0.7200127, 0.7087812
4: -7.8669467, -6.7285452, -7.8559680, -6.7313733, -0.6519001, 0.6458247
5: -0.6410499, 0.2898231, -0.6390734, 0.2880173, -0.5111296, 0.5157824
6: -2.6801329, -1.7317413, -2.6789851, -1.7529845, -0.6070933, 0.6214271
7: -10.3365517, -9.3110790, -10.3231411, -9.3116789, -0.6156287, 0.6087990
8: 7.6408663, 8.2799339, 7.6439390, 8.2786560, -0.4749429, 0.4716138
9: -5.9753113, -4.8490191, -5.9656420, -4.8498192, -0.8043666, 0.7971308

Time for backsubstitution: 22.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3018831
time: 3.85 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3020211
time: 5.10 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 31.75 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3018789
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3020145
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3018796
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020151
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3027468
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3018780, upper bound: 0.3028811
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3027468
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3020135, upper bound: 0.3028820
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3018787
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3020143
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3028812, upper bound: 0.3018787
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3028812, upper bound: 0.3020143
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3018825
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3020211
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3018831
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 31.75
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3020211

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -6.2907219, -5.2168140, -6.2907219, -5.2168140, -0.6201231, 0.6201229
1: -7.0688410, -6.2913294, -7.0688410, -6.2913294, -0.5183911, 0.5183911
2: -7.7145720, -6.7546062, -7.7145720, -6.7546062, -0.5041466, 0.5041466
3: -5.6302543, -4.5252872, -5.6302543, -4.5252872, -0.6992521, 0.6992521
4: -7.8532720, -6.7422695, -7.8532720, -6.7422695, -0.6352339, 0.6352344
5: -0.6362884, 0.2836895, -0.6362884, 0.2836895, -0.5040908, 0.5040905
6: -2.6708522, -1.7544346, -2.6708522, -1.7544346, -0.5966523, 0.5966523
7: -10.3203335, -9.3184814, -10.3203335, -9.3184814, -0.5981996, 0.5982001
8: 7.6483288, 8.2771826, 7.6483288, 8.2771826, -0.4659376, 0.4659376
9: -5.9613199, -4.8557110, -5.9613199, -4.8557110, -0.7879560, 0.7879560

Time for backsubstitution: 22.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018787, upper bound: 0.3018523
time: 6.91 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018786, upper bound: 0.3018793
time: 4.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -6.2907219, -5.2168140, -6.2964373, -5.2081637, -0.6277051, 0.6259084
1: -7.0688410, -6.2913294, -7.0767188, -6.2855148, -0.5245237, 0.5263343
2: -7.7145720, -6.7546062, -7.7172379, -6.7459521, -0.5122986, 0.5065312
3: -5.6302543, -4.5252872, -5.6419692, -4.5196428, -0.7049341, 0.7116356
4: -7.8532720, -6.7422695, -7.8643961, -6.7340708, -0.6434150, 0.6415799
5: -0.6362884, 0.2836895, -0.6389948, 0.2861705, -0.5069617, 0.5067635
6: -2.6708522, -1.7544346, -2.6780343, -1.7326282, -0.6120417, 0.6035039
7: -10.3203335, -9.3184814, -10.3340549, -9.3126698, -0.6043420, 0.6062570
8: 7.6483288, 8.2771826, 7.6446695, 8.2785454, -0.4674344, 0.4703300
9: -5.9613199, -4.8557110, -5.9713607, -4.8529534, -0.7909584, 0.7979214

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018787, upper bound: 0.3019884
time: 6.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018786, upper bound: 0.3020148
time: 4.41 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.2964373, -5.2081637, -6.2907219, -5.2168140, -0.6259081, 0.6277051
1: -7.0767188, -6.2855148, -7.0688410, -6.2913294, -0.5263338, 0.5245237
2: -7.7172379, -6.7459521, -7.7145720, -6.7546062, -0.5065312, 0.5122987
3: -5.6419692, -4.5196428, -5.6302543, -4.5252872, -0.7116356, 0.7049341
4: -7.8643961, -6.7340708, -7.8532720, -6.7422695, -0.6415799, 0.6434152
5: -0.6389948, 0.2861705, -0.6362884, 0.2836895, -0.5067635, 0.5069618
6: -2.6780343, -1.7326282, -2.6708522, -1.7544346, -0.6035044, 0.6120417
7: -10.3340549, -9.3126698, -10.3203335, -9.3184814, -0.6062570, 0.6043420
8: 7.6446695, 8.2785454, 7.6483288, 8.2771826, -0.4703300, 0.4674344
9: -5.9713607, -4.8529534, -5.9613199, -4.8557110, -0.7979214, 0.7909584

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020141, upper bound: 0.3018530
time: 3.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020141, upper bound: 0.3018793
time: 4.06 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.2964373, -5.2081637, -6.2964373, -5.2081637, -0.6305838, 0.6305840
1: -7.0767188, -6.2855148, -7.0767188, -6.2855148, -0.5276465, 0.5276465
2: -7.7172379, -6.7459521, -7.7172379, -6.7459521, -0.5129650, 0.5129650
3: -5.6419692, -4.5196428, -5.6419692, -4.5196428, -0.7117286, 0.7117286
4: -7.8643961, -6.7340708, -7.8643961, -6.7340708, -0.6460996, 0.6460998
5: -0.6389948, 0.2861705, -0.6389948, 0.2861705, -0.5150081, 0.5150084
6: -2.6780343, -1.7326282, -2.6780343, -1.7326282, -0.6136227, 0.6136227
7: -10.3340549, -9.3126698, -10.3340549, -9.3126698, -0.6083498, 0.6083500
8: 7.6446695, 8.2785454, 7.6446695, 8.2785454, -0.4722273, 0.4722272
9: -5.9713607, -4.8529534, -5.9713607, -4.8529534, -0.7979224, 0.7979219

Time for backsubstitution: 22.24 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.75 + 562.25 = 619.00 seconds
