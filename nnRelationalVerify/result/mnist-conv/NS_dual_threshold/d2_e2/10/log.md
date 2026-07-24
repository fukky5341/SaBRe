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
execution time: IAR + RelationalAnalysis = 22.04 + 33.66 = 55.70 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.3028891, upper bound: 0.3028899

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4596
type: A, layer: 1, pos: 4596
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 4596

## Relational analysis of NS_B1

### Relational analysis result of NS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028816, upper bound: 0.3020144
time: 5.94 seconds

## Relational analysis of NS_B2

### Relational analysis result of NS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028868, upper bound: 0.3028874
time: 3.78 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 9.93 seconds
NS_B1, status: Status.UNKNOWN, split count: 1, time: 9.93
Output dim: 8, lower bound: -0.3028816, upper bound: 0.3020144
NS_B2, status: Status.UNKNOWN, split count: 1, time: 9.93
Output dim: 8, lower bound: -0.3028868, upper bound: 0.3028874

## BFS NS instance: NS_B1

### Backsubstitution after applying NS history:
0: -6.2995672, -5.2146235, -6.2955713, -5.2166767, -0.6316323, 0.6300991
1: -7.0728297, -6.2829561, -7.0692120, -6.2877288, -0.5255990, 0.5265889
2: -7.7180443, -6.7522674, -7.7165623, -6.7542505, -0.5076892, 0.5086246
3: -5.6340637, -4.5174656, -5.6310658, -4.5208149, -0.7074161, 0.7072434
4: -7.8559632, -6.7332067, -7.8534293, -6.7369132, -0.6432934, 0.6446135
5: -0.6389737, 0.2867427, -0.6370356, 0.2843680, -0.5100636, 0.5102606
6: -2.6784763, -1.7530367, -2.6769042, -1.7538207, -0.6052699, 0.6047671
7: -10.3224125, -9.3118477, -10.3206367, -9.3132687, -0.6059132, 0.6062346
8: 7.6452265, 8.2785740, 7.6477642, 8.2772732, -0.4695172, 0.4685043
9: -5.9653687, -4.8512201, -5.9617362, -4.8537521, -0.7945571, 0.7937629

Time for backsubstitution: 21.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4596
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 4596

## Relational analysis of NS_B1_A1

### Relational analysis result of NS_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3020145
time: 3.98 seconds

## Relational analysis of NS_B1_A2

### Relational analysis result of NS_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3020141
time: 5.05 seconds

## BFS NS instance: NS_B2

### Backsubstitution after applying NS history:
0: -6.3012581, -5.2145243, -6.3012533, -5.2145228, -0.6356318, 0.6318319
1: -7.0748358, -6.2827549, -7.0748320, -6.2827568, -0.5324411, 0.5263784
2: -7.7188230, -6.7522120, -7.7188196, -6.7522140, -0.5105991, 0.5096536
3: -5.6355658, -4.5174041, -5.6355600, -4.5174046, -0.7119360, 0.7082853
4: -7.8559666, -6.7313643, -7.8559661, -6.7313695, -0.6441641, 0.6489432
5: -0.6390731, 0.2880220, -0.6390729, 0.2880161, -0.5106355, 0.5136383
6: -2.6789908, -1.7529849, -2.6789882, -1.7529861, -0.6068463, 0.6067934
7: -10.3231449, -9.3116760, -10.3231421, -9.3116751, -0.6086016, 0.6082339
8: 7.6439342, 8.2786570, 7.6439390, 8.2786560, -0.4722593, 0.4702854
9: -5.9656434, -4.8498139, -5.9656420, -4.8498163, -0.7948396, 0.7987924

Time for backsubstitution: 20.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 4596

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_B2_A1

### Relational analysis result of NS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3028872
time: 4.39 seconds

## Relational analysis of NS_B2_A2

### Relational analysis result of NS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028866, upper bound: 0.3028872
time: 3.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 29.33 seconds
NS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 29.33
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3020145
NS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 29.33
Output dim: 8, lower bound: -0.3020137, upper bound: 0.3020141
NS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 29.33
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3028872
NS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 29.33
Output dim: 8, lower bound: -0.3028866, upper bound: 0.3028872

## BFS NS instance: NS_B1_A1

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

Time for backsubstitution: 21.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_B1_A1_A1

### Relational analysis result of NS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018789, upper bound: 0.3020143
time: 4.06 seconds

## Relational analysis of NS_B1_A1_A2

### Relational analysis result of NS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020143
time: 3.74 seconds

## BFS NS instance: NS_B1_A2

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

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_B1_A2_B1

### Relational analysis result of NS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3018783
time: 4.93 seconds

## Relational analysis of NS_B1_A2_B2

### Relational analysis result of NS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020142
time: 5.56 seconds

## BFS NS instance: NS_B2_A1

### Backsubstitution after applying NS history:
0: -6.2964053, -5.2146602, -6.3002758, -5.2145529, -0.6291595, 0.6288333
1: -7.0744658, -6.2863560, -7.0747571, -6.2834835, -0.5315380, 0.5228937
2: -7.7168279, -6.7525692, -7.7184181, -6.7522831, -0.5085201, 0.5089452
3: -5.6347528, -4.5218687, -5.6353989, -4.5183029, -0.7103829, 0.7037005
4: -7.8558135, -6.7367229, -7.8559370, -6.7324505, -0.6428528, 0.6435206
5: -0.6383328, 0.2873452, -0.6389225, 0.2878778, -0.5081601, 0.5115902
6: -2.6729431, -1.7536018, -2.6777694, -1.7531054, -0.6000283, 0.6049321
7: -10.3228378, -9.3168888, -10.3230829, -9.3127260, -0.6071467, 0.6024151
8: 7.6445093, 8.2785664, 7.6440535, 8.2786388, -0.4713507, 0.4699399
9: -5.9652228, -4.8517694, -5.9655581, -4.8502131, -0.7938461, 0.7959263

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 4596

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_B2_A1_B1

### Relational analysis result of NS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3027511
time: 4.57 seconds

## Relational analysis of NS_B2_A1_B2

### Relational analysis result of NS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3028872
time: 4.22 seconds

## BFS NS instance: NS_B2_A2

### Backsubstitution after applying NS history:
0: -6.3021421, -5.2060089, -6.3012505, -5.2145247, -0.6377404, 0.6376603
1: -7.0823431, -6.2804747, -7.0748301, -6.2827582, -0.5401721, 0.5284479
2: -7.7194972, -6.7438393, -7.7188158, -6.7522140, -0.5106244, 0.5170361
3: -5.6464796, -4.5161662, -5.6355605, -4.5174065, -0.7227280, 0.7087827
4: -7.8669462, -6.7285409, -7.8559680, -6.7313733, -0.6502881, 0.6506045
5: -0.6410507, 0.2898269, -0.6390734, 0.2880173, -0.5111306, 0.5187857
6: -2.6801343, -1.7317412, -2.6789851, -1.7529845, -0.6070952, 0.6216207
7: -10.3365526, -9.3110771, -10.3231411, -9.3116789, -0.6154895, 0.6088002
8: 7.6408615, 8.2799349, 7.6439390, 8.2786560, -0.4769175, 0.4716140
9: -5.9753103, -4.8490162, -5.9656420, -4.8498192, -0.8043694, 0.8010874

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 4596

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_B2_A2_B1

### Relational analysis result of NS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3027510
time: 5.48 seconds

## Relational analysis of NS_B2_A2_B2

### Relational analysis result of NS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3028865
time: 5.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.71 seconds
NS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3018789, upper bound: 0.3020143
NS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020143
NS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3018783
NS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020142
NS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3027511
NS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3027506, upper bound: 0.3028872
NS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3027510
NS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 32.71
Output dim: 8, lower bound: -0.3028867, upper bound: 0.3028865

## BFS NS instance: NS_B1_A1_A1

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

Time for backsubstitution: 21.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_B1_A1_A1_B1

### Relational analysis result of NS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3018789
time: 4.61 seconds

## Relational analysis of NS_B1_A1_A1_B2

### Relational analysis result of NS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3020145
time: 4.84 seconds

## BFS NS instance: NS_B1_A1_A2

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

Time for backsubstitution: 21.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 481

## Relational analysis of NS_B1_A1_A2_B1

### Relational analysis result of NS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3018796
time: 3.66 seconds

## Relational analysis of NS_B1_A1_A2_B2

### Relational analysis result of NS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020151
time: 3.70 seconds

## BFS NS instance: NS_B1_A2_B1

### Backsubstitution after applying NS history:
0: -6.3002758, -5.2145529, -6.2907219, -5.2168140, -0.6304390, 0.6237266
1: -7.0747571, -6.2834835, -7.0688410, -6.2913294, -0.5242200, 0.5258753
2: -7.7184181, -6.7522831, -7.7145720, -6.7546062, -0.5080390, 0.5062408
3: -5.6353989, -4.5183029, -5.6302543, -4.5252872, -0.7044740, 0.7056828
4: -7.8559370, -6.7324505, -7.8532720, -6.7422695, -0.6378794, 0.6451638
5: -0.6389225, 0.2878778, -0.6362884, 0.2836895, -0.5079269, 0.5090404
6: -2.6777694, -1.7531054, -2.6708522, -1.7544346, -0.6039937, 0.5979867
7: -10.3230829, -9.3127260, -10.3203335, -9.3184814, -0.6011002, 0.6044805
8: 7.6440535, 8.2786388, 7.6483288, 8.2771826, -0.4704370, 0.4675463
9: -5.9655581, -4.8502131, -5.9613199, -4.8557110, -0.7918928, 0.7942080

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_B1_A2_B1_A1

### Relational analysis result of NS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027461, upper bound: 0.3018787
time: 3.68 seconds

## Relational analysis of NS_B1_A2_B1_A2

### Relational analysis result of NS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3018787
time: 3.83 seconds

## BFS NS instance: NS_B1_A2_B2

### Backsubstitution after applying NS history:
0: -6.3012505, -5.2145247, -6.2964373, -5.2081637, -0.6366930, 0.6322999
1: -7.0748301, -6.2827582, -7.0767188, -6.2855148, -0.5297737, 0.5345113
2: -7.7188158, -6.7522140, -7.7172379, -6.7459521, -0.5153468, 0.5083442
3: -5.6355605, -4.5174065, -5.6419692, -4.5196428, -0.7095675, 0.7180135
4: -7.8559680, -6.7313733, -7.8643961, -6.7340708, -0.6449933, 0.6497622
5: -0.6390734, 0.2880173, -0.6389948, 0.2861705, -0.5151154, 0.5120363
6: -2.6789851, -1.7529845, -2.6780343, -1.7326282, -0.6201777, 0.6050401
7: -10.3231411, -9.3116789, -10.3340549, -9.3126698, -0.6074848, 0.6128292
8: 7.6439390, 8.2786560, 7.6446695, 8.2785454, -0.4721079, 0.4731439
9: -5.9656420, -4.8498192, -5.9713607, -4.8529534, -0.7970517, 0.8047297

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 481
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 481

## Relational analysis of NS_B1_A2_B2_A1

### Relational analysis result of NS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3020143
time: 3.80 seconds

## Relational analysis of NS_B1_A2_B2_A2

### Relational analysis result of NS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3020143
time: 3.48 seconds

## BFS NS instance: NS_B2_A1_B1

### Backsubstitution after applying NS history:
0: -6.2964053, -5.2146602, -6.2964015, -5.2146597, -0.6277473, 0.6239469
1: -7.0744658, -6.2863560, -7.0744586, -6.2863560, -0.5287890, 0.5227263
2: -7.7168279, -6.7525692, -7.7168264, -6.7525687, -0.5082862, 0.5073278
3: -5.6347528, -4.5218687, -5.6347475, -4.5218687, -0.7068281, 0.7031755
4: -7.8558135, -6.7367229, -7.8558125, -6.7367268, -0.6385593, 0.6433380
5: -0.6383328, 0.2873452, -0.6383338, 0.2873418, -0.5068712, 0.5098772
6: -2.6729431, -1.7536018, -2.6729426, -1.7536023, -0.5996175, 0.5995655
7: -10.3228378, -9.3168888, -10.3228350, -9.3168888, -0.6025443, 0.6021819
8: 7.6445093, 8.2785664, 7.6445141, 8.2785664, -0.4712145, 0.4692405
9: -5.9652228, -4.8517694, -5.9652209, -4.8517737, -0.7916243, 0.7955787

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 4596

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_B2_A1_B1_A1

### Relational analysis result of NS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3027254
time: 4.01 seconds

## Relational analysis of NS_B2_A1_B1_A2

### Relational analysis result of NS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3027508
time: 3.78 seconds

## BFS NS instance: NS_B2_A1_B2

### Backsubstitution after applying NS history:
0: -6.2964053, -5.2146602, -6.3021374, -5.2060070, -0.6335647, 0.6297390
1: -7.0744658, -6.2863560, -7.0823383, -6.2804770, -0.5349741, 0.5306692
2: -7.7168279, -6.7525692, -7.7194943, -6.7438407, -0.5153414, 0.5097125
3: -5.6347528, -4.5218687, -5.6464744, -4.5161667, -0.7125525, 0.7155619
4: -7.8558135, -6.7367229, -7.8669467, -6.7285452, -0.6467090, 0.6474077
5: -0.6383328, 0.2873452, -0.6410499, 0.2898231, -0.5097497, 0.5125678
6: -2.6729431, -1.7536018, -2.6801329, -1.7317413, -0.6147056, 0.6064248
7: -10.3228378, -9.3168888, -10.3365517, -9.3110790, -0.6086893, 0.6098640
8: 7.6445093, 8.2785664, 7.6408663, 8.2799339, -0.4727159, 0.4736024
9: -5.9652228, -4.8517694, -5.9753113, -4.8490191, -0.7946236, 0.8055887

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 4596

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_B2_A1_B2_A1

### Relational analysis result of NS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3028612
time: 4.26 seconds

## Relational analysis of NS_B2_A1_B2_A2

### Relational analysis result of NS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3028870
time: 3.85 seconds

## BFS NS instance: NS_B2_A2_B1

### Backsubstitution after applying NS history:
0: -6.3021421, -5.2060089, -6.2964015, -5.2146597, -0.6335390, 0.6315472
1: -7.0823431, -6.2804747, -7.0744586, -6.2863560, -0.5367308, 0.5288577
2: -7.7194972, -6.7438393, -7.7168264, -6.7525687, -0.5106711, 0.5150168
3: -5.6464796, -4.5161662, -5.6347475, -4.5218687, -0.7182696, 0.7088461
4: -7.8669462, -6.7285409, -7.8558125, -6.7367268, -0.6449022, 0.6514878
5: -0.6410507, 0.2898269, -0.6383338, 0.2873418, -0.5095186, 0.5127556
6: -2.6801343, -1.7317412, -2.6729426, -1.7536023, -0.6064773, 0.6148984
7: -10.3365526, -9.3110771, -10.3228350, -9.3168888, -0.6097238, 0.6083252
8: 7.6408615, 8.2799349, 7.6445141, 8.2785664, -0.4755764, 0.4707406
9: -5.9753103, -4.8490162, -5.9652209, -4.8517737, -0.8015912, 0.7985804

Time for backsubstitution: 21.91 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 4596

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_B2_A2_B1_A1

### Relational analysis result of NS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027254
time: 4.08 seconds

## Relational analysis of NS_B2_A2_B1_A2

### Relational analysis result of NS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027508
time: 4.08 seconds

## BFS NS instance: NS_B2_A2_B2

### Backsubstitution after applying NS history:
0: -6.3021421, -5.2060089, -6.3021374, -5.2060070, -0.6382146, 0.6344171
1: -7.0823431, -6.2804747, -7.0823383, -6.2804770, -0.5380964, 0.5319827
2: -7.7194972, -6.7438393, -7.7194943, -6.7438407, -0.5171709, 0.5161040
3: -5.6464796, -4.5161662, -5.6464744, -4.5161667, -0.7193594, 0.7156448
4: -7.8669462, -6.7285409, -7.8669467, -6.7285452, -0.6493998, 0.6541803
5: -0.6410507, 0.2898269, -0.6410499, 0.2898231, -0.5177701, 0.5208192
6: -2.6801343, -1.7317412, -2.6801329, -1.7317413, -0.6166468, 0.6165936
7: -10.3365526, -9.3110771, -10.3365517, -9.3110790, -0.6126909, 0.6123688
8: 7.6408615, 8.2799349, 7.6408663, 8.2799339, -0.4774771, 0.4755021
9: -5.9753103, -4.8490162, -5.9753113, -4.8490191, -0.8015907, 0.8055921

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 4596

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_B2_A2_B2_A1

### Relational analysis result of NS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027254
time: 3.82 seconds

## Relational analysis of NS_B2_A2_B2_A2

### Relational analysis result of NS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027508
time: 3.89 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 29.74 seconds
NS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3018789
NS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3018788, upper bound: 0.3020145
NS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3018796
NS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3020143, upper bound: 0.3020151
NS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027461, upper bound: 0.3018787
NS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3018787
NS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3020143
NS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027460, upper bound: 0.3020143
NS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3027254
NS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3027508
NS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3028612
NS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3027504, upper bound: 0.3028870
NS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027254
NS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027508
NS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027254
NS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 29.74
Output dim: 8, lower bound: -0.3028864, upper bound: 0.3027508

## BFS NS instance: NS_B1_A1_A1_B1

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

Time for backsubstitution: 21.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_B1_A1_A1_B1_A1

### Relational analysis result of NS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018787, upper bound: 0.3018523
time: 6.41 seconds

## Relational analysis of NS_B1_A1_A1_B1_A2

### Relational analysis result of NS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018786, upper bound: 0.3018793
time: 4.59 seconds

## BFS NS instance: NS_B1_A1_A1_B2

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

Time for backsubstitution: 21.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 568

## Relational analysis of NS_B1_A1_A1_B2_B1

### Relational analysis result of NS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018523, upper bound: 0.3020149
time: 3.95 seconds

## Relational analysis of NS_B1_A1_A1_B2_B2

### Relational analysis result of NS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3018786, upper bound: 0.3020149
time: 3.69 seconds

## BFS NS instance: NS_B1_A1_A2_B1

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

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 568
type: B, layer: 1, pos: 568
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 568

## Relational analysis of NS_B1_A1_A2_B1_A1

### Relational analysis result of NS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020141, upper bound: 0.3018530
time: 3.88 seconds

## Relational analysis of NS_B1_A1_A2_B1_A2

### Relational analysis result of NS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.3020141, upper bound: 0.3018793
time: 4.02 seconds

## BFS NS instance: NS_B1_A1_A2_B2

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

Time for backsubstitution: 21.84 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 55.70 + 549.71 = 605.41 seconds
