## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.06738792


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0075597, 0.0052502, -0.0075597, 0.0052502, -0.0128099, 0.0128099)
1: (-0.0118563, 0.0272498, -0.0118563, 0.0272498, -0.0391061, 0.0391061)
2: (-0.0022177, 0.0311536, -0.0022177, 0.0311536, -0.0333713, 0.0333713)
3: (-0.0118823, 0.0112060, -0.0118823, 0.0112060, -0.0230883, 0.0230883)
4: (-0.0161376, 0.0112550, -0.0161376, 0.0112550, -0.0273926, 0.0273926)
5: (-0.0107015, 0.0236103, -0.0107015, 0.0236103, -0.0343117, 0.0343117)
6: (-0.0085504, 0.0127865, -0.0085504, 0.0127865, -0.0213370, 0.0213370)
7: (-0.0201290, 0.0113964, -0.0201290, 0.0113964, -0.0315255, 0.0315255)
8: (-0.0108657, 0.0156657, -0.0108657, 0.0156657, -0.0265314, 0.0265314)
9: (0.9239558, 1.0238240, 0.9239558, 1.0238240, -0.0998682, 0.0998682)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.93 + 7.85 = 9.78 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0842349, upper bound: 0.0842349

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788146, upper bound: 0.0808813
time: 2.53 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0820487, upper bound: 0.0820487
time: 1.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.28 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.28
Output dim: 9, lower bound: -0.0788146, upper bound: 0.0808813
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.28
Output dim: 9, lower bound: -0.0820487, upper bound: 0.0820487

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -0.0065652, 0.0026718, -0.0071848, 0.0038684, -0.0104336, 0.0098566
1: -0.0102531, 0.0181769, -0.0113282, 0.0238871, -0.0338818, 0.0295050
2: -0.0013153, 0.0232708, -0.0019622, 0.0279083, -0.0292236, 0.0252329
3: -0.0097537, 0.0052040, -0.0110818, 0.0091859, -0.0189396, 0.0162858
4: -0.0102376, 0.0101337, -0.0141980, 0.0108395, -0.0210771, 0.0243317
5: -0.0084056, 0.0163034, -0.0099646, 0.0210716, -0.0294772, 0.0262679
6: -0.0078791, 0.0067116, -0.0083080, 0.0106051, -0.0184843, 0.0150196
7: -0.0130579, 0.0094914, -0.0171506, 0.0107412, -0.0237991, 0.0266420
8: -0.0076712, 0.0089898, -0.0096628, 0.0131995, -0.0208707, 0.0186527
9: 0.9526412, 1.0210822, 0.9341077, 1.0229824, -0.0703411, 0.0869744

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
time: 1.70 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0791977
time: 2.29 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -0.0071626, 0.0037808, -0.0074524, 0.0047995, -0.0119621, 0.0112332
1: -0.0112746, 0.0231841, -0.0116981, 0.0261693, -0.0374439, 0.0348822
2: -0.0018828, 0.0275455, -0.0021282, 0.0301849, -0.0320677, 0.0296737
3: -0.0110197, 0.0085705, -0.0116470, 0.0105069, -0.0215267, 0.0202175
4: -0.0135320, 0.0107452, -0.0154485, 0.0111168, -0.0246488, 0.0261937
5: -0.0098067, 0.0203117, -0.0104604, 0.0227410, -0.0325477, 0.0307721
6: -0.0082699, 0.0099827, -0.0084740, 0.0120448, -0.0203147, 0.0184567
7: -0.0165102, 0.0106105, -0.0191806, 0.0111848, -0.0276949, 0.0297911
8: -0.0095841, 0.0125812, -0.0105149, 0.0148496, -0.0244337, 0.0230960
9: 0.9368398, 1.0228157, 0.9273748, 1.0235513, -0.0867116, 0.0954409

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0810761, upper bound: 0.0800363
time: 1.65 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0803199, upper bound: 0.0803199
time: 1.90 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.53 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0791977
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 9, lower bound: -0.0810761, upper bound: 0.0800363
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 5.53
Output dim: 9, lower bound: -0.0803199, upper bound: 0.0803199

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -0.0065515, 0.0026016, -0.0071111, 0.0035811, -0.0101326, 0.0097127
1: -0.0100609, 0.0181206, -0.0105893, 0.0235514, -0.0336123, 0.0287099
2: -0.0011491, 0.0231808, -0.0012752, 0.0273712, -0.0285203, 0.0244560
3: -0.0097205, 0.0051889, -0.0109147, 0.0091099, -0.0188304, 0.0161035
4: -0.0102113, 0.0100344, -0.0140436, 0.0104129, -0.0206243, 0.0240780
5: -0.0082748, 0.0162776, -0.0093113, 0.0209201, -0.0291949, 0.0255889
6: -0.0078592, 0.0066913, -0.0082291, 0.0104057, -0.0182649, 0.0149204
7: -0.0130279, 0.0094376, -0.0167909, 0.0104929, -0.0235208, 0.0262285
8: -0.0076186, 0.0089758, -0.0093467, 0.0130323, -0.0206509, 0.0183225
9: 0.9527102, 1.0207635, 0.9346704, 1.0215917, -0.0688815, 0.0860931

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0789348
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0791983
time: 1.82 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -0.0065496, 0.0026049, -0.0077765, 0.0060378, -0.0125873, 0.0103814
1: -0.0100857, 0.0181169, -0.0118183, 0.0303054, -0.0403911, 0.0299351
2: -0.0011734, 0.0231758, -0.0021170, 0.0329528, -0.0341263, 0.0252928
3: -0.0097160, 0.0051866, -0.0123267, 0.0135642, -0.0232802, 0.0175132
4: -0.0102071, 0.0100511, -0.0185745, 0.0114567, -0.0216638, 0.0286256
5: -0.0082907, 0.0162726, -0.0109956, 0.0265472, -0.0348379, 0.0272682
6: -0.0078588, 0.0066883, -0.0087181, 0.0151211, -0.0229799, 0.0154064
7: -0.0130244, 0.0094353, -0.0224177, 0.0118767, -0.0249012, 0.0318529
8: -0.0076124, 0.0089733, -0.0112697, 0.0182490, -0.0258614, 0.0202431
9: 0.9527237, 1.0208144, 0.9132261, 1.0237620, -0.0710382, 0.1075883

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672260, upper bound: 0.0716527
time: 2.15 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0668462, upper bound: 0.0692036
time: 1.53 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0074235, 0.0047269, -0.0118183, 0.0109272
1: -0.0105355, 0.0228542, -0.0115187, 0.0260828, -0.0366183, 0.0343729
2: -0.0011940, 0.0270286, -0.0019568, 0.0300458, -0.0312398, 0.0289854
3: -0.0108587, 0.0084947, -0.0115892, 0.0104878, -0.0213465, 0.0200838
4: -0.0133733, 0.0103177, -0.0153733, 0.0110136, -0.0243869, 0.0256909
5: -0.0091526, 0.0201610, -0.0102034, 0.0227025, -0.0318552, 0.0303644
6: -0.0081909, 0.0097837, -0.0084544, 0.0119668, -0.0201577, 0.0182381
7: -0.0161654, 0.0103601, -0.0190837, 0.0111057, -0.0272711, 0.0294438
8: -0.0092823, 0.0124182, -0.0103771, 0.0148076, -0.0240900, 0.0227954
9: 0.9373958, 1.0214237, 0.9275211, 1.0231279, -0.0857322, 0.0939026

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0769766
time: 2.00 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0795968
time: 1.74 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0074250, 0.0047223, -0.0124406, 0.0131953
1: -0.0117038, 0.0293341, -0.0115404, 0.0260793, -0.0377831, 0.0408745
2: -0.0020246, 0.0323330, -0.0019864, 0.0300386, -0.0320632, 0.0343195
3: -0.0121859, 0.0128294, -0.0115902, 0.0104853, -0.0226713, 0.0244196
4: -0.0177987, 0.0113304, -0.0153804, 0.0110279, -0.0288266, 0.0267108
5: -0.0107822, 0.0256450, -0.0102448, 0.0226974, -0.0334796, 0.0358898
6: -0.0086578, 0.0143819, -0.0084539, 0.0119710, -0.0206288, 0.0228359
7: -0.0216183, 0.0116937, -0.0190767, 0.0111088, -0.0327272, 0.0307704
8: -0.0110805, 0.0174725, -0.0103895, 0.0148036, -0.0258841, 0.0278620
9: 0.9165484, 1.0235258, 0.9275364, 1.0231968, -0.1066484, 0.0959895

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0791977, upper bound: 0.0772707
time: 1.97 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0791977, upper bound: 0.0798269
time: 1.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 5.64 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0789348
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0791983
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0672260, upper bound: 0.0716527
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0668462, upper bound: 0.0692036
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0769766
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0795968
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0791977, upper bound: 0.0772707
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 5.64
Output dim: 9, lower bound: -0.0791977, upper bound: 0.0798269

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0071111, 0.0035811, -0.0100905, 0.0095067
1: -0.0094869, 0.0179468, -0.0105893, 0.0235514, -0.0330383, 0.0285361
2: -0.0006456, 0.0229024, -0.0012752, 0.0273712, -0.0280168, 0.0241776
3: -0.0096166, 0.0051424, -0.0109147, 0.0091099, -0.0187265, 0.0160571
4: -0.0101306, 0.0097369, -0.0140436, 0.0104129, -0.0205436, 0.0237804
5: -0.0078843, 0.0162008, -0.0093113, 0.0209201, -0.0288044, 0.0255120
6: -0.0077993, 0.0066288, -0.0082291, 0.0104057, -0.0182050, 0.0148579
7: -0.0129358, 0.0092739, -0.0167909, 0.0104929, -0.0234287, 0.0260649
8: -0.0074538, 0.0089322, -0.0093467, 0.0130323, -0.0204861, 0.0182789
9: 0.9529222, 1.0198088, 0.9346704, 1.0215917, -0.0686694, 0.0851384

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0723091
time: 1.82 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0071111, 0.0035811, -0.0106381, 0.0103781
1: -0.0105610, 0.0234683, -0.0105893, 0.0235514, -0.0341124, 0.0340576
2: -0.0014662, 0.0266610, -0.0012752, 0.0273712, -0.0288374, 0.0279362
3: -0.0107759, 0.0093700, -0.0109147, 0.0091099, -0.0198858, 0.0202847
4: -0.0145209, 0.0105283, -0.0140436, 0.0104129, -0.0249338, 0.0245719
5: -0.0094379, 0.0213327, -0.0093113, 0.0209201, -0.0303580, 0.0306440
6: -0.0082263, 0.0106983, -0.0082291, 0.0104057, -0.0186321, 0.0189275
7: -0.0167035, 0.0105577, -0.0167909, 0.0104929, -0.0271964, 0.0273486
8: -0.0089806, 0.0132137, -0.0093467, 0.0130323, -0.0220130, 0.0225604
9: 0.9339586, 1.0218134, 0.9346704, 1.0215917, -0.0876330, 0.0871430

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702680, upper bound: 0.0707374
time: 1.58 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 2.91 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0065496, 0.0026049, -0.0075497, 0.0050221, -0.0115717, 0.0101547
1: -0.0100857, 0.0181169, -0.0114680, 0.0281221, -0.0382078, 0.0295849
2: -0.0011734, 0.0231758, -0.0019465, 0.0310523, -0.0322258, 0.0251223
3: -0.0097160, 0.0051866, -0.0118334, 0.0121681, -0.0218841, 0.0170199
4: -0.0102071, 0.0100511, -0.0172463, 0.0111816, -0.0213886, 0.0272974
5: -0.0082907, 0.0162726, -0.0105270, 0.0248505, -0.0331411, 0.0267995
6: -0.0078588, 0.0066883, -0.0085609, 0.0136963, -0.0215551, 0.0152492
7: -0.0130244, 0.0094353, -0.0206805, 0.0114575, -0.0244820, 0.0301157
8: -0.0076124, 0.0089733, -0.0105833, 0.0165898, -0.0242022, 0.0195566
9: 0.9527237, 1.0208144, 0.9199262, 1.0232189, -0.0704951, 0.1008883

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_B1_B1

### Relational analysis result of NS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672260, upper bound: 0.0702500
time: 1.16 seconds

## Relational analysis of NS_A1_B2_B1_B2

### Relational analysis result of NS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0672260, upper bound: 0.0716527
time: 3.12 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0063244, 0.0023698, -0.0068978, 0.0039669, -0.0102912, 0.0092676
1: -0.0097452, 0.0156362, -0.0125940, 0.0195560, -0.0293012, 0.0282302
2: -0.0009142, 0.0215708, -0.0030022, 0.0250749, -0.0259891, 0.0245731
3: -0.0092416, 0.0031836, -0.0104646, 0.0057283, -0.0149700, 0.0136482
4: -0.0082574, 0.0097747, -0.0114732, 0.0112635, -0.0195208, 0.0212479
5: -0.0077162, 0.0139381, -0.0103631, 0.0174337, -0.0251499, 0.0243013
6: -0.0076872, 0.0048494, -0.0083267, 0.0076421, -0.0153293, 0.0131761
7: -0.0113622, 0.0089026, -0.0142733, 0.0108367, -0.0221989, 0.0231758
8: -0.0069942, 0.0069410, -0.0087282, 0.0094660, -0.0164602, 0.0156692
9: 0.9617499, 1.0201558, 0.9497761, 1.0247197, -0.0629699, 0.0703797

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0690997
time: 1.31 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.81 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0065515, 0.0026016, -0.0096930, 0.0100551
1: -0.0105355, 0.0228542, -0.0100609, 0.0181206, -0.0286561, 0.0329151
2: -0.0011940, 0.0270286, -0.0011491, 0.0231808, -0.0243747, 0.0281777
3: -0.0108587, 0.0084947, -0.0097205, 0.0051889, -0.0160476, 0.0182152
4: -0.0133733, 0.0103177, -0.0102113, 0.0100344, -0.0234077, 0.0205290
5: -0.0091526, 0.0201610, -0.0082748, 0.0162776, -0.0254303, 0.0284358
6: -0.0081909, 0.0097837, -0.0078592, 0.0066913, -0.0148822, 0.0176429
7: -0.0161654, 0.0103601, -0.0130279, 0.0094376, -0.0256030, 0.0233881
8: -0.0092823, 0.0124182, -0.0076186, 0.0089758, -0.0182581, 0.0200368
9: 0.9373958, 1.0214237, 0.9527102, 1.0207635, -0.0833678, 0.0687135

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0769766
time: 1.81 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0769766
time: 1.85 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0071354, 0.0037131, -0.0108044, 0.0106391
1: -0.0105355, 0.0228542, -0.0110912, 0.0231018, -0.0336373, 0.0339454
2: -0.0011940, 0.0270286, -0.0017115, 0.0274158, -0.0286098, 0.0287401
3: -0.0108587, 0.0084947, -0.0109653, 0.0085518, -0.0194105, 0.0194600
4: -0.0133733, 0.0103177, -0.0134544, 0.0106406, -0.0240138, 0.0237721
5: -0.0091526, 0.0201610, -0.0095471, 0.0202745, -0.0294271, 0.0297081
6: -0.0081909, 0.0097837, -0.0082503, 0.0099021, -0.0180930, 0.0180341
7: -0.0161654, 0.0103601, -0.0164249, 0.0105317, -0.0266971, 0.0267850
8: -0.0092823, 0.0124182, -0.0094530, 0.0125414, -0.0218237, 0.0218713
9: 0.9373958, 1.0214237, 0.9369770, 1.0223881, -0.0849923, 0.0844467

Time for backsubstitution: 1.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0795968
time: 2.07 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0795968
time: 1.81 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0065496, 0.0026049, -0.0103233, 0.0123199
1: -0.0117038, 0.0293341, -0.0100857, 0.0181169, -0.0298206, 0.0394198
2: -0.0020246, 0.0323330, -0.0011734, 0.0231758, -0.0252004, 0.0335065
3: -0.0121859, 0.0128294, -0.0097160, 0.0051866, -0.0173725, 0.0225454
4: -0.0177987, 0.0113304, -0.0102071, 0.0100511, -0.0278498, 0.0215374
5: -0.0107822, 0.0256450, -0.0082907, 0.0162726, -0.0270548, 0.0339357
6: -0.0086578, 0.0143819, -0.0078588, 0.0066883, -0.0153461, 0.0222407
7: -0.0216183, 0.0116937, -0.0130244, 0.0094353, -0.0310536, 0.0247181
8: -0.0110805, 0.0174725, -0.0076124, 0.0089733, -0.0200538, 0.0250848
9: 0.9165484, 1.0235258, 0.9527237, 1.0208144, -0.1042660, 0.0708021

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716527, upper bound: 0.0672260
time: 1.42 seconds

## Relational analysis of NS_A2_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0692036, upper bound: 0.0668462
time: 3.41 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0071365, 0.0037083, -0.0114267, 0.0129068
1: -0.0117038, 0.0293341, -0.0111134, 0.0230966, -0.0348003, 0.0404475
2: -0.0020246, 0.0323330, -0.0017412, 0.0274064, -0.0294310, 0.0340742
3: -0.0121859, 0.0128294, -0.0109655, 0.0085489, -0.0207348, 0.0237949
4: -0.0177987, 0.0113304, -0.0134615, 0.0106550, -0.0284537, 0.0247919
5: -0.0107822, 0.0256450, -0.0095893, 0.0202682, -0.0310505, 0.0352343
6: -0.0086578, 0.0143819, -0.0082499, 0.0099066, -0.0185644, 0.0226318
7: -0.0216183, 0.0116937, -0.0164154, 0.0105347, -0.0321530, 0.0281091
8: -0.0110805, 0.0174725, -0.0094640, 0.0125363, -0.0236168, 0.0269364
9: 0.9165484, 1.0235258, 0.9369969, 1.0224582, -0.1059098, 0.0865289

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716527, upper bound: 0.0701976
time: 2.75 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0692036, upper bound: 0.0698457
time: 1.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 6.20 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0723091
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0702680, upper bound: 0.0707374
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0672260, upper bound: 0.0702500
NS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0672260, upper bound: 0.0716527
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0690997
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0769766
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0769766
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0795968
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0798250, upper bound: 0.0795968
NS_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0716527, upper bound: 0.0672260
NS_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0692036, upper bound: 0.0668462
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0716527, upper bound: 0.0701976
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 6.20
Output dim: 9, lower bound: -0.0692036, upper bound: 0.0698457

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0069203, 0.0030751, -0.0095846, 0.0093158
1: -0.0094869, 0.0179468, -0.0102987, 0.0215966, -0.0310835, 0.0282455
2: -0.0006456, 0.0229024, -0.0011070, 0.0258105, -0.0264561, 0.0240094
3: -0.0096166, 0.0051424, -0.0105095, 0.0077512, -0.0173678, 0.0156519
4: -0.0101306, 0.0097369, -0.0127673, 0.0101872, -0.0203179, 0.0225042
5: -0.0078843, 0.0162008, -0.0088847, 0.0192923, -0.0271767, 0.0250855
6: -0.0077993, 0.0066288, -0.0080975, 0.0090651, -0.0168644, 0.0147263
7: -0.0129358, 0.0092739, -0.0152941, 0.0101122, -0.0230480, 0.0245681
8: -0.0074538, 0.0089322, -0.0088023, 0.0114927, -0.0189465, 0.0177345
9: 0.9529222, 1.0198088, 0.9410744, 1.0211055, -0.0681833, 0.0787343

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0715980
time: 3.82 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0731205
time: 2.45 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0062850, 0.0021481, -0.0063539, 0.0034747, -0.0097598, 0.0085020
1: -0.0091328, 0.0154668, -0.0115721, 0.0142883, -0.0234212, 0.0270389
2: -0.0003904, 0.0213073, -0.0022248, 0.0213923, -0.0217827, 0.0235321
3: -0.0091449, 0.0031386, -0.0093275, 0.0015680, -0.0107128, 0.0124661
4: -0.0081821, 0.0094647, -0.0071005, 0.0105321, -0.0187142, 0.0165653
5: -0.0073080, 0.0138632, -0.0089329, 0.0123356, -0.0196436, 0.0227961
6: -0.0076258, 0.0047888, -0.0079386, 0.0036341, -0.0112599, 0.0127273
7: -0.0112734, 0.0087440, -0.0105742, 0.0096498, -0.0209232, 0.0193182
8: -0.0068400, 0.0068996, -0.0072339, 0.0052493, -0.0120893, 0.0141335
9: 0.9619496, 1.0191458, 0.9688113, 1.0228453, -0.0608957, 0.0503346

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0689735
time: 1.15 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 1.70 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0071111, 0.0035811, -0.0104553, 0.0100587
1: -0.0102804, 0.0216467, -0.0105893, 0.0235514, -0.0338318, 0.0322360
2: -0.0013008, 0.0254146, -0.0012752, 0.0273712, -0.0286720, 0.0266898
3: -0.0103910, 0.0080113, -0.0109147, 0.0091099, -0.0195009, 0.0189260
4: -0.0132094, 0.0103480, -0.0140436, 0.0104129, -0.0236223, 0.0243916
5: -0.0090217, 0.0197614, -0.0093113, 0.0209201, -0.0299418, 0.0290726
6: -0.0080985, 0.0094561, -0.0082291, 0.0104057, -0.0185042, 0.0176852
7: -0.0155814, 0.0101845, -0.0167909, 0.0104929, -0.0260743, 0.0269755
8: -0.0084719, 0.0118298, -0.0093467, 0.0130323, -0.0215042, 0.0211765
9: 0.9400309, 1.0213395, 0.9346704, 1.0215917, -0.0815608, 0.0866691

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
time: 3.60 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0707374
time: 2.05 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0063058, 0.0034586, -0.0068717, 0.0029705, -0.0092763, 0.0103303
1: -0.0114675, 0.0142369, -0.0102107, 0.0207587, -0.0322262, 0.0244476
2: -0.0023040, 0.0210116, -0.0010276, 0.0253194, -0.0276235, 0.0220392
3: -0.0091753, 0.0018162, -0.0104052, 0.0070517, -0.0162270, 0.0122214
4: -0.0075261, 0.0106122, -0.0120892, 0.0101002, -0.0176263, 0.0227014
5: -0.0089659, 0.0127776, -0.0087126, 0.0184891, -0.0274550, 0.0214902
6: -0.0079179, 0.0039991, -0.0080493, 0.0084311, -0.0163490, 0.0120484
7: -0.0108489, 0.0096242, -0.0147104, 0.0099613, -0.0208102, 0.0243347
8: -0.0068596, 0.0055431, -0.0086592, 0.0108022, -0.0176619, 0.0142023
9: 0.9677531, 1.0228958, 0.9441981, 1.0209155, -0.0531625, 0.0786977

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 1.43 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.74 seconds

## BFS NS instance: NS_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -0.0065496, 0.0026049, -0.0068742, 0.0029476, -0.0094972, 0.0094792
1: -0.0100857, 0.0181169, -0.0102804, 0.0216467, -0.0317324, 0.0283973
2: -0.0011734, 0.0231758, -0.0013008, 0.0254146, -0.0265880, 0.0244766
3: -0.0097160, 0.0051866, -0.0103910, 0.0080113, -0.0177273, 0.0155776
4: -0.0102071, 0.0100511, -0.0132094, 0.0103480, -0.0205551, 0.0232605
5: -0.0082907, 0.0162726, -0.0090217, 0.0197614, -0.0280520, 0.0252943
6: -0.0078588, 0.0066883, -0.0080985, 0.0094561, -0.0173149, 0.0147868
7: -0.0130244, 0.0094353, -0.0155814, 0.0101845, -0.0232090, 0.0250167
8: -0.0076124, 0.0089733, -0.0084719, 0.0118298, -0.0194422, 0.0174452
9: 0.9527237, 1.0208144, 0.9400309, 1.0213395, -0.0686158, 0.0807835

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702395
time: 1.80 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702500
time: 3.35 seconds

## BFS NS instance: NS_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0065496, 0.0026049, -0.0074868, 0.0047480, -0.0112975, 0.0100917
1: -0.0100857, 0.0181169, -0.0113465, 0.0271213, -0.0372070, 0.0294634
2: -0.0011734, 0.0231758, -0.0018529, 0.0303596, -0.0315331, 0.0250287
3: -0.0097160, 0.0051866, -0.0116776, 0.0114324, -0.0211484, 0.0168642
4: -0.0102071, 0.0100511, -0.0164778, 0.0110528, -0.0212598, 0.0265290
5: -0.0082907, 0.0162726, -0.0103079, 0.0239427, -0.0322333, 0.0265805
6: -0.0078588, 0.0066883, -0.0084978, 0.0129430, -0.0208018, 0.0151860
7: -0.0130244, 0.0094353, -0.0198323, 0.0112676, -0.0242920, 0.0292676
8: -0.0076124, 0.0089733, -0.0103719, 0.0157998, -0.0234122, 0.0193452
9: 0.9527237, 1.0208144, 0.9233138, 1.0229756, -0.0702518, 0.0975006

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716066
time: 3.31 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0062850, 0.0021481, -0.0068978, 0.0039669, -0.0102519, 0.0090459
1: -0.0091328, 0.0154668, -0.0125940, 0.0195560, -0.0286889, 0.0280608
2: -0.0003904, 0.0213073, -0.0030022, 0.0250749, -0.0254653, 0.0243096
3: -0.0091449, 0.0031386, -0.0104646, 0.0057283, -0.0148732, 0.0136032
4: -0.0081821, 0.0094647, -0.0114732, 0.0112635, -0.0194456, 0.0209379
5: -0.0073080, 0.0138632, -0.0103631, 0.0174337, -0.0247418, 0.0242264
6: -0.0076258, 0.0047888, -0.0083267, 0.0076421, -0.0152679, 0.0131155
7: -0.0112734, 0.0087440, -0.0142733, 0.0108367, -0.0221101, 0.0230173
8: -0.0068400, 0.0068996, -0.0087282, 0.0094660, -0.0163060, 0.0156278
9: 0.9619496, 1.0191458, 0.9497761, 1.0247197, -0.0627701, 0.0693697

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0667246
time: 0.99 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0690997
time: 1.02 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0068342, 0.0028674, -0.0068978, 0.0039669, -0.0108010, 0.0097652
1: -0.0102035, 0.0208754, -0.0125940, 0.0195560, -0.0297596, 0.0334694
2: -0.0012271, 0.0249953, -0.0030022, 0.0250749, -0.0263020, 0.0279975
3: -0.0103002, 0.0073569, -0.0104646, 0.0057283, -0.0160285, 0.0178216
4: -0.0125772, 0.0102682, -0.0114732, 0.0112635, -0.0238407, 0.0217414
5: -0.0088654, 0.0190138, -0.0103631, 0.0174337, -0.0262991, 0.0293770
6: -0.0080558, 0.0088575, -0.0083267, 0.0076421, -0.0156979, 0.0171842
7: -0.0150379, 0.0100487, -0.0142733, 0.0108367, -0.0258746, 0.0243219
8: -0.0083491, 0.0111824, -0.0087282, 0.0094660, -0.0178151, 0.0199107
9: 0.9428862, 1.0211681, 0.9497761, 1.0247197, -0.0818335, 0.0713920

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0670587
time: 2.48 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.48 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0065094, 0.0023956, -0.0094870, 0.0100131
1: -0.0105355, 0.0228542, -0.0094869, 0.0179468, -0.0284823, 0.0323411
2: -0.0011940, 0.0270286, -0.0006456, 0.0229024, -0.0240964, 0.0276742
3: -0.0108587, 0.0084947, -0.0096166, 0.0051424, -0.0160011, 0.0181113
4: -0.0133733, 0.0103177, -0.0101306, 0.0097369, -0.0231101, 0.0204483
5: -0.0091526, 0.0201610, -0.0078843, 0.0162008, -0.0253534, 0.0280453
6: -0.0081909, 0.0097837, -0.0077993, 0.0066288, -0.0148197, 0.0175830
7: -0.0161654, 0.0103601, -0.0129358, 0.0092739, -0.0254394, 0.0232959
8: -0.0092823, 0.0124182, -0.0074538, 0.0089322, -0.0182146, 0.0198720
9: 0.9373958, 1.0214237, 0.9529222, 1.0198088, -0.0824130, 0.0685015

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0669953
time: 1.41 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 3.34 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0070570, 0.0032670, -0.0103584, 0.0105607
1: -0.0105355, 0.0228542, -0.0105610, 0.0234683, -0.0340039, 0.0334152
2: -0.0011940, 0.0270286, -0.0014662, 0.0266610, -0.0278549, 0.0284948
3: -0.0108587, 0.0084947, -0.0107759, 0.0093700, -0.0202287, 0.0192706
4: -0.0133733, 0.0103177, -0.0145209, 0.0105283, -0.0239016, 0.0248386
5: -0.0091526, 0.0201610, -0.0094379, 0.0213327, -0.0304853, 0.0295989
6: -0.0081909, 0.0097837, -0.0082263, 0.0106983, -0.0188892, 0.0180101
7: -0.0161654, 0.0103601, -0.0167035, 0.0105577, -0.0267231, 0.0270636
8: -0.0092823, 0.0124182, -0.0089806, 0.0132137, -0.0224960, 0.0213989
9: 0.9373958, 1.0214237, 0.9339586, 1.0218134, -0.0844176, 0.0874650

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B2_B1

### Relational analysis result of NS_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707374, upper bound: 0.0702680
time: 1.94 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2

### Relational analysis result of NS_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 2.38 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0070914, 0.0035037, -0.0105951, 0.0105951
1: -0.0105355, 0.0228542, -0.0105355, 0.0228542, -0.0333897, 0.0333897
2: -0.0011940, 0.0270286, -0.0011940, 0.0270286, -0.0282226, 0.0282226
3: -0.0108587, 0.0084947, -0.0108587, 0.0084947, -0.0193534, 0.0193534
4: -0.0133733, 0.0103177, -0.0133733, 0.0103177, -0.0236909, 0.0236909
5: -0.0091526, 0.0201610, -0.0091526, 0.0201610, -0.0293136, 0.0293136
6: -0.0081909, 0.0097837, -0.0081909, 0.0097837, -0.0179746, 0.0179746
7: -0.0161654, 0.0103601, -0.0161654, 0.0103601, -0.0265255, 0.0265255
8: -0.0092823, 0.0124182, -0.0092823, 0.0124182, -0.0217006, 0.0217006
9: 0.9373958, 1.0214237, 0.9373958, 1.0214237, -0.0840279, 0.0840279

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740425, upper bound: 0.0699814
time: 1.45 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 1.57 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0077183, 0.0057703, -0.0128617, 0.0112220
1: -0.0105355, 0.0228542, -0.0117038, 0.0293341, -0.0398696, 0.0345580
2: -0.0011940, 0.0270286, -0.0020246, 0.0323330, -0.0335270, 0.0290533
3: -0.0108587, 0.0084947, -0.0121859, 0.0128294, -0.0236881, 0.0206806
4: -0.0133733, 0.0103177, -0.0177987, 0.0113304, -0.0247036, 0.0281164
5: -0.0091526, 0.0201610, -0.0107822, 0.0256450, -0.0347976, 0.0309432
6: -0.0081909, 0.0097837, -0.0086578, 0.0143819, -0.0225728, 0.0184416
7: -0.0161654, 0.0103601, -0.0216183, 0.0116937, -0.0278591, 0.0319785
8: -0.0092823, 0.0124182, -0.0110805, 0.0174725, -0.0267548, 0.0234988
9: 0.9373958, 1.0214237, 0.9165484, 1.0235258, -0.0861301, 0.1048753

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717283, upper bound: 0.0730510
time: 1.76 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 1.55 seconds

## BFS NS instance: NS_A2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0065496, 0.0026049, -0.0100917, 0.0112975
1: -0.0113465, 0.0271213, -0.0100857, 0.0181169, -0.0294634, 0.0372070
2: -0.0018529, 0.0303596, -0.0011734, 0.0231758, -0.0250287, 0.0315331
3: -0.0116776, 0.0114324, -0.0097160, 0.0051866, -0.0168642, 0.0211484
4: -0.0164778, 0.0110528, -0.0102071, 0.0100511, -0.0265290, 0.0212598
5: -0.0103079, 0.0239427, -0.0082907, 0.0162726, -0.0265805, 0.0322333
6: -0.0084978, 0.0129430, -0.0078588, 0.0066883, -0.0151860, 0.0208018
7: -0.0198323, 0.0112676, -0.0130244, 0.0094353, -0.0292676, 0.0242920
8: -0.0103719, 0.0157998, -0.0076124, 0.0089733, -0.0193452, 0.0234122
9: 0.9233138, 1.0229756, 0.9527237, 1.0208144, -0.0975006, 0.0702518

Time for backsubstitution: 1.91 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_A2_B1_A1_B1

### Relational analysis result of NS_A2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
time: 2.44 seconds

## Relational analysis of NS_A2_A2_B1_A1_B2

### Relational analysis result of NS_A2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
time: 1.57 seconds

## BFS NS instance: NS_A2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0068255, 0.0039014, -0.0063244, 0.0023698, -0.0091953, 0.0102258
1: -0.0124919, 0.0187966, -0.0097452, 0.0156362, -0.0281282, 0.0285418
2: -0.0029244, 0.0245749, -0.0009142, 0.0215708, -0.0244952, 0.0254891
3: -0.0103170, 0.0050977, -0.0092416, 0.0031836, -0.0135006, 0.0143393
4: -0.0107890, 0.0111772, -0.0082574, 0.0097747, -0.0205637, 0.0194345
5: -0.0101817, 0.0166798, -0.0077162, 0.0139381, -0.0241198, 0.0243960
6: -0.0082751, 0.0070263, -0.0076872, 0.0048494, -0.0131245, 0.0147135
7: -0.0136281, 0.0106762, -0.0113622, 0.0089026, -0.0225307, 0.0220384
8: -0.0085291, 0.0088583, -0.0069942, 0.0069410, -0.0154701, 0.0158525
9: 0.9525756, 1.0245186, 0.9617499, 1.0201558, -0.0675802, 0.0627688

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_A2_B1_A2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.98 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.67 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0071365, 0.0037083, -0.0111951, 0.0118844
1: -0.0113465, 0.0271213, -0.0111134, 0.0230966, -0.0344431, 0.0382346
2: -0.0018529, 0.0303596, -0.0017412, 0.0274064, -0.0292593, 0.0321008
3: -0.0116776, 0.0114324, -0.0109655, 0.0085489, -0.0202265, 0.0223979
4: -0.0164778, 0.0110528, -0.0134615, 0.0106550, -0.0271328, 0.0245143
5: -0.0103079, 0.0239427, -0.0095893, 0.0202682, -0.0305761, 0.0335320
6: -0.0084978, 0.0129430, -0.0082499, 0.0099066, -0.0184044, 0.0211928
7: -0.0198323, 0.0112676, -0.0164154, 0.0105347, -0.0303670, 0.0276830
8: -0.0103719, 0.0157998, -0.0094640, 0.0125363, -0.0229082, 0.0252637
9: 0.9233138, 1.0229756, 0.9369969, 1.0224582, -0.0991444, 0.0859786

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
time: 3.24 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
time: 1.62 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0068255, 0.0039014, -0.0068783, 0.0030699, -0.0098954, 0.0107797
1: -0.0124919, 0.0187966, -0.0107288, 0.0202950, -0.0327869, 0.0295254
2: -0.0029244, 0.0245749, -0.0014982, 0.0252069, -0.0281314, 0.0260731
3: -0.0103170, 0.0050977, -0.0104163, 0.0065271, -0.0168441, 0.0155140
4: -0.0107890, 0.0111772, -0.0115364, 0.0103423, -0.0211313, 0.0227136
5: -0.0101817, 0.0166798, -0.0089835, 0.0178510, -0.0280327, 0.0256633
6: -0.0082751, 0.0070263, -0.0080661, 0.0079270, -0.0162021, 0.0150924
7: -0.0136281, 0.0106762, -0.0142337, 0.0099964, -0.0236245, 0.0249099
8: -0.0085291, 0.0088583, -0.0087062, 0.0103023, -0.0188314, 0.0175645
9: 0.9525756, 1.0245186, 0.9464571, 1.0217748, -0.0691992, 0.0780615

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
time: 1.60 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
time: 1.30 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.93 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0715980
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0731205
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0689735
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0707374
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702395
NS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702500
NS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716066
NS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
NS_A1_B2_B2_A1_B1, status: Status.VERIFIED, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0667246
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0690997
NS_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0670587
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
NS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0669953
NS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
NS_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0707374, upper bound: 0.0702680
NS_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0740425, upper bound: 0.0699814
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0717283, upper bound: 0.0730510
NS_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
NS_A2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
NS_A2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
NS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
NS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
NS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.93
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0063378, 0.0022185, -0.0087279, 0.0087334
1: -0.0094869, 0.0179468, -0.0092294, 0.0163209, -0.0258078, 0.0271762
2: -0.0006456, 0.0229024, -0.0004811, 0.0218100, -0.0224556, 0.0233834
3: -0.0096166, 0.0051424, -0.0092587, 0.0038791, -0.0134957, 0.0144011
4: -0.0101306, 0.0097369, -0.0089070, 0.0095615, -0.0196922, 0.0186439
5: -0.0078843, 0.0162008, -0.0074959, 0.0147249, -0.0226092, 0.0236967
6: -0.0077993, 0.0066288, -0.0076797, 0.0054675, -0.0132668, 0.0143085
7: -0.0129358, 0.0092739, -0.0119043, 0.0089149, -0.0218507, 0.0211782
8: -0.0074538, 0.0089322, -0.0069978, 0.0076339, -0.0150877, 0.0159300
9: 0.9529222, 1.0198088, 0.9586030, 1.0193583, -0.0664361, 0.0612057

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0668036, upper bound: 0.0702358
time: 1.64 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0715980
time: 1.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0068715, 0.0029689, -0.0094783, 0.0092671
1: -0.0094869, 0.0179468, -0.0102067, 0.0208570, -0.0303439, 0.0281535
2: -0.0006456, 0.0229024, -0.0010219, 0.0253591, -0.0260047, 0.0239243
3: -0.0096166, 0.0051424, -0.0103941, 0.0071250, -0.0167416, 0.0155365
4: -0.0101306, 0.0097369, -0.0120788, 0.0100930, -0.0202236, 0.0218157
5: -0.0078843, 0.0162008, -0.0086981, 0.0185246, -0.0264090, 0.0248988
6: -0.0077993, 0.0066288, -0.0080465, 0.0084473, -0.0162466, 0.0146752
7: -0.0129358, 0.0092739, -0.0146922, 0.0099509, -0.0228868, 0.0239662
8: -0.0074538, 0.0089322, -0.0086411, 0.0108853, -0.0183390, 0.0175733
9: 0.9529222, 1.0198088, 0.9438275, 1.0209014, -0.0679792, 0.0759813

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678337, upper bound: 0.0726350
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0731205
time: 2.94 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0062345, 0.0020555, -0.0063197, 0.0034069, -0.0096414, 0.0083752
1: -0.0091802, 0.0152972, -0.0114840, 0.0141402, -0.0233204, 0.0267812
2: -0.0005008, 0.0209957, -0.0021632, 0.0211862, -0.0216869, 0.0231589
3: -0.0090136, 0.0030956, -0.0092456, 0.0014892, -0.0105028, 0.0123411
4: -0.0080963, 0.0094393, -0.0069921, 0.0104701, -0.0185664, 0.0164314
5: -0.0070919, 0.0138535, -0.0087957, 0.0122270, -0.0193189, 0.0226492
6: -0.0075771, 0.0047348, -0.0079031, 0.0035472, -0.0111244, 0.0126379
7: -0.0111262, 0.0086327, -0.0104528, 0.0095363, -0.0206625, 0.0190855
8: -0.0064511, 0.0069367, -0.0070588, 0.0051850, -0.0116362, 0.0139955
9: 0.9618829, 1.0191309, 0.9691509, 1.0226840, -0.0608010, 0.0499800

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0670096
time: 1.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0689735
time: 1.41 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0062533, 0.0020826, -0.0063498, 0.0034680, -0.0097213, 0.0084325
1: -0.0090321, 0.0153494, -0.0115600, 0.0142738, -0.0233059, 0.0269094
2: -0.0003345, 0.0211109, -0.0022177, 0.0213633, -0.0216978, 0.0233285
3: -0.0090477, 0.0031019, -0.0093151, 0.0015628, -0.0106105, 0.0124170
4: -0.0081243, 0.0094109, -0.0070923, 0.0105252, -0.0186495, 0.0165032
5: -0.0071809, 0.0138042, -0.0089170, 0.0123273, -0.0195082, 0.0227212
6: -0.0075891, 0.0047445, -0.0079342, 0.0036279, -0.0112170, 0.0126787
7: -0.0112032, 0.0086332, -0.0105649, 0.0096359, -0.0208391, 0.0191981
8: -0.0066281, 0.0068659, -0.0072073, 0.0052449, -0.0118729, 0.0140732
9: 0.9621280, 1.0189878, 0.9688357, 1.0228257, -0.0606977, 0.0501521

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
time: 4.25 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 2.70 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0065094, 0.0023956, -0.0092698, 0.0094571
1: -0.0102804, 0.0216467, -0.0094869, 0.0179468, -0.0282272, 0.0311336
2: -0.0013008, 0.0254146, -0.0006456, 0.0229024, -0.0242032, 0.0260602
3: -0.0103910, 0.0080113, -0.0096166, 0.0051424, -0.0155334, 0.0176279
4: -0.0132094, 0.0103480, -0.0101306, 0.0097369, -0.0229462, 0.0204786
5: -0.0090217, 0.0197614, -0.0078843, 0.0162008, -0.0252225, 0.0276457
6: -0.0080985, 0.0094561, -0.0077993, 0.0066288, -0.0147272, 0.0172554
7: -0.0155814, 0.0101845, -0.0129358, 0.0092739, -0.0248554, 0.0231204
8: -0.0084719, 0.0118298, -0.0074538, 0.0089322, -0.0174041, 0.0192836
9: 0.9400309, 1.0213395, 0.9529222, 1.0198088, -0.0797779, 0.0684173

Time for backsubstitution: 2.16 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0695874, upper bound: 0.0673037
time: 3.52 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
time: 1.96 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0070914, 0.0035037, -0.0103779, 0.0100390
1: -0.0102804, 0.0216467, -0.0105355, 0.0228542, -0.0331346, 0.0321822
2: -0.0013008, 0.0254146, -0.0011940, 0.0270286, -0.0283294, 0.0266085
3: -0.0103910, 0.0080113, -0.0108587, 0.0084947, -0.0188857, 0.0188700
4: -0.0132094, 0.0103480, -0.0133733, 0.0103177, -0.0235270, 0.0237212
5: -0.0090217, 0.0197614, -0.0091526, 0.0201610, -0.0291827, 0.0289140
6: -0.0080985, 0.0094561, -0.0081909, 0.0097837, -0.0178822, 0.0176470
7: -0.0155814, 0.0101845, -0.0161654, 0.0103601, -0.0259416, 0.0263500
8: -0.0084719, 0.0118298, -0.0092823, 0.0124182, -0.0208901, 0.0211121
9: 0.9400309, 1.0213395, 0.9373958, 1.0214237, -0.0813928, 0.0839438

Time for backsubstitution: 2.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0686353, upper bound: 0.0690829
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0707374
time: 1.74 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0063058, 0.0034586, -0.0062850, 0.0021481, -0.0084539, 0.0097437
1: -0.0114675, 0.0142369, -0.0091328, 0.0154668, -0.0269344, 0.0233697
2: -0.0023040, 0.0210116, -0.0003904, 0.0213073, -0.0236114, 0.0214020
3: -0.0091753, 0.0018162, -0.0091449, 0.0031386, -0.0123139, 0.0109610
4: -0.0075261, 0.0106122, -0.0081821, 0.0094647, -0.0169908, 0.0187944
5: -0.0089659, 0.0127776, -0.0073080, 0.0138632, -0.0228292, 0.0200857
6: -0.0079179, 0.0039991, -0.0076258, 0.0047888, -0.0127066, 0.0116249
7: -0.0108489, 0.0096242, -0.0112734, 0.0087440, -0.0195929, 0.0208976
8: -0.0068596, 0.0055431, -0.0068400, 0.0068996, -0.0137592, 0.0123831
9: 0.9677531, 1.0228958, 0.9619496, 1.0191458, -0.0513928, 0.0609462

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0652147, upper bound: 0.0659473
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 1.43 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0063058, 0.0034586, -0.0068350, 0.0028899, -0.0091958, 0.0102936
1: -0.0114675, 0.0142369, -0.0101356, 0.0201205, -0.0315880, 0.0243725
2: -0.0023040, 0.0210116, -0.0009514, 0.0249395, -0.0272435, 0.0219630
3: -0.0091753, 0.0018162, -0.0103104, 0.0064883, -0.0156636, 0.0121265
4: -0.0075261, 0.0106122, -0.0114520, 0.0100159, -0.0175420, 0.0220642
5: -0.0089659, 0.0127776, -0.0085530, 0.0177793, -0.0267452, 0.0213306
6: -0.0079179, 0.0039991, -0.0080060, 0.0078609, -0.0157788, 0.0120051
7: -0.0108489, 0.0096242, -0.0141464, 0.0098218, -0.0206706, 0.0237706
8: -0.0068596, 0.0055431, -0.0085326, 0.0102598, -0.0171195, 0.0140757
9: 0.9677531, 1.0228958, 0.9466549, 1.0207385, -0.0529854, 0.0762410

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0652147, upper bound: 0.0682734
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 2.03 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0068742, 0.0029476, -0.0094571, 0.0092698
1: -0.0094869, 0.0179468, -0.0102804, 0.0216467, -0.0311336, 0.0282272
2: -0.0006456, 0.0229024, -0.0013008, 0.0254146, -0.0260602, 0.0242032
3: -0.0096166, 0.0051424, -0.0103910, 0.0080113, -0.0176279, 0.0155334
4: -0.0101306, 0.0097369, -0.0132094, 0.0103480, -0.0204786, 0.0229462
5: -0.0078843, 0.0162008, -0.0090217, 0.0197614, -0.0276457, 0.0252225
6: -0.0077993, 0.0066288, -0.0080985, 0.0094561, -0.0172554, 0.0147272
7: -0.0129358, 0.0092739, -0.0155814, 0.0101845, -0.0231204, 0.0248554
8: -0.0074538, 0.0089322, -0.0084719, 0.0118298, -0.0192836, 0.0174041
9: 0.9529222, 1.0198088, 0.9400309, 1.0213395, -0.0684173, 0.0797779

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0662044, upper bound: 0.0695820
time: 1.73 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702395
time: 1.60 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0068742, 0.0029476, -0.0100046, 0.0101412
1: -0.0105610, 0.0234683, -0.0102804, 0.0216467, -0.0322077, 0.0337487
2: -0.0014662, 0.0266610, -0.0013008, 0.0254146, -0.0268808, 0.0279618
3: -0.0107759, 0.0093700, -0.0103910, 0.0080113, -0.0187872, 0.0197610
4: -0.0145209, 0.0105283, -0.0132094, 0.0103480, -0.0248689, 0.0237377
5: -0.0094379, 0.0213327, -0.0090217, 0.0197614, -0.0291992, 0.0303544
6: -0.0082263, 0.0106983, -0.0080985, 0.0094561, -0.0176824, 0.0187968
7: -0.0167035, 0.0105577, -0.0155814, 0.0101845, -0.0268880, 0.0261391
8: -0.0089806, 0.0132137, -0.0084719, 0.0118298, -0.0208104, 0.0216856
9: 0.9339586, 1.0218134, 0.9400309, 1.0213395, -0.0873809, 0.0817825

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0646220, upper bound: 0.0684970
time: 1.67 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702500
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0074868, 0.0047480, -0.0112574, 0.0098824
1: -0.0094869, 0.0179468, -0.0113465, 0.0271213, -0.0366082, 0.0292933
2: -0.0006456, 0.0229024, -0.0018529, 0.0303596, -0.0310053, 0.0247553
3: -0.0096166, 0.0051424, -0.0116776, 0.0114324, -0.0210490, 0.0168200
4: -0.0101306, 0.0097369, -0.0164778, 0.0110528, -0.0211834, 0.0262147
5: -0.0078843, 0.0162008, -0.0103079, 0.0239427, -0.0318270, 0.0265087
6: -0.0077993, 0.0066288, -0.0084978, 0.0129430, -0.0207423, 0.0151265
7: -0.0129358, 0.0092739, -0.0198323, 0.0112676, -0.0242034, 0.0291063
8: -0.0074538, 0.0089322, -0.0103719, 0.0157998, -0.0232536, 0.0193041
9: 0.9529222, 1.0198088, 0.9233138, 1.0229756, -0.0700533, 0.0964950

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_B2_A1_B1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0662044, upper bound: 0.0709357
time: 1.73 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716066
time: 3.41 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0074868, 0.0047480, -0.0118050, 0.0107538
1: -0.0105610, 0.0234683, -0.0113465, 0.0271213, -0.0376823, 0.0348148
2: -0.0014662, 0.0266610, -0.0018529, 0.0303596, -0.0318258, 0.0285139
3: -0.0107759, 0.0093700, -0.0116776, 0.0114324, -0.0222083, 0.0210476
4: -0.0145209, 0.0105283, -0.0164778, 0.0110528, -0.0255737, 0.0270062
5: -0.0094379, 0.0213327, -0.0103079, 0.0239427, -0.0333806, 0.0316406
6: -0.0082263, 0.0106983, -0.0084978, 0.0129430, -0.0211693, 0.0191961
7: -0.0167035, 0.0105577, -0.0198323, 0.0112676, -0.0279710, 0.0303900
8: -0.0089806, 0.0132137, -0.0103719, 0.0157998, -0.0247804, 0.0235856
9: 0.9339586, 1.0218134, 0.9233138, 1.0229756, -0.0890169, 0.0984996

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_B2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
time: 3.30 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
time: 1.84 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0062850, 0.0021481, -0.0068255, 0.0039014, -0.0101864, 0.0089736
1: -0.0091328, 0.0154668, -0.0124919, 0.0187966, -0.0279295, 0.0279588
2: -0.0003904, 0.0213073, -0.0029244, 0.0245749, -0.0249653, 0.0242318
3: -0.0091449, 0.0031386, -0.0103170, 0.0050977, -0.0142426, 0.0134556
4: -0.0081821, 0.0094647, -0.0107890, 0.0111772, -0.0193593, 0.0202537
5: -0.0073080, 0.0138632, -0.0101817, 0.0166798, -0.0239879, 0.0240449
6: -0.0076258, 0.0047888, -0.0082751, 0.0070263, -0.0146521, 0.0130639
7: -0.0112734, 0.0087440, -0.0136281, 0.0106762, -0.0219496, 0.0223721
8: -0.0068400, 0.0068996, -0.0085291, 0.0088583, -0.0156983, 0.0154287
9: 0.9619496, 1.0191458, 0.9525756, 1.0245186, -0.0625690, 0.0665703

Time for backsubstitution: 2.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659973, upper bound: 0.0669314
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
time: 1.78 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0067967, 0.0027754, -0.0068933, 0.0039578, -0.0107545, 0.0096686
1: -0.0101060, 0.0207606, -0.0125816, 0.0195408, -0.0296468, 0.0333421
2: -0.0011671, 0.0248293, -0.0029943, 0.0250517, -0.0262187, 0.0278237
3: -0.0102120, 0.0073227, -0.0104528, 0.0057237, -0.0159357, 0.0177754
4: -0.0125204, 0.0102099, -0.0114651, 0.0112557, -0.0237762, 0.0216750
5: -0.0087339, 0.0189557, -0.0103459, 0.0174257, -0.0261596, 0.0293016
6: -0.0080189, 0.0088134, -0.0083219, 0.0076360, -0.0156549, 0.0171353
7: -0.0149673, 0.0099363, -0.0142635, 0.0108219, -0.0257892, 0.0241998
8: -0.0081553, 0.0111491, -0.0087029, 0.0094618, -0.0176170, 0.0198520
9: 0.9430635, 1.0210063, 0.9497988, 1.0246987, -0.0816352, 0.0712075

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
time: 2.91 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.79 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0065094, 0.0023956, -0.0092671, 0.0094783
1: -0.0102067, 0.0208570, -0.0094869, 0.0179468, -0.0281535, 0.0303439
2: -0.0010219, 0.0253591, -0.0006456, 0.0229024, -0.0239243, 0.0260047
3: -0.0103941, 0.0071250, -0.0096166, 0.0051424, -0.0155365, 0.0167416
4: -0.0120788, 0.0100930, -0.0101306, 0.0097369, -0.0218157, 0.0202236
5: -0.0086981, 0.0185246, -0.0078843, 0.0162008, -0.0248988, 0.0264090
6: -0.0080465, 0.0084473, -0.0077993, 0.0066288, -0.0146752, 0.0162466
7: -0.0146922, 0.0099509, -0.0129358, 0.0092739, -0.0239662, 0.0228868
8: -0.0086411, 0.0108853, -0.0074538, 0.0089322, -0.0175733, 0.0183390
9: 0.9438275, 1.0209014, 0.9529222, 1.0198088, -0.0759813, 0.0679792

Time for backsubstitution: 2.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726350, upper bound: 0.0678337
time: 2.44 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
time: 1.48 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0063068, 0.0034323, -0.0062850, 0.0021481, -0.0084549, 0.0097174
1: -0.0114994, 0.0137490, -0.0091328, 0.0154668, -0.0269663, 0.0228818
2: -0.0021608, 0.0209964, -0.0003904, 0.0213073, -0.0234682, 0.0213868
3: -0.0092192, 0.0010684, -0.0091449, 0.0031386, -0.0123578, 0.0102132
4: -0.0065257, 0.0104626, -0.0081821, 0.0094647, -0.0159905, 0.0186447
5: -0.0087947, 0.0117027, -0.0073080, 0.0138632, -0.0226579, 0.0190108
6: -0.0079002, 0.0031316, -0.0076258, 0.0047888, -0.0126890, 0.0107574
7: -0.0100647, 0.0095251, -0.0112734, 0.0087440, -0.0188087, 0.0207985
8: -0.0070877, 0.0047627, -0.0068400, 0.0068996, -0.0139873, 0.0116027
9: 0.9710250, 1.0226896, 0.9619496, 1.0191458, -0.0481209, 0.0607400

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0689735, upper bound: 0.0664895
time: 1.55 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
time: 3.48 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0068742, 0.0029476, -0.0100390, 0.0103779
1: -0.0105355, 0.0228542, -0.0102804, 0.0216467, -0.0321822, 0.0331346
2: -0.0011940, 0.0270286, -0.0013008, 0.0254146, -0.0266085, 0.0283294
3: -0.0108587, 0.0084947, -0.0103910, 0.0080113, -0.0188700, 0.0188857
4: -0.0133733, 0.0103177, -0.0132094, 0.0103480, -0.0237212, 0.0235270
5: -0.0091526, 0.0201610, -0.0090217, 0.0197614, -0.0289140, 0.0291827
6: -0.0081909, 0.0097837, -0.0080985, 0.0094561, -0.0176470, 0.0178822
7: -0.0161654, 0.0103601, -0.0155814, 0.0101845, -0.0263500, 0.0259416
8: -0.0092823, 0.0124182, -0.0084719, 0.0118298, -0.0211121, 0.0208901
9: 0.9373958, 1.0214237, 0.9400309, 1.0213395, -0.0839438, 0.0813928

Time for backsubstitution: 2.18 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690829, upper bound: 0.0687197
time: 1.78 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707374, upper bound: 0.0702681
time: 2.03 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0068364, 0.0028922, -0.0063058, 0.0034586, -0.0102950, 0.0091981
1: -0.0101376, 0.0201230, -0.0114675, 0.0142369, -0.0243745, 0.0315906
2: -0.0009517, 0.0249459, -0.0023040, 0.0210116, -0.0219633, 0.0272499
3: -0.0103133, 0.0064884, -0.0091753, 0.0018162, -0.0121295, 0.0156637
4: -0.0114524, 0.0100163, -0.0075261, 0.0106122, -0.0220646, 0.0175423
5: -0.0085547, 0.0177797, -0.0089659, 0.0127776, -0.0213323, 0.0267456
6: -0.0080066, 0.0078611, -0.0079179, 0.0039991, -0.0120057, 0.0157790
7: -0.0141469, 0.0098233, -0.0108489, 0.0096242, -0.0237711, 0.0206722
8: -0.0085372, 0.0102599, -0.0068596, 0.0055431, -0.0140802, 0.0171195
9: 0.9466540, 1.0207405, 0.9677531, 1.0228958, -0.0762418, 0.0529875

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682734, upper bound: 0.0652203
time: 1.84 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 3.48 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0070914, 0.0035037, -0.0103752, 0.0100603
1: -0.0102067, 0.0208570, -0.0105355, 0.0228542, -0.0330610, 0.0313925
2: -0.0010219, 0.0253591, -0.0011940, 0.0270286, -0.0280506, 0.0265531
3: -0.0103941, 0.0071250, -0.0108587, 0.0084947, -0.0188887, 0.0179837
4: -0.0120788, 0.0100930, -0.0133733, 0.0103177, -0.0223965, 0.0234663
5: -0.0086981, 0.0185246, -0.0091526, 0.0201610, -0.0288591, 0.0276772
6: -0.0080465, 0.0084473, -0.0081909, 0.0097837, -0.0178302, 0.0166382
7: -0.0146922, 0.0099509, -0.0161654, 0.0103601, -0.0250523, 0.0261164
8: -0.0086411, 0.0108853, -0.0092823, 0.0124182, -0.0210594, 0.0201676
9: 0.9438275, 1.0209014, 0.9373958, 1.0214237, -0.0775962, 0.0835057

Time for backsubstitution: 2.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0737349, upper bound: 0.0700827
time: 1.50 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749766, upper bound: 0.0717021
time: 1.87 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0063068, 0.0034323, -0.0068364, 0.0028922, -0.0091990, 0.0102687
1: -0.0114994, 0.0137490, -0.0101376, 0.0201230, -0.0316225, 0.0238865
2: -0.0021608, 0.0209964, -0.0009517, 0.0249459, -0.0271067, 0.0219481
3: -0.0092192, 0.0010684, -0.0103133, 0.0064884, -0.0157076, 0.0113817
4: -0.0065257, 0.0104626, -0.0114524, 0.0100163, -0.0165420, 0.0219150
5: -0.0087947, 0.0117027, -0.0085547, 0.0177797, -0.0265744, 0.0202574
6: -0.0079002, 0.0031316, -0.0080066, 0.0078611, -0.0157614, 0.0111382
7: -0.0100647, 0.0095251, -0.0141469, 0.0098233, -0.0198880, 0.0236720
8: -0.0070877, 0.0047627, -0.0085372, 0.0102599, -0.0173475, 0.0132998
9: 0.9710250, 1.0226896, 0.9466540, 1.0207405, -0.0497155, 0.0760356

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705570, upper bound: 0.0698083
time: 2.07 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
time: 1.83 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0074868, 0.0047480, -0.0118394, 0.0109905
1: -0.0105355, 0.0228542, -0.0113465, 0.0271213, -0.0376568, 0.0342007
2: -0.0011940, 0.0270286, -0.0018529, 0.0303596, -0.0315536, 0.0288816
3: -0.0108587, 0.0084947, -0.0116776, 0.0114324, -0.0222911, 0.0201723
4: -0.0133733, 0.0103177, -0.0164778, 0.0110528, -0.0244260, 0.0267955
5: -0.0091526, 0.0201610, -0.0103079, 0.0239427, -0.0330953, 0.0304689
6: -0.0081909, 0.0097837, -0.0084978, 0.0129430, -0.0211339, 0.0182815
7: -0.0161654, 0.0103601, -0.0198323, 0.0112676, -0.0274330, 0.0301924
8: -0.0092823, 0.0124182, -0.0103719, 0.0157998, -0.0250821, 0.0227901
9: 0.9373958, 1.0214237, 0.9233138, 1.0229756, -0.0855798, 0.0981099

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 1.73 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 2.17 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0068364, 0.0028922, -0.0068255, 0.0039014, -0.0107378, 0.0097178
1: -0.0101376, 0.0201230, -0.0124919, 0.0187966, -0.0289342, 0.0326150
2: -0.0009517, 0.0249459, -0.0029244, 0.0245749, -0.0255266, 0.0278703
3: -0.0103133, 0.0064884, -0.0103170, 0.0050977, -0.0154110, 0.0168054
4: -0.0114524, 0.0100163, -0.0107890, 0.0111772, -0.0226295, 0.0208053
5: -0.0085547, 0.0177797, -0.0101817, 0.0166798, -0.0252345, 0.0279614
6: -0.0080066, 0.0078611, -0.0082751, 0.0070263, -0.0150329, 0.0161362
7: -0.0141469, 0.0098233, -0.0136281, 0.0106762, -0.0248231, 0.0234514
8: -0.0085372, 0.0102599, -0.0085291, 0.0088583, -0.0173955, 0.0187890
9: 0.9466540, 1.0207405, 0.9525756, 1.0245186, -0.0778646, 0.0681649

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0694977, upper bound: 0.0684055
time: 2.50 seconds

## Relational analysis of NS_A2_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 1.49 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0065094, 0.0023956, -0.0098824, 0.0112574
1: -0.0113465, 0.0271213, -0.0094869, 0.0179468, -0.0292933, 0.0366082
2: -0.0018529, 0.0303596, -0.0006456, 0.0229024, -0.0247553, 0.0310053
3: -0.0116776, 0.0114324, -0.0096166, 0.0051424, -0.0168200, 0.0210490
4: -0.0164778, 0.0110528, -0.0101306, 0.0097369, -0.0262147, 0.0211834
5: -0.0103079, 0.0239427, -0.0078843, 0.0162008, -0.0265087, 0.0318270
6: -0.0084978, 0.0129430, -0.0077993, 0.0066288, -0.0151265, 0.0207423
7: -0.0198323, 0.0112676, -0.0129358, 0.0092739, -0.0291063, 0.0242034
8: -0.0103719, 0.0157998, -0.0074538, 0.0089322, -0.0193041, 0.0232536
9: 0.9233138, 1.0229756, 0.9529222, 1.0198088, -0.0964950, 0.0700533

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0709357, upper bound: 0.0662981
time: 1.30 seconds

## Relational analysis of NS_A2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
time: 1.46 seconds

## BFS NS instance: NS_A2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0070570, 0.0032670, -0.0107538, 0.0118050
1: -0.0113465, 0.0271213, -0.0105610, 0.0234683, -0.0348148, 0.0376823
2: -0.0018529, 0.0303596, -0.0014662, 0.0266610, -0.0285139, 0.0318258
3: -0.0116776, 0.0114324, -0.0107759, 0.0093700, -0.0210476, 0.0222083
4: -0.0164778, 0.0110528, -0.0145209, 0.0105283, -0.0270062, 0.0255737
5: -0.0103079, 0.0239427, -0.0094379, 0.0213327, -0.0316406, 0.0333806
6: -0.0084978, 0.0129430, -0.0082263, 0.0106983, -0.0191961, 0.0211693
7: -0.0198323, 0.0112676, -0.0167035, 0.0105577, -0.0303900, 0.0279710
8: -0.0103719, 0.0157998, -0.0089806, 0.0132137, -0.0235856, 0.0247804
9: 0.9233138, 1.0229756, 0.9339586, 1.0218134, -0.0984996, 0.0890169

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
time: 2.45 seconds

## Relational analysis of NS_A2_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
time: 1.56 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0068255, 0.0039014, -0.0062850, 0.0021481, -0.0089736, 0.0101864
1: -0.0124919, 0.0187966, -0.0091328, 0.0154668, -0.0279588, 0.0279295
2: -0.0029244, 0.0245749, -0.0003904, 0.0213073, -0.0242318, 0.0249653
3: -0.0103170, 0.0050977, -0.0091449, 0.0031386, -0.0134556, 0.0142426
4: -0.0107890, 0.0111772, -0.0081821, 0.0094647, -0.0202537, 0.0193593
5: -0.0101817, 0.0166798, -0.0073080, 0.0138632, -0.0240449, 0.0239879
6: -0.0082751, 0.0070263, -0.0076258, 0.0047888, -0.0130639, 0.0146521
7: -0.0136281, 0.0106762, -0.0112734, 0.0087440, -0.0223721, 0.0219496
8: -0.0085291, 0.0088583, -0.0068400, 0.0068996, -0.0154287, 0.0156983
9: 0.9525756, 1.0245186, 0.9619496, 1.0191458, -0.0665703, 0.0625690

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669314, upper bound: 0.0651608
time: 1.83 seconds

## Relational analysis of NS_A2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.74 seconds

## BFS NS instance: NS_A2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068255, 0.0039014, -0.0068342, 0.0028674, -0.0096929, 0.0107356
1: -0.0124919, 0.0187966, -0.0102035, 0.0208754, -0.0333673, 0.0290002
2: -0.0029244, 0.0245749, -0.0012271, 0.0249953, -0.0279197, 0.0258020
3: -0.0103170, 0.0050977, -0.0103002, 0.0073569, -0.0176739, 0.0153979
4: -0.0107890, 0.0111772, -0.0125772, 0.0102682, -0.0210572, 0.0237544
5: -0.0101817, 0.0166798, -0.0088654, 0.0190138, -0.0291955, 0.0255452
6: -0.0082751, 0.0070263, -0.0080558, 0.0088575, -0.0171326, 0.0150821
7: -0.0136281, 0.0106762, -0.0150379, 0.0100487, -0.0236768, 0.0257141
8: -0.0085291, 0.0088583, -0.0083491, 0.0111824, -0.0197115, 0.0172075
9: 0.9525756, 1.0245186, 0.9428862, 1.0211681, -0.0685925, 0.0816324

Time for backsubstitution: 2.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0670587, upper bound: 0.0643809
time: 1.69 seconds

## Relational analysis of NS_A2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.47 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0070914, 0.0035037, -0.0109905, 0.0118394
1: -0.0113465, 0.0271213, -0.0105355, 0.0228542, -0.0342007, 0.0376568
2: -0.0018529, 0.0303596, -0.0011940, 0.0270286, -0.0288816, 0.0315536
3: -0.0116776, 0.0114324, -0.0108587, 0.0084947, -0.0201723, 0.0222911
4: -0.0164778, 0.0110528, -0.0133733, 0.0103177, -0.0267955, 0.0244260
5: -0.0103079, 0.0239427, -0.0091526, 0.0201610, -0.0304689, 0.0330953
6: -0.0084978, 0.0129430, -0.0081909, 0.0097837, -0.0182815, 0.0211339
7: -0.0198323, 0.0112676, -0.0161654, 0.0103601, -0.0301924, 0.0274330
8: -0.0103719, 0.0157998, -0.0092823, 0.0124182, -0.0227901, 0.0250821
9: 0.9233138, 1.0229756, 0.9373958, 1.0214237, -0.0981099, 0.0855798

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
time: 1.53 seconds

## Relational analysis of NS_A2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
time: 1.57 seconds

## BFS NS instance: NS_A2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0077183, 0.0057703, -0.0132571, 0.0124663
1: -0.0113465, 0.0271213, -0.0117038, 0.0293341, -0.0406806, 0.0388250
2: -0.0018529, 0.0303596, -0.0020246, 0.0323330, -0.0341860, 0.0323843
3: -0.0116776, 0.0114324, -0.0121859, 0.0128294, -0.0245070, 0.0236183
4: -0.0164778, 0.0110528, -0.0177987, 0.0113304, -0.0278082, 0.0288515
5: -0.0103079, 0.0239427, -0.0107822, 0.0256450, -0.0359529, 0.0347249
6: -0.0084978, 0.0129430, -0.0086578, 0.0143819, -0.0228797, 0.0216008
7: -0.0198323, 0.0112676, -0.0216183, 0.0116937, -0.0315260, 0.0328859
8: -0.0103719, 0.0157998, -0.0110805, 0.0174725, -0.0278443, 0.0268803
9: 0.9233138, 1.0229756, 0.9165484, 1.0235258, -0.1002120, 0.1064271

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
time: 1.55 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
time: 5.80 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0068255, 0.0039014, -0.0068364, 0.0028922, -0.0097178, 0.0107378
1: -0.0124919, 0.0187966, -0.0101376, 0.0201230, -0.0326150, 0.0289342
2: -0.0029244, 0.0245749, -0.0009517, 0.0249459, -0.0278703, 0.0255266
3: -0.0103170, 0.0050977, -0.0103133, 0.0064884, -0.0168054, 0.0154110
4: -0.0107890, 0.0111772, -0.0114524, 0.0100163, -0.0208053, 0.0226295
5: -0.0101817, 0.0166798, -0.0085547, 0.0177797, -0.0279614, 0.0252345
6: -0.0082751, 0.0070263, -0.0080066, 0.0078611, -0.0161362, 0.0150329
7: -0.0136281, 0.0106762, -0.0141469, 0.0098233, -0.0234514, 0.0248231
8: -0.0085291, 0.0088583, -0.0085372, 0.0102599, -0.0187890, 0.0173955
9: 0.9525756, 1.0245186, 0.9466540, 1.0207405, -0.0681649, 0.0778646

Time for backsubstitution: 2.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687034, upper bound: 0.0678018
time: 3.31 seconds

## Relational analysis of NS_A2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
time: 3.52 seconds

## BFS NS instance: NS_A2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068255, 0.0039014, -0.0074549, 0.0040956, -0.0109211, 0.0113563
1: -0.0124919, 0.0187966, -0.0112924, 0.0257222, -0.0382142, 0.0300890
2: -0.0029244, 0.0245749, -0.0017885, 0.0289898, -0.0319142, 0.0263634
3: -0.0103170, 0.0050977, -0.0116232, 0.0106874, -0.0210044, 0.0167209
4: -0.0107890, 0.0111772, -0.0158700, 0.0108226, -0.0216116, 0.0270472
5: -0.0101817, 0.0166798, -0.0101784, 0.0229328, -0.0331144, 0.0268582
6: -0.0082751, 0.0070263, -0.0084650, 0.0119290, -0.0202041, 0.0154913
7: -0.0136281, 0.0106762, -0.0179113, 0.0111584, -0.0247865, 0.0285875
8: -0.0085291, 0.0088583, -0.0103028, 0.0145022, -0.0230313, 0.0191611
9: 0.9525756, 1.0245186, 0.9277030, 1.0228342, -0.0702586, 0.0968156

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687034, upper bound: 0.0678018
time: 1.98 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
time: 1.52 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.66 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0668036, upper bound: 0.0702358
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0715980
NS_A1_B1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0678337, upper bound: 0.0726350
NS_A1_B1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0731205
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0670096
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0689735
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
NS_A1_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0695874, upper bound: 0.0673037
NS_A1_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
NS_A1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0686353, upper bound: 0.0690829
NS_A1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0707374
NS_A1_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0652147, upper bound: 0.0659473
NS_A1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
NS_A1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0652147, upper bound: 0.0682734
NS_A1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0662044, upper bound: 0.0695820
NS_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702395
NS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0646220, upper bound: 0.0684970
NS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702500
NS_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0662044, upper bound: 0.0709357
NS_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716066
NS_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
NS_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
NS_A1_B2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0659973, upper bound: 0.0669314
NS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
NS_A1_B2_B2_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
NS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
NS_A2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0726350, upper bound: 0.0678337
NS_A2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
NS_A2_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0689735, upper bound: 0.0664895
NS_A2_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
NS_A2_A1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0690829, upper bound: 0.0687197
NS_A2_A1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0707374, upper bound: 0.0702681
NS_A2_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0682734, upper bound: 0.0652203
NS_A2_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
NS_A2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0737349, upper bound: 0.0700827
NS_A2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0749766, upper bound: 0.0717021
NS_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0705570, upper bound: 0.0698083
NS_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
NS_A2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0694977, upper bound: 0.0684055
NS_A2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0709357, upper bound: 0.0662981
NS_A2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
NS_A2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
NS_A2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
NS_A2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0669314, upper bound: 0.0651608
NS_A2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0670587, upper bound: 0.0643809
NS_A2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
NS_A2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
NS_A2_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
NS_A2_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0701976
NS_A2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0687034, upper bound: 0.0678018
NS_A2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
NS_A2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0687034, upper bound: 0.0678018
NS_A2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.66
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0064488, 0.0022578, -0.0063011, 0.0021277, -0.0085765, 0.0085589
1: -0.0095097, 0.0176102, -0.0091262, 0.0161441, -0.0256538, 0.0267364
2: -0.0007257, 0.0225469, -0.0004123, 0.0216076, -0.0223333, 0.0229592
3: -0.0094767, 0.0049233, -0.0091729, 0.0037902, -0.0132670, 0.0140961
4: -0.0098849, 0.0096843, -0.0087954, 0.0094927, -0.0193776, 0.0184797
5: -0.0076221, 0.0159830, -0.0073419, 0.0146079, -0.0222300, 0.0233248
6: -0.0077355, 0.0064152, -0.0076388, 0.0053757, -0.0131111, 0.0140540
7: -0.0126403, 0.0091126, -0.0117742, 0.0087871, -0.0214274, 0.0208868
8: -0.0070755, 0.0087734, -0.0068167, 0.0075563, -0.0146318, 0.0155901
9: 0.9536883, 1.0197392, 0.9589826, 1.0191753, -0.0654870, 0.0607566

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0668036, upper bound: 0.0702358
time: 1.56 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0668036, upper bound: 0.0702358
time: 1.90 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0064755, 0.0023222, -0.0063338, 0.0022096, -0.0086850, 0.0086560
1: -0.0093863, 0.0178275, -0.0092167, 0.0163061, -0.0256924, 0.0270442
2: -0.0005869, 0.0227365, -0.0004740, 0.0217856, -0.0223725, 0.0232105
3: -0.0095225, 0.0051060, -0.0092465, 0.0038742, -0.0133967, 0.0143524
4: -0.0100722, 0.0096803, -0.0088995, 0.0095547, -0.0196269, 0.0185799
5: -0.0077537, 0.0161402, -0.0074797, 0.0147170, -0.0224706, 0.0236199
6: -0.0077621, 0.0065835, -0.0076750, 0.0054617, -0.0132238, 0.0142586
7: -0.0128652, 0.0091612, -0.0118952, 0.0089008, -0.0217660, 0.0210564
8: -0.0072495, 0.0088983, -0.0069712, 0.0076294, -0.0148789, 0.0158695
9: 0.9530996, 1.0196470, 0.9586262, 1.0193383, -0.0662386, 0.0610209

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0715980
time: 1.19 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0715980
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0064709, 0.0023031, -0.0067996, 0.0027983, -0.0092692, 0.0091027
1: -0.0093818, 0.0177670, -0.0102541, 0.0203536, -0.0297354, 0.0280211
2: -0.0005758, 0.0227090, -0.0011133, 0.0249592, -0.0255350, 0.0238223
3: -0.0095304, 0.0050553, -0.0102487, 0.0067791, -0.0163095, 0.0153040
4: -0.0100212, 0.0096674, -0.0117097, 0.0100318, -0.0200530, 0.0213771
5: -0.0077286, 0.0160872, -0.0084290, 0.0181274, -0.0258560, 0.0245163
6: -0.0077578, 0.0065375, -0.0079783, 0.0081134, -0.0158712, 0.0145157
7: -0.0128066, 0.0091445, -0.0142517, 0.0097797, -0.0225863, 0.0233962
8: -0.0072740, 0.0088559, -0.0083038, 0.0105636, -0.0178377, 0.0171596
9: 0.9532983, 1.0196233, 0.9452925, 1.0208427, -0.0675443, 0.0743308

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678337, upper bound: 0.0726350
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678337, upper bound: 0.0726350
time: 1.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0065052, 0.0023865, -0.0068378, 0.0028761, -0.0093813, 0.0092242
1: -0.0094744, 0.0179318, -0.0101117, 0.0207469, -0.0302213, 0.0280435
2: -0.0006384, 0.0228807, -0.0009622, 0.0252030, -0.0258414, 0.0238429
3: -0.0096048, 0.0051376, -0.0103113, 0.0070918, -0.0166967, 0.0154489
4: -0.0101232, 0.0097299, -0.0120218, 0.0100356, -0.0201587, 0.0217517
5: -0.0078681, 0.0161930, -0.0085694, 0.0184669, -0.0263349, 0.0247624
6: -0.0077947, 0.0066229, -0.0080107, 0.0084037, -0.0161984, 0.0146336
7: -0.0129268, 0.0092599, -0.0146232, 0.0098416, -0.0227684, 0.0238831
8: -0.0074283, 0.0089278, -0.0084568, 0.0108535, -0.0182818, 0.0173846
9: 0.9529452, 1.0197886, 0.9439958, 1.0207425, -0.0677973, 0.0757928

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0731205
time: 1.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685241, upper bound: 0.0731205
time: 1.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0062345, 0.0020555, -0.0062737, 0.0033670, -0.0096015, 0.0083292
1: -0.0091802, 0.0152972, -0.0114136, 0.0136106, -0.0227907, 0.0267108
2: -0.0005008, 0.0209957, -0.0021010, 0.0207943, -0.0212950, 0.0230966
3: -0.0090136, 0.0030956, -0.0091385, 0.0009955, -0.0100091, 0.0122341
4: -0.0080963, 0.0094393, -0.0064219, 0.0104024, -0.0184987, 0.0158612
5: -0.0070919, 0.0138535, -0.0086612, 0.0115995, -0.0186914, 0.0225146
6: -0.0075771, 0.0047348, -0.0078658, 0.0030495, -0.0106266, 0.0126006
7: -0.0111262, 0.0086327, -0.0099434, 0.0094149, -0.0205411, 0.0185761
8: -0.0064511, 0.0069367, -0.0069143, 0.0047044, -0.0111556, 0.0138510
9: 0.9618829, 1.0191309, 0.9713435, 1.0225322, -0.0606493, 0.0477874

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0689735
time: 3.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0664895, upper bound: 0.0689735
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0062533, 0.0020826, -0.0057827, 0.0029787, -0.0092320, 0.0078653
1: -0.0090321, 0.0153494, -0.0104523, 0.0094820, -0.0185141, 0.0258017
2: -0.0003345, 0.0211109, -0.0015156, 0.0173915, -0.0177260, 0.0226264
3: -0.0090477, 0.0031019, -0.0080893, -0.0017420, -0.0073057, 0.0111912
4: -0.0081243, 0.0094109, -0.0033169, 0.0098681, -0.0179924, 0.0127278
5: -0.0071809, 0.0138042, -0.0075353, 0.0078206, -0.0150014, 0.0213395
6: -0.0075891, 0.0047445, -0.0075397, 0.0001936, -0.0077827, 0.0122842
7: -0.0112032, 0.0086332, -0.0074313, 0.0084280, -0.0196312, 0.0160645
8: -0.0066281, 0.0068659, -0.0054756, 0.0015148, -0.0081428, 0.0123415
9: 0.9621280, 1.0189878, 0.9860274, 1.0209832, -0.0588552, 0.0329604

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
time: 1.53 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
time: 3.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0062533, 0.0020826, -0.0063028, 0.0034258, -0.0096791, 0.0083855
1: -0.0090321, 0.0153494, -0.0114875, 0.0137353, -0.0227674, 0.0268369
2: -0.0003345, 0.0211109, -0.0021539, 0.0209670, -0.0213015, 0.0232648
3: -0.0090477, 0.0031019, -0.0092069, 0.0010635, -0.0101112, 0.0123088
4: -0.0081243, 0.0094109, -0.0065176, 0.0104560, -0.0185802, 0.0159285
5: -0.0071809, 0.0138042, -0.0087792, 0.0116947, -0.0188755, 0.0225834
6: -0.0075891, 0.0047445, -0.0078959, 0.0031255, -0.0107146, 0.0126404
7: -0.0112032, 0.0086332, -0.0100550, 0.0095116, -0.0207148, 0.0186882
8: -0.0066281, 0.0068659, -0.0070610, 0.0047586, -0.0113867, 0.0139269
9: 0.9621280, 1.0189878, 0.9710485, 1.0226704, -0.0605423, 0.0479392

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 2.67 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0067903, 0.0027438, -0.0064709, 0.0023031, -0.0090935, 0.0092147
1: -0.0102714, 0.0210318, -0.0093818, 0.0177670, -0.0280384, 0.0304136
2: -0.0013526, 0.0249398, -0.0005758, 0.0227090, -0.0240617, 0.0255156
3: -0.0102214, 0.0075425, -0.0095304, 0.0050553, -0.0152767, 0.0170729
4: -0.0127231, 0.0102489, -0.0100212, 0.0096674, -0.0223905, 0.0202701
5: -0.0086851, 0.0192475, -0.0077286, 0.0160872, -0.0247724, 0.0269762
6: -0.0080102, 0.0089958, -0.0077578, 0.0065375, -0.0145477, 0.0167536
7: -0.0150697, 0.0099443, -0.0128066, 0.0091445, -0.0242142, 0.0227509
8: -0.0080966, 0.0113904, -0.0072740, 0.0088559, -0.0169525, 0.0186645
9: 0.9418369, 1.0211862, 0.9532983, 1.0196233, -0.0777864, 0.0678879

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0695874, upper bound: 0.0673037
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0695874, upper bound: 0.0673037
time: 2.65 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0068366, 0.0028538, -0.0065052, 0.0023865, -0.0092231, 0.0093590
1: -0.0101813, 0.0215317, -0.0094744, 0.0179318, -0.0281131, 0.0310061
2: -0.0012400, 0.0252488, -0.0006384, 0.0228807, -0.0241207, 0.0258872
3: -0.0103014, 0.0079773, -0.0096048, 0.0051376, -0.0154390, 0.0175821
4: -0.0131525, 0.0102890, -0.0101232, 0.0097299, -0.0228824, 0.0204122
5: -0.0088883, 0.0197033, -0.0078681, 0.0161930, -0.0250813, 0.0275714
6: -0.0080610, 0.0094121, -0.0077947, 0.0066229, -0.0146839, 0.0172068
7: -0.0155122, 0.0100705, -0.0129268, 0.0092599, -0.0247721, 0.0229973
8: -0.0082754, 0.0117967, -0.0074283, 0.0089278, -0.0172032, 0.0192250
9: 0.9402030, 1.0211754, 0.9529452, 1.0197886, -0.0795856, 0.0682302

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
time: 1.77 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -0.0068329, 0.0028433, -0.0070399, 0.0033665, -0.0101994, 0.0098832
1: -0.0101716, 0.0214753, -0.0106199, 0.0224583, -0.0326298, 0.0320952
2: -0.0012324, 0.0252231, -0.0013009, 0.0267418, -0.0279742, 0.0265240
3: -0.0103063, 0.0079315, -0.0107763, 0.0082473, -0.0185536, 0.0187078
4: -0.0131085, 0.0102797, -0.0131119, 0.0102698, -0.0233783, 0.0233916
5: -0.0088663, 0.0196574, -0.0089389, 0.0198854, -0.0287517, 0.0285963
6: -0.0080567, 0.0093711, -0.0081387, 0.0095402, -0.0175970, 0.0175098
7: -0.0154599, 0.0100561, -0.0157845, 0.0102328, -0.0256927, 0.0258407
8: -0.0083040, 0.0117585, -0.0090491, 0.0121833, -0.0204872, 0.0208076
9: 0.9403818, 1.0211542, 0.9384461, 1.0214263, -0.0810446, 0.0827081

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687197, upper bound: 0.0690829
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687197, upper bound: 0.0690829
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0068695, 0.0029358, -0.0070566, 0.0033911, -0.0102607, 0.0099924
1: -0.0102681, 0.0216322, -0.0104487, 0.0227094, -0.0329775, 0.0320809
2: -0.0012933, 0.0253938, -0.0011340, 0.0268350, -0.0281283, 0.0265279
3: -0.0103797, 0.0080068, -0.0107872, 0.0084555, -0.0188352, 0.0187941
4: -0.0132021, 0.0103407, -0.0133180, 0.0102528, -0.0234549, 0.0236587
5: -0.0090050, 0.0197539, -0.0090290, 0.0200883, -0.0290933, 0.0287830
6: -0.0080938, 0.0094505, -0.0081576, 0.0097178, -0.0178116, 0.0176080
7: -0.0155724, 0.0101703, -0.0160244, 0.0102564, -0.0258288, 0.0261946
8: -0.0084470, 0.0118255, -0.0091215, 0.0123526, -0.0207996, 0.0209470
9: 0.9400532, 1.0213192, 0.9376540, 1.0212716, -0.0812184, 0.0836651

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702680, upper bound: 0.0707374
time: 2.22 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702680, upper bound: 0.0707374
time: 1.53 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -0.0063016, 0.0034514, -0.0062533, 0.0020826, -0.0083843, 0.0097047
1: -0.0114549, 0.0142223, -0.0090321, 0.0153494, -0.0268042, 0.0232544
2: -0.0022969, 0.0209819, -0.0003345, 0.0211109, -0.0234078, 0.0213164
3: -0.0091621, 0.0018110, -0.0090477, 0.0031019, -0.0122641, 0.0108587
4: -0.0075178, 0.0106052, -0.0081243, 0.0094109, -0.0169287, 0.0187295
5: -0.0089495, 0.0127693, -0.0071809, 0.0138042, -0.0227537, 0.0199502
6: -0.0079132, 0.0039928, -0.0075891, 0.0047445, -0.0126577, 0.0115819
7: -0.0108393, 0.0096099, -0.0112032, 0.0086332, -0.0194725, 0.0208131
8: -0.0068318, 0.0055387, -0.0066281, 0.0068659, -0.0136978, 0.0121667
9: 0.9677775, 1.0228754, 0.9621280, 1.0189878, -0.0512103, 0.0607474

Time for backsubstitution: 2.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A2_B1_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 1.66 seconds

## Relational analysis of NS_A1_B1_A2_A2_B1_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -0.0062703, 0.0033860, -0.0067763, 0.0027464, -0.0090166, 0.0101622
1: -0.0113771, 0.0140831, -0.0102104, 0.0199192, -0.0312963, 0.0242934
2: -0.0022430, 0.0208014, -0.0010733, 0.0246741, -0.0269171, 0.0218748
3: -0.0090898, 0.0017325, -0.0101978, 0.0064129, -0.0155026, 0.0119303
4: -0.0074147, 0.0105502, -0.0113345, 0.0099883, -0.0174029, 0.0218847
5: -0.0088267, 0.0126651, -0.0083445, 0.0177090, -0.0265357, 0.0210097
6: -0.0078815, 0.0039089, -0.0079541, 0.0077635, -0.0156449, 0.0118630
7: -0.0107232, 0.0095089, -0.0139215, 0.0097040, -0.0204271, 0.0234305
8: -0.0066794, 0.0054749, -0.0082203, 0.0102341, -0.0169135, 0.0136951
9: 0.9681052, 1.0227323, 0.9468869, 1.0207468, -0.0526416, 0.0758454

Time for backsubstitution: 2.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0652203, upper bound: 0.0682734
time: 2.02 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B1_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0652203, upper bound: 0.0682734
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -0.0063016, 0.0034514, -0.0068019, 0.0027998, -0.0091014, 0.0102533
1: -0.0114549, 0.0142223, -0.0100428, 0.0200112, -0.0314661, 0.0242651
2: -0.0022969, 0.0209819, -0.0008926, 0.0247847, -0.0270816, 0.0218745
3: -0.0091621, 0.0018110, -0.0102295, 0.0064551, -0.0156172, 0.0120405
4: -0.0075178, 0.0106052, -0.0113955, 0.0099595, -0.0174773, 0.0220007
5: -0.0089495, 0.0127693, -0.0084269, 0.0177217, -0.0266712, 0.0211962
6: -0.0079132, 0.0039928, -0.0079710, 0.0078173, -0.0157305, 0.0119638
7: -0.0108393, 0.0096099, -0.0140771, 0.0097146, -0.0205539, 0.0236870
8: -0.0068318, 0.0055387, -0.0083512, 0.0102281, -0.0170599, 0.0138899
9: 0.9677775, 1.0228754, 0.9468251, 1.0205828, -0.0528053, 0.0760503

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.67 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.71 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0064709, 0.0023031, -0.0067903, 0.0027438, -0.0092147, 0.0090935
1: -0.0093818, 0.0177670, -0.0102714, 0.0210318, -0.0304136, 0.0280384
2: -0.0005758, 0.0227090, -0.0013526, 0.0249398, -0.0255156, 0.0240617
3: -0.0095304, 0.0050553, -0.0102214, 0.0075425, -0.0170729, 0.0152767
4: -0.0100212, 0.0096674, -0.0127231, 0.0102489, -0.0202701, 0.0223905
5: -0.0077286, 0.0160872, -0.0086851, 0.0192475, -0.0269762, 0.0247724
6: -0.0077578, 0.0065375, -0.0080102, 0.0089958, -0.0167536, 0.0145477
7: -0.0128066, 0.0091445, -0.0150697, 0.0099443, -0.0227509, 0.0242142
8: -0.0072740, 0.0088559, -0.0080966, 0.0113904, -0.0186645, 0.0169525
9: 0.9532983, 1.0196233, 0.9418369, 1.0211862, -0.0678879, 0.0777864

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673037, upper bound: 0.0695874
time: 2.20 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673037, upper bound: 0.0695874
time: 1.49 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0065052, 0.0023865, -0.0068366, 0.0028538, -0.0093590, 0.0092231
1: -0.0094744, 0.0179318, -0.0101813, 0.0215317, -0.0310061, 0.0281131
2: -0.0006384, 0.0228807, -0.0012400, 0.0252488, -0.0258872, 0.0241207
3: -0.0096048, 0.0051376, -0.0103014, 0.0079773, -0.0175821, 0.0154390
4: -0.0101232, 0.0097299, -0.0131525, 0.0102890, -0.0204122, 0.0228824
5: -0.0078681, 0.0161930, -0.0088883, 0.0197033, -0.0275714, 0.0250813
6: -0.0077947, 0.0066229, -0.0080610, 0.0094121, -0.0172068, 0.0146839
7: -0.0129268, 0.0092599, -0.0155122, 0.0100705, -0.0229973, 0.0247721
8: -0.0074283, 0.0089278, -0.0082754, 0.0117967, -0.0192250, 0.0172032
9: 0.9529452, 1.0197886, 0.9402030, 1.0211754, -0.0682302, 0.0795856

Time for backsubstitution: 2.23 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682461, upper bound: 0.0702529
time: 1.71 seconds

## Relational analysis of NS_A1_B2_B1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682461, upper bound: 0.0702529
time: 1.73 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -0.0069740, 0.0030637, -0.0068329, 0.0028433, -0.0098173, 0.0098966
1: -0.0105473, 0.0229043, -0.0101716, 0.0214753, -0.0320226, 0.0330759
2: -0.0015239, 0.0262203, -0.0012324, 0.0252231, -0.0267470, 0.0274527
3: -0.0106157, 0.0089642, -0.0103063, 0.0079315, -0.0185472, 0.0192705
4: -0.0140905, 0.0104368, -0.0131085, 0.0102797, -0.0243702, 0.0235453
5: -0.0091148, 0.0208930, -0.0088663, 0.0196574, -0.0287722, 0.0297593
6: -0.0081384, 0.0102914, -0.0080567, 0.0093711, -0.0175096, 0.0183481
7: -0.0162454, 0.0103219, -0.0154599, 0.0100561, -0.0263015, 0.0257818
8: -0.0086191, 0.0128412, -0.0083040, 0.0117585, -0.0203776, 0.0211452
9: 0.9354767, 1.0216691, 0.9403818, 1.0211542, -0.0856774, 0.0812874

Time for backsubstitution: 2.22 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0646220, upper bound: 0.0684970
time: 1.96 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0646220, upper bound: 0.0684970
time: 1.90 seconds

## BFS NS instance: NS_A1_B2_B1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -0.0070181, 0.0031731, -0.0068695, 0.0029358, -0.0099539, 0.0100426
1: -0.0104641, 0.0233527, -0.0102681, 0.0216322, -0.0320963, 0.0336208
2: -0.0014039, 0.0265015, -0.0012933, 0.0253938, -0.0267978, 0.0277948
3: -0.0106905, 0.0093361, -0.0103797, 0.0080068, -0.0186973, 0.0197159
4: -0.0144643, 0.0104679, -0.0132021, 0.0103407, -0.0248050, 0.0236700
5: -0.0093039, 0.0212749, -0.0090050, 0.0197539, -0.0290579, 0.0302799
6: -0.0081892, 0.0106544, -0.0080938, 0.0094505, -0.0176396, 0.0187482
7: -0.0166370, 0.0104440, -0.0155724, 0.0101703, -0.0268072, 0.0260164
8: -0.0087933, 0.0131806, -0.0084470, 0.0118255, -0.0206187, 0.0216276
9: 0.9341322, 1.0216491, 0.9400532, 1.0213192, -0.0871869, 0.0815960

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702500
time: 3.40 seconds

## Relational analysis of NS_A1_B2_B1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0702500
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0064709, 0.0023031, -0.0073962, 0.0045279, -0.0109988, 0.0096993
1: -0.0093818, 0.0177670, -0.0113802, 0.0263547, -0.0357365, 0.0291472
2: -0.0005758, 0.0227090, -0.0019338, 0.0298529, -0.0304287, 0.0246429
3: -0.0095304, 0.0050553, -0.0115446, 0.0109021, -0.0204325, 0.0165999
4: -0.0100212, 0.0096674, -0.0159508, 0.0109433, -0.0209645, 0.0256183
5: -0.0077286, 0.0160872, -0.0100036, 0.0233543, -0.0310829, 0.0260908
6: -0.0077578, 0.0065375, -0.0084161, 0.0124242, -0.0201820, 0.0149535
7: -0.0128066, 0.0091445, -0.0191740, 0.0110512, -0.0238578, 0.0283185
8: -0.0072740, 0.0088559, -0.0100900, 0.0152672, -0.0225412, 0.0189459
9: 0.9532983, 1.0196233, 0.9255337, 1.0228748, -0.0695765, 0.0940896

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673037, upper bound: 0.0709358
time: 1.55 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673037, upper bound: 0.0709358
time: 1.31 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0065052, 0.0023865, -0.0074490, 0.0046364, -0.0111416, 0.0098355
1: -0.0094744, 0.0179318, -0.0112643, 0.0269737, -0.0364481, 0.0291961
2: -0.0006384, 0.0228807, -0.0017914, 0.0301773, -0.0308157, 0.0246721
3: -0.0096048, 0.0051376, -0.0116141, 0.0113938, -0.0209986, 0.0167517
4: -0.0101232, 0.0097299, -0.0164241, 0.0109842, -0.0211073, 0.0261540
5: -0.0078681, 0.0161930, -0.0101840, 0.0238712, -0.0317393, 0.0263770
6: -0.0077947, 0.0066229, -0.0084656, 0.0128776, -0.0206722, 0.0150885
7: -0.0129268, 0.0092599, -0.0196819, 0.0111651, -0.0240919, 0.0289418
8: -0.0074283, 0.0089278, -0.0102277, 0.0157352, -0.0231635, 0.0191555
9: 0.9529452, 1.0197886, 0.9235650, 1.0228243, -0.0698791, 0.0962236

Time for backsubstitution: 2.20 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682461, upper bound: 0.0716066
time: 1.48 seconds

## Relational analysis of NS_A1_B2_B1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682461, upper bound: 0.0716066
time: 2.05 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2_A1

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0074868, 0.0047480, -0.0116222, 0.0104344
1: -0.0102804, 0.0216467, -0.0113465, 0.0271213, -0.0374017, 0.0329932
2: -0.0013008, 0.0254146, -0.0018529, 0.0303596, -0.0316605, 0.0272675
3: -0.0103910, 0.0080113, -0.0116776, 0.0114324, -0.0218234, 0.0196889
4: -0.0132094, 0.0103480, -0.0164778, 0.0110528, -0.0242621, 0.0268258
5: -0.0090217, 0.0197614, -0.0103079, 0.0239427, -0.0329644, 0.0300693
6: -0.0080985, 0.0094561, -0.0084978, 0.0129430, -0.0210415, 0.0179539
7: -0.0155814, 0.0101845, -0.0198323, 0.0112676, -0.0268490, 0.0300169
8: -0.0084719, 0.0118298, -0.0103719, 0.0157998, -0.0242717, 0.0222017
9: 0.9400309, 1.0213395, 0.9233138, 1.0229756, -0.0829447, 0.0980257

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0662044, upper bound: 0.0709633
time: 1.21 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
time: 2.08 seconds

## BFS NS instance: NS_A1_B2_B1_B2_A2_A2

### Backsubstitution after applying NS history:
0: -0.0063058, 0.0034586, -0.0074868, 0.0047480, -0.0110538, 0.0109454
1: -0.0114675, 0.0142369, -0.0113465, 0.0271213, -0.0385888, 0.0255834
2: -0.0023040, 0.0210116, -0.0018529, 0.0303596, -0.0326637, 0.0228645
3: -0.0091753, 0.0018162, -0.0116776, 0.0114324, -0.0206077, 0.0134938
4: -0.0075261, 0.0106122, -0.0164778, 0.0110528, -0.0185788, 0.0270901
5: -0.0089659, 0.0127776, -0.0103079, 0.0239427, -0.0329086, 0.0230855
6: -0.0079179, 0.0039991, -0.0084978, 0.0129430, -0.0208609, 0.0124969
7: -0.0108489, 0.0096242, -0.0198323, 0.0112676, -0.0221164, 0.0294566
8: -0.0068596, 0.0055431, -0.0103719, 0.0157998, -0.0226594, 0.0159150
9: 0.9677531, 1.0228958, 0.9233138, 1.0229756, -0.0552225, 0.0995820

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B1_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0662044, upper bound: 0.0709633
time: 1.51 seconds

## Relational analysis of NS_A1_B2_B1_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669953, upper bound: 0.0716527
time: 2.00 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -0.0062811, 0.0021396, -0.0067906, 0.0038337, -0.0101148, 0.0089302
1: -0.0091204, 0.0154521, -0.0123938, 0.0186790, -0.0277994, 0.0278459
2: -0.0003836, 0.0212821, -0.0028614, 0.0243917, -0.0247752, 0.0241436
3: -0.0091328, 0.0031338, -0.0102232, 0.0050613, -0.0141941, 0.0133570
4: -0.0081748, 0.0094581, -0.0107239, 0.0111162, -0.0192910, 0.0201820
5: -0.0072922, 0.0138557, -0.0100460, 0.0166181, -0.0239103, 0.0239017
6: -0.0076213, 0.0047831, -0.0082376, 0.0069775, -0.0145987, 0.0130207
7: -0.0112645, 0.0087303, -0.0135501, 0.0105596, -0.0218241, 0.0222803
8: -0.0068137, 0.0068952, -0.0083262, 0.0088266, -0.0156403, 0.0152214
9: 0.9619726, 1.0191264, 0.9527490, 1.0243526, -0.0623800, 0.0663775

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659473, upper bound: 0.0671227
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0659473, upper bound: 0.0690997
time: 2.09 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0067967, 0.0027754, -0.0068212, 0.0038926, -0.0106894, 0.0095966
1: -0.0101060, 0.0207606, -0.0124797, 0.0187820, -0.0288880, 0.0332403
2: -0.0011671, 0.0248293, -0.0029166, 0.0245513, -0.0257184, 0.0277460
3: -0.0102120, 0.0073227, -0.0103052, 0.0050931, -0.0153051, 0.0176279
4: -0.0125204, 0.0102099, -0.0107809, 0.0111696, -0.0236901, 0.0209908
5: -0.0087339, 0.0189557, -0.0101648, 0.0166720, -0.0254059, 0.0291205
6: -0.0080189, 0.0088134, -0.0082704, 0.0070202, -0.0150391, 0.0170838
7: -0.0149673, 0.0099363, -0.0136183, 0.0106617, -0.0256290, 0.0235546
8: -0.0081553, 0.0111491, -0.0085036, 0.0088543, -0.0170095, 0.0196528
9: 0.9430635, 1.0210063, 0.9525976, 1.0244980, -0.0814345, 0.0684087

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.59 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.53 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -0.0067996, 0.0027983, -0.0064709, 0.0023031, -0.0091027, 0.0092692
1: -0.0102541, 0.0203536, -0.0093818, 0.0177670, -0.0280211, 0.0297354
2: -0.0011133, 0.0249592, -0.0005758, 0.0227090, -0.0238223, 0.0255350
3: -0.0102487, 0.0067791, -0.0095304, 0.0050553, -0.0153040, 0.0163095
4: -0.0117097, 0.0100318, -0.0100212, 0.0096674, -0.0213771, 0.0200530
5: -0.0084290, 0.0181274, -0.0077286, 0.0160872, -0.0245163, 0.0258560
6: -0.0079783, 0.0081134, -0.0077578, 0.0065375, -0.0145157, 0.0158712
7: -0.0142517, 0.0097797, -0.0128066, 0.0091445, -0.0233962, 0.0225863
8: -0.0083038, 0.0105636, -0.0072740, 0.0088559, -0.0171596, 0.0178377
9: 0.9452925, 1.0208427, 0.9532983, 1.0196233, -0.0743308, 0.0675443

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726350, upper bound: 0.0678337
time: 1.60 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A1_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726350, upper bound: 0.0678337
time: 2.37 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -0.0068378, 0.0028761, -0.0065052, 0.0023865, -0.0092242, 0.0093813
1: -0.0101117, 0.0207469, -0.0094744, 0.0179318, -0.0280435, 0.0302213
2: -0.0009622, 0.0252030, -0.0006384, 0.0228807, -0.0238429, 0.0258414
3: -0.0103113, 0.0070918, -0.0096048, 0.0051376, -0.0154489, 0.0166967
4: -0.0120218, 0.0100356, -0.0101232, 0.0097299, -0.0217517, 0.0201587
5: -0.0085694, 0.0184669, -0.0078681, 0.0161930, -0.0247624, 0.0263349
6: -0.0080107, 0.0084037, -0.0077947, 0.0066229, -0.0146336, 0.0161984
7: -0.0146232, 0.0098416, -0.0129268, 0.0092599, -0.0238831, 0.0227684
8: -0.0084568, 0.0108535, -0.0074283, 0.0089278, -0.0173846, 0.0182818
9: 0.9439958, 1.0207425, 0.9529452, 1.0197886, -0.0757928, 0.0677973

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
time: 1.84 seconds

## Relational analysis of NS_A2_A1_B1_B1_A1_A2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
time: 1.29 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0062737, 0.0033670, -0.0062345, 0.0020555, -0.0083292, 0.0096015
1: -0.0114136, 0.0136106, -0.0091802, 0.0152972, -0.0267108, 0.0227907
2: -0.0021010, 0.0207943, -0.0005008, 0.0209957, -0.0230966, 0.0212950
3: -0.0091385, 0.0009955, -0.0090136, 0.0030956, -0.0122341, 0.0100091
4: -0.0064219, 0.0104024, -0.0080963, 0.0094393, -0.0158612, 0.0184987
5: -0.0086612, 0.0115995, -0.0070919, 0.0138535, -0.0225146, 0.0186914
6: -0.0078658, 0.0030495, -0.0075771, 0.0047348, -0.0126006, 0.0106266
7: -0.0099434, 0.0094149, -0.0111262, 0.0086327, -0.0185761, 0.0205411
8: -0.0069143, 0.0047044, -0.0064511, 0.0069367, -0.0138510, 0.0111556
9: 0.9713435, 1.0225322, 0.9618829, 1.0191309, -0.0477874, 0.0606493

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0689735, upper bound: 0.0664895
time: 1.30 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0689735, upper bound: 0.0664895
time: 1.66 seconds

## BFS NS instance: NS_A2_A1_B1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0063028, 0.0034258, -0.0062533, 0.0020826, -0.0083855, 0.0096791
1: -0.0114875, 0.0137353, -0.0090321, 0.0153494, -0.0268369, 0.0227674
2: -0.0021539, 0.0209670, -0.0003345, 0.0211109, -0.0232648, 0.0213015
3: -0.0092069, 0.0010635, -0.0090477, 0.0031019, -0.0123088, 0.0101112
4: -0.0065176, 0.0104560, -0.0081243, 0.0094109, -0.0159285, 0.0185802
5: -0.0087792, 0.0116947, -0.0071809, 0.0138042, -0.0225834, 0.0188755
6: -0.0078959, 0.0031255, -0.0075891, 0.0047445, -0.0126404, 0.0107146
7: -0.0100550, 0.0095116, -0.0112032, 0.0086332, -0.0186882, 0.0207148
8: -0.0070610, 0.0047586, -0.0066281, 0.0068659, -0.0139269, 0.0113867
9: 0.9710485, 1.0226704, 0.9621280, 1.0189878, -0.0479392, 0.0605423

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
time: 1.65 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
time: 1.90 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0070399, 0.0033665, -0.0068329, 0.0028433, -0.0098832, 0.0101994
1: -0.0106199, 0.0224583, -0.0101716, 0.0214753, -0.0320952, 0.0326298
2: -0.0013009, 0.0267418, -0.0012324, 0.0252231, -0.0265240, 0.0279742
3: -0.0107763, 0.0082473, -0.0103063, 0.0079315, -0.0187078, 0.0185536
4: -0.0131119, 0.0102698, -0.0131085, 0.0102797, -0.0233916, 0.0233783
5: -0.0089389, 0.0198854, -0.0088663, 0.0196574, -0.0285963, 0.0287517
6: -0.0081387, 0.0095402, -0.0080567, 0.0093711, -0.0175098, 0.0175970
7: -0.0157845, 0.0102328, -0.0154599, 0.0100561, -0.0258407, 0.0256927
8: -0.0090491, 0.0121833, -0.0083040, 0.0117585, -0.0208076, 0.0204872
9: 0.9384461, 1.0214263, 0.9403818, 1.0211542, -0.0827081, 0.0810446

Time for backsubstitution: 2.19 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B2_B1_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690829, upper bound: 0.0687197
time: 1.68 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690829, upper bound: 0.0687197
time: 1.78 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0070566, 0.0033911, -0.0068695, 0.0029358, -0.0099924, 0.0102607
1: -0.0104487, 0.0227094, -0.0102681, 0.0216322, -0.0320809, 0.0329775
2: -0.0011340, 0.0268350, -0.0012933, 0.0253938, -0.0265279, 0.0281283
3: -0.0107872, 0.0084555, -0.0103797, 0.0080068, -0.0187941, 0.0188352
4: -0.0133180, 0.0102528, -0.0132021, 0.0103407, -0.0236587, 0.0234549
5: -0.0090290, 0.0200883, -0.0090050, 0.0197539, -0.0287830, 0.0290933
6: -0.0081576, 0.0097178, -0.0080938, 0.0094505, -0.0176080, 0.0178116
7: -0.0160244, 0.0102564, -0.0155724, 0.0101703, -0.0261946, 0.0258288
8: -0.0091215, 0.0123526, -0.0084470, 0.0118255, -0.0209470, 0.0207996
9: 0.9376540, 1.0212716, 0.9400532, 1.0213192, -0.0836651, 0.0812184

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B2_B1_A2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707374, upper bound: 0.0702681
time: 1.15 seconds

## Relational analysis of NS_A2_A1_B1_B2_B1_A2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0707374, upper bound: 0.0702680
time: 2.18 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0068027, 0.0027954, -0.0062703, 0.0033860, -0.0101887, 0.0090656
1: -0.0102529, 0.0199642, -0.0113771, 0.0140831, -0.0243359, 0.0313413
2: -0.0010808, 0.0247977, -0.0022430, 0.0208014, -0.0218822, 0.0270407
3: -0.0102580, 0.0064167, -0.0090898, 0.0017325, -0.0119905, 0.0155064
4: -0.0113449, 0.0099968, -0.0074147, 0.0105502, -0.0218951, 0.0174114
5: -0.0083817, 0.0177200, -0.0088267, 0.0126651, -0.0210469, 0.0265467
6: -0.0079685, 0.0077707, -0.0078815, 0.0039089, -0.0118773, 0.0156522
7: -0.0139359, 0.0097390, -0.0107232, 0.0095089, -0.0234448, 0.0204621
8: -0.0083175, 0.0102355, -0.0066794, 0.0054749, -0.0137923, 0.0169149
9: 0.9468634, 1.0207930, 0.9681052, 1.0227323, -0.0758689, 0.0526878

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B2_B2_A1_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682734, upper bound: 0.0652203
time: 1.54 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_A1_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682734, upper bound: 0.0652203
time: 1.68 seconds

## BFS NS instance: NS_A2_A1_B1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0068033, 0.0028023, -0.0063016, 0.0034514, -0.0102547, 0.0091039
1: -0.0100450, 0.0200137, -0.0114549, 0.0142223, -0.0242673, 0.0314686
2: -0.0008929, 0.0247913, -0.0022969, 0.0209819, -0.0218748, 0.0270882
3: -0.0102327, 0.0064552, -0.0091621, 0.0018110, -0.0120437, 0.0156174
4: -0.0113958, 0.0099598, -0.0075178, 0.0106052, -0.0220011, 0.0174777
5: -0.0084287, 0.0177222, -0.0089495, 0.0127693, -0.0211980, 0.0266717
6: -0.0079717, 0.0078176, -0.0079132, 0.0039928, -0.0119645, 0.0157308
7: -0.0140777, 0.0097164, -0.0108393, 0.0096099, -0.0236876, 0.0205557
8: -0.0083563, 0.0102282, -0.0068318, 0.0055387, -0.0138950, 0.0170600
9: 0.9468241, 1.0205849, 0.9677775, 1.0228754, -0.0760513, 0.0528075

Time for backsubstitution: 2.21 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B1_B2_B2_A2_A1

### Relational analysis result of NS_A2_A1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 1.76 seconds

## Relational analysis of NS_A2_A1_B1_B2_B2_A2_A2

### Relational analysis result of NS_A2_A1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 2.12 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068336, 0.0028693, -0.0070399, 0.0033665, -0.0102001, 0.0099092
1: -0.0101026, 0.0206958, -0.0106199, 0.0224583, -0.0325608, 0.0313157
2: -0.0009532, 0.0251797, -0.0013009, 0.0267418, -0.0276950, 0.0264805
3: -0.0103150, 0.0070528, -0.0107763, 0.0082473, -0.0185623, 0.0178291
4: -0.0119821, 0.0100249, -0.0131119, 0.0102698, -0.0222519, 0.0231368
5: -0.0085458, 0.0184259, -0.0089389, 0.0198854, -0.0284312, 0.0273648
6: -0.0080061, 0.0083674, -0.0081387, 0.0095402, -0.0175463, 0.0165061
7: -0.0145758, 0.0098248, -0.0157845, 0.0102328, -0.0248086, 0.0256094
8: -0.0084825, 0.0108206, -0.0090491, 0.0121833, -0.0206658, 0.0198697
9: 0.9441505, 1.0207195, 0.9384461, 1.0214263, -0.0772758, 0.0822734

Time for backsubstitution: 2.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0737349, upper bound: 0.0700827
time: 1.23 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0737349, upper bound: 0.0700827
time: 1.58 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068674, 0.0029572, -0.0070566, 0.0033911, -0.0102585, 0.0100138
1: -0.0101950, 0.0208434, -0.0104487, 0.0227094, -0.0329044, 0.0312921
2: -0.0010146, 0.0253397, -0.0011340, 0.0268350, -0.0278497, 0.0264738
3: -0.0103837, 0.0071208, -0.0107872, 0.0084555, -0.0188392, 0.0179080
4: -0.0120717, 0.0100859, -0.0133180, 0.0102528, -0.0223245, 0.0234040
5: -0.0086821, 0.0185174, -0.0090290, 0.0200883, -0.0287704, 0.0275465
6: -0.0080420, 0.0084419, -0.0081576, 0.0097178, -0.0177598, 0.0165994
7: -0.0146836, 0.0099374, -0.0160244, 0.0102564, -0.0249400, 0.0259618
8: -0.0086179, 0.0108813, -0.0091215, 0.0123526, -0.0209704, 0.0200028
9: 0.9438486, 1.0208819, 0.9376540, 1.0212716, -0.0774230, 0.0832279

Time for backsubstitution: 2.01 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749766, upper bound: 0.0717021
time: 1.97 seconds

## Relational analysis of NS_A2_A1_B2_B1_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749766, upper bound: 0.0717021
time: 1.77 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0062737, 0.0033670, -0.0068027, 0.0027954, -0.0090691, 0.0101697
1: -0.0114136, 0.0136106, -0.0102529, 0.0199642, -0.0313778, 0.0238634
2: -0.0021010, 0.0207943, -0.0010808, 0.0247977, -0.0268987, 0.0218750
3: -0.0091385, 0.0009955, -0.0102580, 0.0064167, -0.0155552, 0.0112535
4: -0.0064219, 0.0104024, -0.0113449, 0.0099968, -0.0164186, 0.0217472
5: -0.0086612, 0.0115995, -0.0083817, 0.0177200, -0.0263812, 0.0199812
6: -0.0078658, 0.0030495, -0.0079685, 0.0077707, -0.0156365, 0.0110180
7: -0.0099434, 0.0094149, -0.0139359, 0.0097390, -0.0196824, 0.0233508
8: -0.0069143, 0.0047044, -0.0083175, 0.0102355, -0.0171499, 0.0130219
9: 0.9713435, 1.0225322, 0.9468634, 1.0207930, -0.0494494, 0.0756689

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705570, upper bound: 0.0698083
time: 1.40 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B1_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705570, upper bound: 0.0698083
time: 1.03 seconds

## BFS NS instance: NS_A2_A1_B2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0063028, 0.0034258, -0.0068033, 0.0028023, -0.0091052, 0.0102291
1: -0.0114875, 0.0137353, -0.0100450, 0.0200137, -0.0315012, 0.0237803
2: -0.0021539, 0.0209670, -0.0008929, 0.0247913, -0.0269453, 0.0218599
3: -0.0092069, 0.0010635, -0.0102327, 0.0064552, -0.0156621, 0.0112962
4: -0.0065176, 0.0104560, -0.0113958, 0.0099598, -0.0164775, 0.0218518
5: -0.0087792, 0.0116947, -0.0084287, 0.0177222, -0.0265014, 0.0201234
6: -0.0078959, 0.0031255, -0.0079717, 0.0078176, -0.0157135, 0.0110972
7: -0.0100550, 0.0095116, -0.0140777, 0.0097164, -0.0197714, 0.0235893
8: -0.0070610, 0.0047586, -0.0083563, 0.0102282, -0.0172892, 0.0131149
9: 0.9710485, 1.0226704, 0.9468241, 1.0205849, -0.0495364, 0.0758463

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
time: 1.78 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2_B2_B2

### Relational analysis result of NS_A2_A1_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
time: 1.43 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0074868, 0.0047480, -0.0116195, 0.0104557
1: -0.0102067, 0.0208570, -0.0113465, 0.0271213, -0.0373280, 0.0322035
2: -0.0010219, 0.0253591, -0.0018529, 0.0303596, -0.0313816, 0.0272121
3: -0.0103941, 0.0071250, -0.0116776, 0.0114324, -0.0218265, 0.0188026
4: -0.0120788, 0.0100930, -0.0164778, 0.0110528, -0.0231316, 0.0265708
5: -0.0086981, 0.0185246, -0.0103079, 0.0239427, -0.0326407, 0.0288325
6: -0.0080465, 0.0084473, -0.0084978, 0.0129430, -0.0209894, 0.0169450
7: -0.0146922, 0.0099509, -0.0198323, 0.0112676, -0.0259598, 0.0297833
8: -0.0086411, 0.0108853, -0.0103719, 0.0157998, -0.0244409, 0.0212571
9: 0.9438275, 1.0209014, 0.9233138, 1.0229756, -0.0791481, 0.0975876

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706765, upper bound: 0.0725256
time: 1.61 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A1_B2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717283, upper bound: 0.0730510
time: 1.55 seconds

## BFS NS instance: NS_A2_A1_B2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0063068, 0.0034323, -0.0074868, 0.0047480, -0.0110547, 0.0109191
1: -0.0114994, 0.0137490, -0.0113465, 0.0271213, -0.0386207, 0.0250955
2: -0.0021608, 0.0209964, -0.0018529, 0.0303596, -0.0325205, 0.0228493
3: -0.0092192, 0.0010684, -0.0116776, 0.0114324, -0.0206516, 0.0127460
4: -0.0065257, 0.0104626, -0.0164778, 0.0110528, -0.0175785, 0.0269404
5: -0.0087947, 0.0117027, -0.0103079, 0.0239427, -0.0327374, 0.0220106
6: -0.0079002, 0.0031316, -0.0084978, 0.0129430, -0.0208432, 0.0116293
7: -0.0100647, 0.0095251, -0.0198323, 0.0112676, -0.0213322, 0.0293575
8: -0.0070877, 0.0047627, -0.0103719, 0.0157998, -0.0228874, 0.0151345
9: 0.9710250, 1.0226896, 0.9233138, 1.0229756, -0.0519506, 0.0993758

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 159
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 107
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B1

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706765, upper bound: 0.0725256
time: 1.89 seconds

## Relational analysis of NS_A2_A1_B2_B2_B1_A2_B2

### Relational analysis result of NS_A2_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0717283, upper bound: 0.0730511
time: 1.86 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 9.78 + 592.55 = 602.33 seconds
