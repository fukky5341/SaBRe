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
execution time: IAR + RelationalAnalysis = 1.85 + 7.74 = 9.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0842349, upper bound: 0.0842349

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0788146, upper bound: 0.0808813
time: 2.54 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0820487, upper bound: 0.0820487
time: 1.60 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 4.30 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 4.30
Output dim: 9, lower bound: -0.0788146, upper bound: 0.0808813
NS_A2, status: Status.UNKNOWN, split count: 1, time: 4.30
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

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
time: 1.66 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0791977
time: 2.32 seconds

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

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800363, upper bound: 0.0810761
time: 2.48 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0803199, upper bound: 0.0803199
time: 1.52 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 5.99 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 5.99
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 5.99
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0791977
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 5.99
Output dim: 9, lower bound: -0.0800363, upper bound: 0.0810761
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 5.99
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

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0789348
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0791983
time: 1.84 seconds

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

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0789343
time: 1.85 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0791983
time: 2.29 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -0.0071354, 0.0037131, -0.0073755, 0.0045030, -0.0116384, 0.0110886
1: -0.0110912, 0.0231018, -0.0109762, 0.0258230, -0.0369142, 0.0340780
2: -0.0017115, 0.0274158, -0.0014398, 0.0296303, -0.0313418, 0.0288556
3: -0.0109653, 0.0085518, -0.0114742, 0.0104296, -0.0213948, 0.0200259
4: -0.0134544, 0.0106406, -0.0152889, 0.0106963, -0.0241507, 0.0259294
5: -0.0095471, 0.0202745, -0.0098001, 0.0225861, -0.0321332, 0.0300745
6: -0.0082503, 0.0099021, -0.0083947, 0.0118409, -0.0200912, 0.0182968
7: -0.0164249, 0.0105317, -0.0187904, 0.0109308, -0.0273557, 0.0293221
8: -0.0094530, 0.0125414, -0.0101851, 0.0146789, -0.0241319, 0.0227265
9: 0.9369770, 1.0223881, 0.9279639, 1.0221602, -0.0851832, 0.0944242

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800363, upper bound: 0.0800363
time: 1.78 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0800363, upper bound: 0.0803199
time: 2.11 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -0.0071365, 0.0037083, -0.0080344, 0.0073059, -0.0144424, 0.0117427
1: -0.0111134, 0.0230966, -0.0122119, 0.0324566, -0.0435700, 0.0353084
2: -0.0017412, 0.0274064, -0.0022765, 0.0351141, -0.0368552, 0.0296829
3: -0.0109655, 0.0085489, -0.0129016, 0.0148288, -0.0257944, 0.0214505
4: -0.0134615, 0.0106550, -0.0198072, 0.0117418, -0.0252033, 0.0304622
5: -0.0095893, 0.0202682, -0.0114846, 0.0281198, -0.0377091, 0.0317529
6: -0.0082499, 0.0099066, -0.0088951, 0.0164555, -0.0247054, 0.0188017
7: -0.0164154, 0.0105347, -0.0243304, 0.0123137, -0.0287291, 0.0348650
8: -0.0094640, 0.0125363, -0.0121418, 0.0197717, -0.0292356, 0.0246781
9: 0.9369969, 1.0224582, 0.9069588, 1.0243294, -0.0873325, 0.1154994

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0803199, upper bound: 0.0800363
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0803199, upper bound: 0.0803199
time: 1.49 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 4.97 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0789348
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0791983
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0789343
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0772707, upper bound: 0.0791983
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0800363, upper bound: 0.0800363
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0800363, upper bound: 0.0803199
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0803199, upper bound: 0.0800363
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.97
Output dim: 9, lower bound: -0.0803199, upper bound: 0.0803199

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

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0778668
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
time: 1.28 seconds

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

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0778662
time: 1.45 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
time: 1.63 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0077765, 0.0060378, -0.0125472, 0.0101720
1: -0.0094869, 0.0179468, -0.0118183, 0.0303054, -0.0397923, 0.0297651
2: -0.0006456, 0.0229024, -0.0021170, 0.0329528, -0.0335985, 0.0250194
3: -0.0096166, 0.0051424, -0.0123267, 0.0135642, -0.0231808, 0.0174691
4: -0.0101306, 0.0097369, -0.0185745, 0.0114567, -0.0215874, 0.0283113
5: -0.0078843, 0.0162008, -0.0109956, 0.0265472, -0.0344315, 0.0271964
6: -0.0077993, 0.0066288, -0.0087181, 0.0151211, -0.0229204, 0.0153469
7: -0.0129358, 0.0092739, -0.0224177, 0.0118767, -0.0248126, 0.0316916
8: -0.0074538, 0.0089322, -0.0112697, 0.0182490, -0.0257028, 0.0202020
9: 0.9529222, 1.0198088, 0.9132261, 1.0237620, -0.0708398, 0.1065826

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0769766
time: 2.77 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0789348
time: 1.93 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0077765, 0.0060378, -0.0130948, 0.0110435
1: -0.0105610, 0.0234683, -0.0118183, 0.0303054, -0.0408663, 0.0352866
2: -0.0014662, 0.0266610, -0.0021170, 0.0329528, -0.0344190, 0.0287780
3: -0.0107759, 0.0093700, -0.0123267, 0.0135642, -0.0243401, 0.0216967
4: -0.0145209, 0.0105283, -0.0185745, 0.0114567, -0.0259776, 0.0291028
5: -0.0094379, 0.0213327, -0.0109956, 0.0265472, -0.0359851, 0.0323283
6: -0.0082263, 0.0106983, -0.0087181, 0.0151211, -0.0233474, 0.0194164
7: -0.0167035, 0.0105577, -0.0224177, 0.0118767, -0.0285802, 0.0329753
8: -0.0089806, 0.0132137, -0.0112697, 0.0182490, -0.0272296, 0.0244834
9: 0.9339586, 1.0218134, 0.9132261, 1.0237620, -0.0898033, 0.1085873

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0772707
time: 1.68 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0791977
time: 1.74 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0073755, 0.0045030, -0.0115944, 0.0108792
1: -0.0105355, 0.0228542, -0.0109762, 0.0258230, -0.0363586, 0.0338304
2: -0.0011940, 0.0270286, -0.0014398, 0.0296303, -0.0308243, 0.0284684
3: -0.0108587, 0.0084947, -0.0114742, 0.0104296, -0.0212883, 0.0199688
4: -0.0133733, 0.0103177, -0.0152889, 0.0106963, -0.0240696, 0.0256065
5: -0.0091526, 0.0201610, -0.0098001, 0.0225861, -0.0317387, 0.0299611
6: -0.0081909, 0.0097837, -0.0083947, 0.0118409, -0.0200318, 0.0181784
7: -0.0161654, 0.0103601, -0.0187904, 0.0109308, -0.0270962, 0.0291505
8: -0.0092823, 0.0124182, -0.0101851, 0.0146789, -0.0239613, 0.0226034
9: 0.9373958, 1.0214237, 0.9279639, 1.0221602, -0.0847644, 0.0934598

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0778662
time: 3.13 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0806019
time: 3.78 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0073755, 0.0045030, -0.0122214, 0.0131459
1: -0.0117038, 0.0293341, -0.0109762, 0.0258230, -0.0375268, 0.0403103
2: -0.0020246, 0.0323330, -0.0014398, 0.0296303, -0.0316550, 0.0337728
3: -0.0121859, 0.0128294, -0.0114742, 0.0104296, -0.0226155, 0.0243035
4: -0.0177987, 0.0113304, -0.0152889, 0.0106963, -0.0284950, 0.0266192
5: -0.0107822, 0.0256450, -0.0098001, 0.0225861, -0.0333683, 0.0354451
6: -0.0086578, 0.0143819, -0.0083947, 0.0118409, -0.0204987, 0.0227766
7: -0.0216183, 0.0116937, -0.0187904, 0.0109308, -0.0325491, 0.0304840
8: -0.0110805, 0.0174725, -0.0101851, 0.0146789, -0.0257594, 0.0276576
9: 0.9165484, 1.0235258, 0.9279639, 1.0221602, -0.1056117, 0.0955619

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0778662
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0806019
time: 1.58 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0080344, 0.0073059, -0.0143973, 0.0115381
1: -0.0105355, 0.0228542, -0.0122119, 0.0324566, -0.0429922, 0.0350661
2: -0.0011940, 0.0270286, -0.0022765, 0.0351141, -0.0363080, 0.0293051
3: -0.0108587, 0.0084947, -0.0129016, 0.0148288, -0.0256875, 0.0213963
4: -0.0133733, 0.0103177, -0.0198072, 0.0117418, -0.0251150, 0.0301249
5: -0.0091526, 0.0201610, -0.0114846, 0.0281198, -0.0372724, 0.0316456
6: -0.0081909, 0.0097837, -0.0088951, 0.0164555, -0.0246464, 0.0186788
7: -0.0161654, 0.0103601, -0.0243304, 0.0123137, -0.0284792, 0.0346905
8: -0.0092823, 0.0124182, -0.0121418, 0.0197717, -0.0290540, 0.0245600
9: 0.9373958, 1.0214237, 0.9069588, 1.0243294, -0.0869337, 0.1144649

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0769766
time: 1.92 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0795968
time: 2.39 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0080344, 0.0073059, -0.0150242, 0.0138047
1: -0.0117038, 0.0293341, -0.0122119, 0.0324566, -0.0441604, 0.0415460
2: -0.0020246, 0.0323330, -0.0022765, 0.0351141, -0.0371387, 0.0346095
3: -0.0121859, 0.0128294, -0.0129016, 0.0148288, -0.0270148, 0.0257310
4: -0.0177987, 0.0113304, -0.0198072, 0.0117418, -0.0295404, 0.0311376
5: -0.0107822, 0.0256450, -0.0114846, 0.0281198, -0.0389020, 0.0371296
6: -0.0086578, 0.0143819, -0.0088951, 0.0164555, -0.0251133, 0.0232770
7: -0.0216183, 0.0116937, -0.0243304, 0.0123137, -0.0339321, 0.0360240
8: -0.0110805, 0.0174725, -0.0121418, 0.0197717, -0.0308522, 0.0296142
9: 0.9165484, 1.0235258, 0.9069588, 1.0243294, -0.1077810, 0.1165671

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0772707
time: 1.77 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0798269
time: 1.76 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 5.39 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0778668
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0778662
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0798250
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0769766
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0789348
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0772707
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0769763, upper bound: 0.0791977
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0778662
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0806019
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0778662
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0806019
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0769766
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0795968
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0772707
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 5.39
Output dim: 9, lower bound: -0.0789343, upper bound: 0.0798269

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0065094, 0.0023956, -0.0089050, 0.0089050
1: -0.0094869, 0.0179468, -0.0094869, 0.0179468, -0.0274337, 0.0274337
2: -0.0006456, 0.0229024, -0.0006456, 0.0229024, -0.0235480, 0.0235480
3: -0.0096166, 0.0051424, -0.0096166, 0.0051424, -0.0147590, 0.0147590
4: -0.0101306, 0.0097369, -0.0101306, 0.0097369, -0.0198675, 0.0198675
5: -0.0078843, 0.0162008, -0.0078843, 0.0162008, -0.0240851, 0.0240851
6: -0.0077993, 0.0066288, -0.0077993, 0.0066288, -0.0144281, 0.0144281
7: -0.0129358, 0.0092739, -0.0129358, 0.0092739, -0.0222098, 0.0222098
8: -0.0074538, 0.0089322, -0.0074538, 0.0089322, -0.0163860, 0.0163860
9: 0.9529222, 1.0198088, 0.9529222, 1.0198088, -0.0668865, 0.0668865

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715980, upper bound: 0.0685241
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0070914, 0.0035037, -0.0100131, 0.0094870
1: -0.0094869, 0.0179468, -0.0105355, 0.0228542, -0.0323411, 0.0284823
2: -0.0006456, 0.0229024, -0.0011940, 0.0270286, -0.0276742, 0.0240964
3: -0.0096166, 0.0051424, -0.0108587, 0.0084947, -0.0181113, 0.0160011
4: -0.0101306, 0.0097369, -0.0133733, 0.0103177, -0.0204483, 0.0231101
5: -0.0078843, 0.0162008, -0.0091526, 0.0201610, -0.0280453, 0.0253534
6: -0.0077993, 0.0066288, -0.0081909, 0.0097837, -0.0175830, 0.0148197
7: -0.0129358, 0.0092739, -0.0161654, 0.0103601, -0.0232959, 0.0254394
8: -0.0074538, 0.0089322, -0.0092823, 0.0124182, -0.0198720, 0.0182146
9: 0.9529222, 1.0198088, 0.9373958, 1.0214237, -0.0685015, 0.0824130

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715980, upper bound: 0.0712505
time: 1.38 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0065094, 0.0023956, -0.0094526, 0.0097765
1: -0.0105610, 0.0234683, -0.0094869, 0.0179468, -0.0285078, 0.0329552
2: -0.0014662, 0.0266610, -0.0006456, 0.0229024, -0.0243686, 0.0273066
3: -0.0107759, 0.0093700, -0.0096166, 0.0051424, -0.0159183, 0.0189866
4: -0.0145209, 0.0105283, -0.0101306, 0.0097369, -0.0242578, 0.0206589
5: -0.0094379, 0.0213327, -0.0078843, 0.0162008, -0.0256387, 0.0292171
6: -0.0082263, 0.0106983, -0.0077993, 0.0066288, -0.0148551, 0.0184976
7: -0.0167035, 0.0105577, -0.0129358, 0.0092739, -0.0259774, 0.0234935
8: -0.0089806, 0.0132137, -0.0074538, 0.0089322, -0.0179129, 0.0206675
9: 0.9339586, 1.0218134, 0.9529222, 1.0198088, -0.0858501, 0.0688912

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
time: 3.53 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 1.42 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0070914, 0.0035037, -0.0105607, 0.0103584
1: -0.0105610, 0.0234683, -0.0105355, 0.0228542, -0.0334152, 0.0340039
2: -0.0014662, 0.0266610, -0.0011940, 0.0270286, -0.0284948, 0.0278549
3: -0.0107759, 0.0093700, -0.0108587, 0.0084947, -0.0192706, 0.0202287
4: -0.0145209, 0.0105283, -0.0133733, 0.0103177, -0.0248386, 0.0239016
5: -0.0094379, 0.0213327, -0.0091526, 0.0201610, -0.0295989, 0.0304853
6: -0.0082263, 0.0106983, -0.0081909, 0.0097837, -0.0180101, 0.0188892
7: -0.0167035, 0.0105577, -0.0161654, 0.0103601, -0.0270636, 0.0267231
8: -0.0089806, 0.0132137, -0.0092823, 0.0124182, -0.0213989, 0.0224960
9: 0.9339586, 1.0218134, 0.9373958, 1.0214237, -0.0874650, 0.0844176

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0707374
time: 2.05 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.75 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0070570, 0.0032846, -0.0097940, 0.0094526
1: -0.0094869, 0.0179468, -0.0105610, 0.0234959, -0.0329828, 0.0285078
2: -0.0006456, 0.0229024, -0.0014662, 0.0267031, -0.0273488, 0.0243686
3: -0.0096166, 0.0051424, -0.0107759, 0.0093753, -0.0189918, 0.0159183
4: -0.0101306, 0.0097369, -0.0145209, 0.0105345, -0.0206651, 0.0242578
5: -0.0078843, 0.0162008, -0.0094379, 0.0213454, -0.0292297, 0.0256387
6: -0.0077993, 0.0066288, -0.0082263, 0.0107164, -0.0185157, 0.0148551
7: -0.0129358, 0.0092739, -0.0167598, 0.0105577, -0.0234935, 0.0260338
8: -0.0074538, 0.0089322, -0.0089806, 0.0132422, -0.0206960, 0.0179129
9: 0.9529222, 1.0198088, 0.9338889, 1.0218134, -0.0688912, 0.0859199

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708206, upper bound: 0.0669953
time: 1.29 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
time: 1.17 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0065094, 0.0023956, -0.0077183, 0.0057703, -0.0122798, 0.0101139
1: -0.0094869, 0.0179468, -0.0117038, 0.0293341, -0.0388210, 0.0296505
2: -0.0006456, 0.0229024, -0.0020246, 0.0323330, -0.0329786, 0.0249270
3: -0.0096166, 0.0051424, -0.0121859, 0.0128294, -0.0224460, 0.0173283
4: -0.0101306, 0.0097369, -0.0177987, 0.0113304, -0.0214610, 0.0275356
5: -0.0078843, 0.0162008, -0.0107822, 0.0256450, -0.0335293, 0.0269830
6: -0.0077993, 0.0066288, -0.0086578, 0.0143819, -0.0221812, 0.0152866
7: -0.0129358, 0.0092739, -0.0216183, 0.0116937, -0.0246295, 0.0308923
8: -0.0074538, 0.0089322, -0.0110805, 0.0174725, -0.0249262, 0.0200127
9: 0.9529222, 1.0198088, 0.9165484, 1.0235258, -0.0706036, 0.1032603

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708206, upper bound: 0.0696418
time: 1.64 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0070570, 0.0032846, -0.0103416, 0.0103240
1: -0.0105610, 0.0234683, -0.0105610, 0.0234959, -0.0340568, 0.0340293
2: -0.0014662, 0.0266610, -0.0014662, 0.0267031, -0.0281693, 0.0281271
3: -0.0107759, 0.0093700, -0.0107759, 0.0093753, -0.0201512, 0.0201459
4: -0.0145209, 0.0105283, -0.0145209, 0.0105345, -0.0250554, 0.0250492
5: -0.0094379, 0.0213327, -0.0094379, 0.0213454, -0.0307833, 0.0307706
6: -0.0082263, 0.0106983, -0.0082263, 0.0107164, -0.0189427, 0.0189247
7: -0.0167035, 0.0105577, -0.0167598, 0.0105577, -0.0272612, 0.0273175
8: -0.0089806, 0.0132137, -0.0089806, 0.0132422, -0.0222229, 0.0221943
9: 0.9339586, 1.0218134, 0.9338889, 1.0218134, -0.0878547, 0.0879245

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702395, upper bound: 0.0672260
time: 1.51 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0070570, 0.0032670, -0.0077183, 0.0057703, -0.0128273, 0.0109854
1: -0.0105610, 0.0234683, -0.0117038, 0.0293341, -0.0398951, 0.0351721
2: -0.0014662, 0.0266610, -0.0020246, 0.0323330, -0.0337992, 0.0286856
3: -0.0107759, 0.0093700, -0.0121859, 0.0128294, -0.0236053, 0.0215559
4: -0.0145209, 0.0105283, -0.0177987, 0.0113304, -0.0258513, 0.0283270
5: -0.0094379, 0.0213327, -0.0107822, 0.0256450, -0.0350829, 0.0321150
6: -0.0082263, 0.0106983, -0.0086578, 0.0143819, -0.0226082, 0.0193562
7: -0.0167035, 0.0105577, -0.0216183, 0.0116937, -0.0283971, 0.0321760
8: -0.0089806, 0.0132137, -0.0110805, 0.0174725, -0.0264531, 0.0242942
9: 0.9339586, 1.0218134, 0.9165484, 1.0235258, -0.0895672, 0.1052650

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702395, upper bound: 0.0698043
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1_B1

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

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
time: 2.33 seconds

## BFS NS instance: NS_A2_B1_A1_B2

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

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0717021
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0714703
time: 1.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0065094, 0.0023956, -0.0101139, 0.0122798
1: -0.0117038, 0.0293341, -0.0094869, 0.0179468, -0.0296505, 0.0388210
2: -0.0020246, 0.0323330, -0.0006456, 0.0229024, -0.0249270, 0.0329786
3: -0.0121859, 0.0128294, -0.0096166, 0.0051424, -0.0173283, 0.0224460
4: -0.0177987, 0.0113304, -0.0101306, 0.0097369, -0.0275356, 0.0214610
5: -0.0107822, 0.0256450, -0.0078843, 0.0162008, -0.0269830, 0.0335293
6: -0.0086578, 0.0143819, -0.0077993, 0.0066288, -0.0152866, 0.0221812
7: -0.0216183, 0.0116937, -0.0129358, 0.0092739, -0.0308923, 0.0246295
8: -0.0110805, 0.0174725, -0.0074538, 0.0089322, -0.0200127, 0.0249262
9: 0.9165484, 1.0235258, 0.9529222, 1.0198088, -0.1032603, 0.0706036

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0682461
time: 1.43 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
time: 1.55 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0070914, 0.0035037, -0.0112220, 0.0128617
1: -0.0117038, 0.0293341, -0.0105355, 0.0228542, -0.0345580, 0.0398696
2: -0.0020246, 0.0323330, -0.0011940, 0.0270286, -0.0290533, 0.0335270
3: -0.0121859, 0.0128294, -0.0108587, 0.0084947, -0.0206806, 0.0236881
4: -0.0177987, 0.0113304, -0.0133733, 0.0103177, -0.0281164, 0.0247036
5: -0.0107822, 0.0256450, -0.0091526, 0.0201610, -0.0309432, 0.0347976
6: -0.0086578, 0.0143819, -0.0081909, 0.0097837, -0.0184416, 0.0225728
7: -0.0216183, 0.0116937, -0.0161654, 0.0103601, -0.0319785, 0.0278591
8: -0.0110805, 0.0174725, -0.0092823, 0.0124182, -0.0234988, 0.0267548
9: 0.9165484, 1.0235258, 0.9373958, 1.0214237, -0.1048753, 0.0861301

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0712693
time: 1.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0708310
time: 1.35 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0070914, 0.0035037, -0.0070570, 0.0032846, -0.0103760, 0.0105607
1: -0.0105355, 0.0228542, -0.0105610, 0.0234959, -0.0340314, 0.0334152
2: -0.0011940, 0.0270286, -0.0014662, 0.0267031, -0.0278971, 0.0284948
3: -0.0108587, 0.0084947, -0.0107759, 0.0093753, -0.0202340, 0.0192706
4: -0.0133733, 0.0103177, -0.0145209, 0.0105345, -0.0239077, 0.0248386
5: -0.0091526, 0.0201610, -0.0094379, 0.0213454, -0.0304980, 0.0295989
6: -0.0081909, 0.0097837, -0.0082263, 0.0107164, -0.0189073, 0.0180101
7: -0.0161654, 0.0103601, -0.0167598, 0.0105577, -0.0267231, 0.0271200
8: -0.0092823, 0.0124182, -0.0089806, 0.0132422, -0.0225246, 0.0213989
9: 0.9373958, 1.0214237, 0.9338889, 1.0218134, -0.0844176, 0.0875348

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0669953
time: 1.51 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 2.17 seconds

## BFS NS instance: NS_A2_B2_A1_B2

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

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0699814
time: 1.41 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0697385
time: 2.41 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0070570, 0.0032846, -0.0110029, 0.0128273
1: -0.0117038, 0.0293341, -0.0105610, 0.0234959, -0.0351996, 0.0398951
2: -0.0020246, 0.0323330, -0.0014662, 0.0267031, -0.0287278, 0.0337992
3: -0.0121859, 0.0128294, -0.0107759, 0.0093753, -0.0215612, 0.0236053
4: -0.0177987, 0.0113304, -0.0145209, 0.0105345, -0.0283332, 0.0258513
5: -0.0107822, 0.0256450, -0.0094379, 0.0213454, -0.0321276, 0.0350829
6: -0.0086578, 0.0143819, -0.0082263, 0.0107164, -0.0193742, 0.0226082
7: -0.0216183, 0.0116937, -0.0167598, 0.0105577, -0.0321760, 0.0284535
8: -0.0110805, 0.0174725, -0.0089806, 0.0132422, -0.0243227, 0.0264531
9: 0.9165484, 1.0235258, 0.9338889, 1.0218134, -0.1052650, 0.0896369

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
time: 1.89 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0077183, 0.0057703, -0.0077183, 0.0057703, -0.0134887, 0.0134887
1: -0.0117038, 0.0293341, -0.0117038, 0.0293341, -0.0410379, 0.0410379
2: -0.0020246, 0.0323330, -0.0020246, 0.0323330, -0.0343577, 0.0343577
3: -0.0121859, 0.0128294, -0.0121859, 0.0128294, -0.0250153, 0.0250153
4: -0.0177987, 0.0113304, -0.0177987, 0.0113304, -0.0291291, 0.0291291
5: -0.0107822, 0.0256450, -0.0107822, 0.0256450, -0.0364272, 0.0364272
6: -0.0086578, 0.0143819, -0.0086578, 0.0143819, -0.0230397, 0.0230397
7: -0.0216183, 0.0116937, -0.0216183, 0.0116937, -0.0333120, 0.0333120
8: -0.0110805, 0.0174725, -0.0110805, 0.0174725, -0.0285530, 0.0285530
9: 0.9165484, 1.0235258, 0.9165484, 1.0235258, -0.1069774, 0.1069774

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0701976
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0698457
time: 1.53 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.73 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0715980, upper bound: 0.0685241
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0715980, upper bound: 0.0712505
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0707374
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0708206, upper bound: 0.0669953
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0708206, upper bound: 0.0696418
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0702395, upper bound: 0.0672260
NS_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0702395, upper bound: 0.0698043
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0717021
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0714703
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0682461
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0712693
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0708310
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0669953
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0699814
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0697385
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0672260
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0701976
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0698457

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0065094, 0.0023956, -0.0087334, 0.0087279
1: -0.0092294, 0.0163209, -0.0094869, 0.0179468, -0.0271762, 0.0258078
2: -0.0004811, 0.0218100, -0.0006456, 0.0229024, -0.0233834, 0.0224556
3: -0.0092587, 0.0038791, -0.0096166, 0.0051424, -0.0144011, 0.0134957
4: -0.0089070, 0.0095615, -0.0101306, 0.0097369, -0.0186439, 0.0196922
5: -0.0074959, 0.0147249, -0.0078843, 0.0162008, -0.0236967, 0.0226092
6: -0.0076797, 0.0054675, -0.0077993, 0.0066288, -0.0143085, 0.0132668
7: -0.0119043, 0.0089149, -0.0129358, 0.0092739, -0.0211782, 0.0218507
8: -0.0069978, 0.0076339, -0.0074538, 0.0089322, -0.0159300, 0.0150877
9: 0.9586030, 1.0193583, 0.9529222, 1.0198088, -0.0612057, 0.0664361

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
time: 1.08 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057862, 0.0029840, -0.0062850, 0.0021481, -0.0079343, 0.0092691
1: -0.0104647, 0.0095069, -0.0091328, 0.0154668, -0.0259316, 0.0186398
2: -0.0015217, 0.0174271, -0.0003904, 0.0213073, -0.0228290, 0.0178175
3: -0.0081027, -0.0017373, -0.0091449, 0.0031386, -0.0112413, 0.0074076
4: -0.0033245, 0.0098740, -0.0081821, 0.0094647, -0.0127893, 0.0180561
5: -0.0075501, 0.0078286, -0.0073080, 0.0138632, -0.0214133, 0.0151367
6: -0.0075438, 0.0001989, -0.0076258, 0.0047888, -0.0123326, 0.0078247
7: -0.0074399, 0.0084411, -0.0112734, 0.0087440, -0.0161839, 0.0197145
8: -0.0055036, 0.0015188, -0.0068400, 0.0068996, -0.0124032, 0.0083588
9: 0.9860020, 1.0210018, 0.9619496, 1.0191458, -0.0331438, 0.0590522

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0670096, upper bound: 0.0664895
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
time: 1.31 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0070914, 0.0035037, -0.0098415, 0.0093099
1: -0.0092294, 0.0163209, -0.0105355, 0.0228542, -0.0320836, 0.0268564
2: -0.0004811, 0.0218100, -0.0011940, 0.0270286, -0.0275097, 0.0230040
3: -0.0092587, 0.0038791, -0.0108587, 0.0084947, -0.0177534, 0.0147378
4: -0.0089070, 0.0095615, -0.0133733, 0.0103177, -0.0192247, 0.0229348
5: -0.0074959, 0.0147249, -0.0091526, 0.0201610, -0.0276569, 0.0238775
6: -0.0076797, 0.0054675, -0.0081909, 0.0097837, -0.0174634, 0.0136584
7: -0.0119043, 0.0089149, -0.0161654, 0.0103601, -0.0222644, 0.0250803
8: -0.0069978, 0.0076339, -0.0092823, 0.0124182, -0.0194161, 0.0169162
9: 0.9586030, 1.0193583, 0.9373958, 1.0214237, -0.0628207, 0.0819625

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 1.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 1.22 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057862, 0.0029840, -0.0068364, 0.0028922, -0.0086784, 0.0098204
1: -0.0104647, 0.0095069, -0.0101376, 0.0201230, -0.0305878, 0.0196445
2: -0.0015217, 0.0174271, -0.0009517, 0.0249459, -0.0264676, 0.0183789
3: -0.0081027, -0.0017373, -0.0103133, 0.0064884, -0.0145911, 0.0085760
4: -0.0033245, 0.0098740, -0.0114524, 0.0100163, -0.0133408, 0.0213263
5: -0.0075501, 0.0078286, -0.0085547, 0.0177797, -0.0253298, 0.0163833
6: -0.0075438, 0.0001989, -0.0080066, 0.0078611, -0.0154050, 0.0082055
7: -0.0074399, 0.0084411, -0.0141469, 0.0098233, -0.0172632, 0.0225880
8: -0.0055036, 0.0015188, -0.0085372, 0.0102599, -0.0157635, 0.0100560
9: 0.9860020, 1.0210018, 0.9466540, 1.0207405, -0.0347385, 0.0743478

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0670096, upper bound: 0.0689694
time: 2.31 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
time: 1.34 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

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

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 2.37 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 2.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

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

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0652147, upper bound: 0.0659473
time: 1.32 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
time: 1.47 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

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

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.23 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0063058, 0.0034586, -0.0068364, 0.0028922, -0.0091981, 0.0102950
1: -0.0114675, 0.0142369, -0.0101376, 0.0201230, -0.0315906, 0.0243745
2: -0.0023040, 0.0210116, -0.0009517, 0.0249459, -0.0272499, 0.0219633
3: -0.0091753, 0.0018162, -0.0103133, 0.0064884, -0.0156637, 0.0121295
4: -0.0075261, 0.0106122, -0.0114524, 0.0100163, -0.0175423, 0.0220646
5: -0.0089659, 0.0127776, -0.0085547, 0.0177797, -0.0267456, 0.0213323
6: -0.0079179, 0.0039991, -0.0080066, 0.0078611, -0.0157790, 0.0120057
7: -0.0108489, 0.0096242, -0.0141469, 0.0098233, -0.0206722, 0.0237711
8: -0.0068596, 0.0055431, -0.0085372, 0.0102599, -0.0171195, 0.0140802
9: 0.9677531, 1.0228958, 0.9466540, 1.0207405, -0.0529875, 0.0762418

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0652203, upper bound: 0.0682734
time: 1.42 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
time: 1.51 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0070570, 0.0032846, -0.0096224, 0.0092755
1: -0.0092294, 0.0163209, -0.0105610, 0.0234959, -0.0327253, 0.0268819
2: -0.0004811, 0.0218100, -0.0014662, 0.0267031, -0.0271842, 0.0232762
3: -0.0092587, 0.0038791, -0.0107759, 0.0093753, -0.0186340, 0.0146550
4: -0.0089070, 0.0095615, -0.0145209, 0.0105345, -0.0194415, 0.0240824
5: -0.0074959, 0.0147249, -0.0094379, 0.0213454, -0.0288413, 0.0241628
6: -0.0076797, 0.0054675, -0.0082263, 0.0107164, -0.0183961, 0.0136938
7: -0.0119043, 0.0089149, -0.0167598, 0.0105577, -0.0224619, 0.0256747
8: -0.0069978, 0.0076339, -0.0089806, 0.0132422, -0.0202400, 0.0166145
9: 0.9586030, 1.0193583, 0.9338889, 1.0218134, -0.0632104, 0.0854694

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
time: 1.38 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
time: 1.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0057862, 0.0029840, -0.0068342, 0.0028674, -0.0086535, 0.0098182
1: -0.0104647, 0.0095069, -0.0102035, 0.0208754, -0.0313401, 0.0197104
2: -0.0015217, 0.0174271, -0.0012271, 0.0249953, -0.0265170, 0.0186542
3: -0.0081027, -0.0017373, -0.0103002, 0.0073569, -0.0154596, 0.0085629
4: -0.0033245, 0.0098740, -0.0125772, 0.0102682, -0.0135927, 0.0224512
5: -0.0075501, 0.0078286, -0.0088654, 0.0190138, -0.0265639, 0.0166940
6: -0.0075438, 0.0001989, -0.0080558, 0.0088575, -0.0164013, 0.0082547
7: -0.0074399, 0.0084411, -0.0150379, 0.0100487, -0.0174886, 0.0234790
8: -0.0055036, 0.0015188, -0.0083491, 0.0111824, -0.0166860, 0.0098680
9: 0.9860020, 1.0210018, 0.9428862, 1.0211681, -0.0351661, 0.0781156

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659973, upper bound: 0.0642977
time: 1.33 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0077183, 0.0057703, -0.0121081, 0.0099368
1: -0.0092294, 0.0163209, -0.0117038, 0.0293341, -0.0385635, 0.0280246
2: -0.0004811, 0.0218100, -0.0020246, 0.0323330, -0.0328141, 0.0238346
3: -0.0092587, 0.0038791, -0.0121859, 0.0128294, -0.0220881, 0.0160650
4: -0.0089070, 0.0095615, -0.0177987, 0.0113304, -0.0202374, 0.0273602
5: -0.0074959, 0.0147249, -0.0107822, 0.0256450, -0.0331409, 0.0255071
6: -0.0076797, 0.0054675, -0.0086578, 0.0143819, -0.0220616, 0.0141253
7: -0.0119043, 0.0089149, -0.0216183, 0.0116937, -0.0235979, 0.0305332
8: -0.0069978, 0.0076339, -0.0110805, 0.0174725, -0.0244703, 0.0187144
9: 0.9586030, 1.0193583, 0.9165484, 1.0235258, -0.0649228, 0.1028098

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
time: 2.71 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0057862, 0.0029840, -0.0074549, 0.0046669, -0.0104530, 0.0104390
1: -0.0104647, 0.0095069, -0.0112924, 0.0263954, -0.0368602, 0.0207993
2: -0.0015217, 0.0174271, -0.0017885, 0.0300154, -0.0315371, 0.0192156
3: -0.0081027, -0.0017373, -0.0116232, 0.0108123, -0.0189150, 0.0098859
4: -0.0033245, 0.0098740, -0.0158700, 0.0109734, -0.0142980, 0.0257440
5: -0.0075501, 0.0078286, -0.0101784, 0.0232333, -0.0307834, 0.0180070
6: -0.0075438, 0.0001989, -0.0084650, 0.0123627, -0.0199065, 0.0086639
7: -0.0074399, 0.0084411, -0.0192828, 0.0111584, -0.0185983, 0.0277238
8: -0.0055036, 0.0015188, -0.0103028, 0.0151758, -0.0206793, 0.0118216
9: 0.9860020, 1.0210018, 0.9260604, 1.0228342, -0.0368322, 0.0949414

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0659973, upper bound: 0.0669314
time: 1.49 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
time: 1.36 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0070570, 0.0032846, -0.0101588, 0.0100046
1: -0.0102804, 0.0216467, -0.0105610, 0.0234959, -0.0337763, 0.0322077
2: -0.0013008, 0.0254146, -0.0014662, 0.0267031, -0.0280040, 0.0268808
3: -0.0103910, 0.0080113, -0.0107759, 0.0093753, -0.0197662, 0.0187872
4: -0.0132094, 0.0103480, -0.0145209, 0.0105345, -0.0237439, 0.0248689
5: -0.0090217, 0.0197614, -0.0094379, 0.0213454, -0.0303670, 0.0291992
6: -0.0080985, 0.0094561, -0.0082263, 0.0107164, -0.0188148, 0.0176824
7: -0.0155814, 0.0101845, -0.0167598, 0.0105577, -0.0261391, 0.0269444
8: -0.0084719, 0.0118298, -0.0089806, 0.0132422, -0.0217141, 0.0208104
9: 0.9400309, 1.0213395, 0.9338889, 1.0218134, -0.0817825, 0.0874506

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
time: 3.02 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
time: 1.33 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0077183, 0.0057703, -0.0126445, 0.0106660
1: -0.0102804, 0.0216467, -0.0117038, 0.0293341, -0.0396145, 0.0333504
2: -0.0013008, 0.0254146, -0.0020246, 0.0323330, -0.0336338, 0.0274392
3: -0.0103910, 0.0080113, -0.0121859, 0.0128294, -0.0232204, 0.0201972
4: -0.0132094, 0.0103480, -0.0177987, 0.0113304, -0.0245397, 0.0281467
5: -0.0090217, 0.0197614, -0.0107822, 0.0256450, -0.0346667, 0.0305436
6: -0.0080985, 0.0094561, -0.0086578, 0.0143819, -0.0224804, 0.0181139
7: -0.0155814, 0.0101845, -0.0216183, 0.0116937, -0.0272751, 0.0318029
8: -0.0084719, 0.0118298, -0.0110805, 0.0174725, -0.0259443, 0.0229103
9: 0.9400309, 1.0213395, 0.9165484, 1.0235258, -0.0834950, 0.1047911

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.47 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0063058, 0.0034586, -0.0074549, 0.0046669, -0.0109727, 0.0109136
1: -0.0114675, 0.0142369, -0.0112924, 0.0263954, -0.0378630, 0.0255293
2: -0.0023040, 0.0210116, -0.0017885, 0.0300154, -0.0323195, 0.0228000
3: -0.0091753, 0.0018162, -0.0116232, 0.0108123, -0.0199875, 0.0134394
4: -0.0075261, 0.0106122, -0.0158700, 0.0109734, -0.0184995, 0.0264822
5: -0.0089659, 0.0127776, -0.0101784, 0.0232333, -0.0321992, 0.0229560
6: -0.0079179, 0.0039991, -0.0084650, 0.0123627, -0.0202805, 0.0124641
7: -0.0108489, 0.0096242, -0.0192828, 0.0111584, -0.0220072, 0.0289070
8: -0.0068596, 0.0055431, -0.0103028, 0.0151758, -0.0220354, 0.0158459
9: 0.9677531, 1.0228958, 0.9260604, 1.0228342, -0.0550811, 0.0968354

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0651608, upper bound: 0.0669817
time: 1.10 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
time: 1.58 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

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

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
time: 1.22 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
time: 1.48 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

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

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0689735, upper bound: 0.0664895
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
time: 3.14 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

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

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
time: 1.94 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
time: 1.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

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

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0705570, upper bound: 0.0698083
time: 1.88 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

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

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
time: 1.45 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

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

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0671227, upper bound: 0.0659473
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
time: 1.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

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

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0708310
time: 1.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0708310
time: 2.56 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

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

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687243, upper bound: 0.0690855
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0708310
time: 2.43 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0070570, 0.0032846, -0.0101561, 0.0100259
1: -0.0102067, 0.0208570, -0.0105610, 0.0234959, -0.0337026, 0.0314180
2: -0.0010219, 0.0253591, -0.0014662, 0.0267031, -0.0277251, 0.0268253
3: -0.0103941, 0.0071250, -0.0107759, 0.0093753, -0.0197693, 0.0179009
4: -0.0120788, 0.0100930, -0.0145209, 0.0105345, -0.0226133, 0.0246139
5: -0.0086981, 0.0185246, -0.0094379, 0.0213454, -0.0300434, 0.0279625
6: -0.0080465, 0.0084473, -0.0082263, 0.0107164, -0.0187628, 0.0166736
7: -0.0146922, 0.0099509, -0.0167598, 0.0105577, -0.0252499, 0.0267108
8: -0.0086411, 0.0108853, -0.0089806, 0.0132422, -0.0218833, 0.0198659
9: 0.9438275, 1.0209014, 0.9338889, 1.0218134, -0.0779859, 0.0870125

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 1.38 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 3.58 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -0.0063068, 0.0034323, -0.0068342, 0.0028674, -0.0091742, 0.0102665
1: -0.0114994, 0.0137490, -0.0102035, 0.0208754, -0.0323748, 0.0239525
2: -0.0021608, 0.0209964, -0.0012271, 0.0249953, -0.0271561, 0.0222235
3: -0.0092192, 0.0010684, -0.0103002, 0.0073569, -0.0165761, 0.0113686
4: -0.0065257, 0.0104626, -0.0125772, 0.0102682, -0.0167939, 0.0230399
5: -0.0087947, 0.0117027, -0.0088654, 0.0190138, -0.0278086, 0.0205681
6: -0.0079002, 0.0031316, -0.0080558, 0.0088575, -0.0167577, 0.0111874
7: -0.0100647, 0.0095251, -0.0150379, 0.0100487, -0.0201134, 0.0245630
8: -0.0070877, 0.0047627, -0.0083491, 0.0111824, -0.0182701, 0.0131118
9: 0.9710250, 1.0226896, 0.9428862, 1.0211681, -0.0501431, 0.0798033

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678722, upper bound: 0.0642977
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
time: 2.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0077183, 0.0057703, -0.0126418, 0.0106872
1: -0.0102067, 0.0208570, -0.0117038, 0.0293341, -0.0395409, 0.0325608
2: -0.0010219, 0.0253591, -0.0020246, 0.0323330, -0.0333550, 0.0273838
3: -0.0103941, 0.0071250, -0.0121859, 0.0128294, -0.0232235, 0.0193109
4: -0.0120788, 0.0100930, -0.0177987, 0.0113304, -0.0234092, 0.0278917
5: -0.0086981, 0.0185246, -0.0107822, 0.0256450, -0.0343431, 0.0293069
6: -0.0080465, 0.0084473, -0.0086578, 0.0143819, -0.0224284, 0.0171051
7: -0.0146922, 0.0099509, -0.0216183, 0.0116937, -0.0263859, 0.0315693
8: -0.0086411, 0.0108853, -0.0110805, 0.0174725, -0.0261136, 0.0219658
9: 0.9438275, 1.0209014, 0.9165484, 1.0235258, -0.0796983, 0.1043530

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 1.58 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 1.50 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -0.0063068, 0.0034323, -0.0074549, 0.0046669, -0.0109736, 0.0108872
1: -0.0114994, 0.0137490, -0.0112924, 0.0263954, -0.0378949, 0.0250413
2: -0.0021608, 0.0209964, -0.0017885, 0.0300154, -0.0321762, 0.0227848
3: -0.0092192, 0.0010684, -0.0116232, 0.0108123, -0.0200314, 0.0126916
4: -0.0065257, 0.0104626, -0.0158700, 0.0109734, -0.0174991, 0.0263326
5: -0.0087947, 0.0117027, -0.0101784, 0.0232333, -0.0320280, 0.0218812
6: -0.0079002, 0.0031316, -0.0084650, 0.0123627, -0.0202629, 0.0115966
7: -0.0100647, 0.0095251, -0.0192828, 0.0111584, -0.0212231, 0.0288079
8: -0.0070877, 0.0047627, -0.0103028, 0.0151758, -0.0222634, 0.0150655
9: 0.9710250, 1.0226896, 0.9260604, 1.0228342, -0.0518092, 0.0966291

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0694993, upper bound: 0.0677252
time: 1.43 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
time: 1.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0070570, 0.0032846, -0.0107714, 0.0118050
1: -0.0113465, 0.0271213, -0.0105610, 0.0234959, -0.0348424, 0.0376823
2: -0.0018529, 0.0303596, -0.0014662, 0.0267031, -0.0285561, 0.0318258
3: -0.0116776, 0.0114324, -0.0107759, 0.0093753, -0.0210529, 0.0222083
4: -0.0164778, 0.0110528, -0.0145209, 0.0105345, -0.0270123, 0.0255737
5: -0.0103079, 0.0239427, -0.0094379, 0.0213454, -0.0316533, 0.0333806
6: -0.0084978, 0.0129430, -0.0082263, 0.0107164, -0.0192141, 0.0211693
7: -0.0198323, 0.0112676, -0.0167598, 0.0105577, -0.0303900, 0.0280274
8: -0.0103719, 0.0157998, -0.0089806, 0.0132422, -0.0236141, 0.0247804
9: 0.9233138, 1.0229756, 0.9338889, 1.0218134, -0.0984996, 0.0890867

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.18 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.46 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

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

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0670587, upper bound: 0.0643809
time: 1.34 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
time: 1.44 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

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

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 144

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
time: 2.16 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -0.0068255, 0.0039014, -0.0074549, 0.0046669, -0.0114924, 0.0113563
1: -0.0124919, 0.0187966, -0.0112924, 0.0263954, -0.0388874, 0.0300890
2: -0.0029244, 0.0245749, -0.0017885, 0.0300154, -0.0329398, 0.0263634
3: -0.0103170, 0.0050977, -0.0116232, 0.0108123, -0.0211292, 0.0167209
4: -0.0107890, 0.0111772, -0.0158700, 0.0109734, -0.0217624, 0.0270472
5: -0.0101817, 0.0166798, -0.0101784, 0.0232333, -0.0334150, 0.0268582
6: -0.0082751, 0.0070263, -0.0084650, 0.0123627, -0.0206378, 0.0154913
7: -0.0136281, 0.0106762, -0.0192828, 0.0111584, -0.0247865, 0.0299590
8: -0.0085291, 0.0088583, -0.0103028, 0.0151758, -0.0237049, 0.0191611
9: 0.9525756, 1.0245186, 0.9260604, 1.0228342, -0.0702586, 0.0984582

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 107
type: B, layer: 1, pos: 104
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 159
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0687034, upper bound: 0.0678018
time: 1.93 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
time: 1.62 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 5.32 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
NS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0670096, upper bound: 0.0664895
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0682390
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0670096, upper bound: 0.0689694
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0682390, upper bound: 0.0706720
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
NS_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0652147, upper bound: 0.0659473
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0677821
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0652203, upper bound: 0.0682734
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0700213
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
NS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0659973, upper bound: 0.0642977
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0667246
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
NS_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0659973, upper bound: 0.0669314
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0677821, upper bound: 0.0690997
NS_A1_B2_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
NS_A1_B2_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0668462
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
NS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0651608, upper bound: 0.0669817
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0667246, upper bound: 0.0692036
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0689735, upper bound: 0.0664895
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0706720, upper bound: 0.0682390
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0705570, upper bound: 0.0698083
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0718496, upper bound: 0.0714703
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
NS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0671227, upper bound: 0.0659473
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0677821
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0708310
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0708310
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0687243, upper bound: 0.0690855
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0708310
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0678722, upper bound: 0.0642977
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0700213, upper bound: 0.0667246
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0694993, upper bound: 0.0677252
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0712349, upper bound: 0.0697385
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0670587, upper bound: 0.0643809
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0690997, upper bound: 0.0668462
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0687034, upper bound: 0.0678018
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.32
Output dim: 9, lower bound: -0.0701166, upper bound: 0.0698457

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0063378, 0.0022185, -0.0085563, 0.0085563
1: -0.0092294, 0.0163209, -0.0092294, 0.0163209, -0.0255503, 0.0255503
2: -0.0004811, 0.0218100, -0.0004811, 0.0218100, -0.0222910, 0.0222910
3: -0.0092587, 0.0038791, -0.0092587, 0.0038791, -0.0131378, 0.0131378
4: -0.0089070, 0.0095615, -0.0089070, 0.0095615, -0.0184686, 0.0184686
5: -0.0074959, 0.0147249, -0.0074959, 0.0147249, -0.0222208, 0.0222208
6: -0.0076797, 0.0054675, -0.0076797, 0.0054675, -0.0131472, 0.0131472
7: -0.0119043, 0.0089149, -0.0119043, 0.0089149, -0.0208192, 0.0208192
8: -0.0069978, 0.0076339, -0.0069978, 0.0076339, -0.0146317, 0.0146317
9: 0.9586030, 1.0193583, 0.9586030, 1.0193583, -0.0607553, 0.0607553

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711679, upper bound: 0.0678337
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715980, upper bound: 0.0685241
time: 1.91 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0057862, 0.0029840, -0.0093219, 0.0080047
1: -0.0092294, 0.0163209, -0.0104647, 0.0095069, -0.0187363, 0.0267856
2: -0.0004811, 0.0218100, -0.0015217, 0.0174271, -0.0179082, 0.0233317
3: -0.0092587, 0.0038791, -0.0081027, -0.0017373, -0.0075214, 0.0119818
4: -0.0089070, 0.0095615, -0.0033245, 0.0098740, -0.0187810, 0.0128861
5: -0.0074959, 0.0147249, -0.0075501, 0.0078286, -0.0153245, 0.0222750
6: -0.0076797, 0.0054675, -0.0075438, 0.0001989, -0.0078786, 0.0130113
7: -0.0119043, 0.0089149, -0.0074399, 0.0084411, -0.0203454, 0.0163548
8: -0.0069978, 0.0076339, -0.0055036, 0.0015188, -0.0085166, 0.0131375
9: 0.9586030, 1.0193583, 0.9860020, 1.0210018, -0.0623988, 0.0333562

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711679, upper bound: 0.0678337
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0715980, upper bound: 0.0685241
time: 1.54 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057827, 0.0029787, -0.0062533, 0.0020826, -0.0078653, 0.0092320
1: -0.0104523, 0.0094820, -0.0090321, 0.0153494, -0.0258017, 0.0185141
2: -0.0015156, 0.0173915, -0.0003345, 0.0211109, -0.0226264, 0.0177260
3: -0.0080893, -0.0017420, -0.0090477, 0.0031019, -0.0111912, 0.0073057
4: -0.0033169, 0.0098681, -0.0081243, 0.0094109, -0.0127278, 0.0179924
5: -0.0075353, 0.0078206, -0.0071809, 0.0138042, -0.0213395, 0.0150014
6: -0.0075397, 0.0001936, -0.0075891, 0.0047445, -0.0122842, 0.0077827
7: -0.0074313, 0.0084280, -0.0112032, 0.0086332, -0.0160645, 0.0196312
8: -0.0054756, 0.0015148, -0.0066281, 0.0068659, -0.0123415, 0.0081428
9: 0.9860274, 1.0209832, 0.9621280, 1.0189878, -0.0329604, 0.0588552

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0607955, upper bound: 0.0615322
time: 1.54 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678950, upper bound: 0.0678950
time: 1.44 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0068715, 0.0029689, -0.0093067, 0.0090900
1: -0.0092294, 0.0163209, -0.0102067, 0.0208570, -0.0300864, 0.0265276
2: -0.0004811, 0.0218100, -0.0010219, 0.0253591, -0.0258402, 0.0228319
3: -0.0092587, 0.0038791, -0.0103941, 0.0071250, -0.0163837, 0.0142731
4: -0.0089070, 0.0095615, -0.0120788, 0.0100930, -0.0190000, 0.0216403
5: -0.0074959, 0.0147249, -0.0086981, 0.0185246, -0.0260205, 0.0234230
6: -0.0076797, 0.0054675, -0.0080465, 0.0084473, -0.0161270, 0.0135139
7: -0.0119043, 0.0089149, -0.0146922, 0.0099509, -0.0218552, 0.0236071
8: -0.0069978, 0.0076339, -0.0086411, 0.0108853, -0.0178831, 0.0162750
9: 0.9586030, 1.0193583, 0.9438275, 1.0209014, -0.0622984, 0.0755308

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711826, upper bound: 0.0701845
time: 2.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716086, upper bound: 0.0712505
time: 1.57 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0063068, 0.0034323, -0.0097701, 0.0085253
1: -0.0092294, 0.0163209, -0.0114994, 0.0137490, -0.0229784, 0.0278203
2: -0.0004811, 0.0218100, -0.0021608, 0.0209964, -0.0214774, 0.0239708
3: -0.0092587, 0.0038791, -0.0092192, 0.0010684, -0.0103271, 0.0130983
4: -0.0089070, 0.0095615, -0.0065257, 0.0104626, -0.0193696, 0.0160873
5: -0.0074959, 0.0147249, -0.0087947, 0.0117027, -0.0191987, 0.0235196
6: -0.0076797, 0.0054675, -0.0079002, 0.0031316, -0.0108113, 0.0133677
7: -0.0119043, 0.0089149, -0.0100647, 0.0095251, -0.0214294, 0.0189796
8: -0.0069978, 0.0076339, -0.0070877, 0.0047627, -0.0117605, 0.0147215
9: 0.9586030, 1.0193583, 0.9710250, 1.0226896, -0.0640866, 0.0483333

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0711826, upper bound: 0.0701845
time: 1.36 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716086, upper bound: 0.0712505
time: 1.59 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0057553, 0.0029259, -0.0068027, 0.0027954, -0.0085507, 0.0097286
1: -0.0103779, 0.0093090, -0.0102529, 0.0199642, -0.0303421, 0.0195619
2: -0.0014645, 0.0171811, -0.0010808, 0.0247977, -0.0262622, 0.0182618
3: -0.0080149, -0.0018038, -0.0102580, 0.0064167, -0.0144315, 0.0084542
4: -0.0032145, 0.0098155, -0.0113449, 0.0099968, -0.0132112, 0.0211604
5: -0.0074180, 0.0077136, -0.0083817, 0.0177200, -0.0251380, 0.0160953
6: -0.0075101, 0.0001176, -0.0079685, 0.0077707, -0.0152808, 0.0080860
7: -0.0073247, 0.0083315, -0.0139359, 0.0097390, -0.0170637, 0.0222674
8: -0.0053121, 0.0014537, -0.0083175, 0.0102355, -0.0155476, 0.0097711
9: 0.9863868, 1.0208461, 0.9468634, 1.0207930, -0.0344062, 0.0739828

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0583225, upper bound: 0.0584885
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0665718, upper bound: 0.0686112
time: 1.20 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057827, 0.0029787, -0.0068033, 0.0028023, -0.0085850, 0.0097820
1: -0.0104523, 0.0094820, -0.0100450, 0.0200137, -0.0304660, 0.0195270
2: -0.0015156, 0.0173915, -0.0008929, 0.0247913, -0.0263069, 0.0182844
3: -0.0080893, -0.0017420, -0.0102327, 0.0064552, -0.0145445, 0.0084907
4: -0.0033169, 0.0098681, -0.0113958, 0.0099598, -0.0132767, 0.0212640
5: -0.0075353, 0.0078206, -0.0084287, 0.0177222, -0.0252574, 0.0162493
6: -0.0075397, 0.0001936, -0.0079717, 0.0078176, -0.0153572, 0.0081653
7: -0.0074313, 0.0084280, -0.0140777, 0.0097164, -0.0171477, 0.0225057
8: -0.0054756, 0.0015148, -0.0083563, 0.0102282, -0.0157038, 0.0098711
9: 0.9860274, 1.0209832, 0.9468241, 1.0205849, -0.0345576, 0.0741591

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0607955, upper bound: 0.0628329
time: 1.51 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0678950, upper bound: 0.0703105
time: 1.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0063378, 0.0022185, -0.0090927, 0.0092855
1: -0.0102804, 0.0216467, -0.0092294, 0.0163209, -0.0266013, 0.0308761
2: -0.0013008, 0.0254146, -0.0004811, 0.0218100, -0.0231108, 0.0258956
3: -0.0103910, 0.0080113, -0.0092587, 0.0038791, -0.0142701, 0.0172700
4: -0.0132094, 0.0103480, -0.0089070, 0.0095615, -0.0227709, 0.0192550
5: -0.0090217, 0.0197614, -0.0074959, 0.0147249, -0.0237466, 0.0272573
6: -0.0080985, 0.0094561, -0.0076797, 0.0054675, -0.0135659, 0.0171358
7: -0.0155814, 0.0101845, -0.0119043, 0.0089149, -0.0244963, 0.0220888
8: -0.0084719, 0.0118298, -0.0069978, 0.0076339, -0.0161058, 0.0188276
9: 0.9400309, 1.0213395, 0.9586030, 1.0193583, -0.0793274, 0.0627365

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0695874, upper bound: 0.0673037
time: 1.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
time: 1.39 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0057862, 0.0029840, -0.0098582, 0.0087338
1: -0.0102804, 0.0216467, -0.0104647, 0.0095069, -0.0197873, 0.0321114
2: -0.0013008, 0.0254146, -0.0015217, 0.0174271, -0.0187279, 0.0269363
3: -0.0103910, 0.0080113, -0.0081027, -0.0017373, -0.0086537, 0.0161140
4: -0.0132094, 0.0103480, -0.0033245, 0.0098740, -0.0230834, 0.0136725
5: -0.0090217, 0.0197614, -0.0075501, 0.0078286, -0.0168503, 0.0273114
6: -0.0080985, 0.0094561, -0.0075438, 0.0001989, -0.0082973, 0.0169999
7: -0.0155814, 0.0101845, -0.0074399, 0.0084411, -0.0240225, 0.0176244
8: -0.0084719, 0.0118298, -0.0055036, 0.0015188, -0.0099907, 0.0173334
9: 0.9400309, 1.0213395, 0.9860020, 1.0210018, -0.0809709, 0.0353375

Time for backsubstitution: 1.95 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0695874, upper bound: 0.0673037
time: 2.63 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702529, upper bound: 0.0682461
time: 1.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

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

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0659973
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0677821
time: 1.86 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0068715, 0.0029689, -0.0098431, 0.0098191
1: -0.0102804, 0.0216467, -0.0102067, 0.0208570, -0.0311374, 0.0318534
2: -0.0013008, 0.0254146, -0.0010219, 0.0253591, -0.0266600, 0.0264365
3: -0.0103910, 0.0080113, -0.0103941, 0.0071250, -0.0175160, 0.0184054
4: -0.0132094, 0.0103480, -0.0120788, 0.0100930, -0.0233024, 0.0224268
5: -0.0090217, 0.0197614, -0.0086981, 0.0185246, -0.0275463, 0.0284594
6: -0.0080985, 0.0094561, -0.0080465, 0.0084473, -0.0165458, 0.0175026
7: -0.0155814, 0.0101845, -0.0146922, 0.0099509, -0.0255324, 0.0248768
8: -0.0084719, 0.0118298, -0.0086411, 0.0108853, -0.0193571, 0.0204709
9: 0.9400309, 1.0213395, 0.9438275, 1.0209014, -0.0808706, 0.0775120

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696368, upper bound: 0.0694705
time: 2.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702680, upper bound: 0.0707374
time: 3.36 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0063068, 0.0034323, -0.0103065, 0.0092544
1: -0.0102804, 0.0216467, -0.0114994, 0.0137490, -0.0240294, 0.0331461
2: -0.0013008, 0.0254146, -0.0021608, 0.0209964, -0.0222972, 0.0275754
3: -0.0103910, 0.0080113, -0.0092192, 0.0010684, -0.0114594, 0.0172305
4: -0.0132094, 0.0103480, -0.0065257, 0.0104626, -0.0236720, 0.0168737
5: -0.0090217, 0.0197614, -0.0087947, 0.0117027, -0.0207244, 0.0285561
6: -0.0080985, 0.0094561, -0.0079002, 0.0031316, -0.0112300, 0.0173563
7: -0.0155814, 0.0101845, -0.0100647, 0.0095251, -0.0251066, 0.0202492
8: -0.0084719, 0.0118298, -0.0070877, 0.0047627, -0.0132345, 0.0189174
9: 0.9400309, 1.0213395, 0.9710250, 1.0226896, -0.0826587, 0.0503145

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696368, upper bound: 0.0694705
time: 1.75 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702680, upper bound: 0.0707374
time: 1.40 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0062703, 0.0033860, -0.0068027, 0.0027954, -0.0090656, 0.0101887
1: -0.0113771, 0.0140831, -0.0102529, 0.0199642, -0.0313413, 0.0243359
2: -0.0022430, 0.0208014, -0.0010808, 0.0247977, -0.0270407, 0.0218822
3: -0.0090898, 0.0017325, -0.0102580, 0.0064167, -0.0155064, 0.0119905
4: -0.0074147, 0.0105502, -0.0113449, 0.0099968, -0.0174114, 0.0218951
5: -0.0088267, 0.0126651, -0.0083817, 0.0177200, -0.0265467, 0.0210469
6: -0.0078815, 0.0039089, -0.0079685, 0.0077707, -0.0156522, 0.0118773
7: -0.0107232, 0.0095089, -0.0139359, 0.0097390, -0.0204621, 0.0234448
8: -0.0066794, 0.0054749, -0.0083175, 0.0102355, -0.0169149, 0.0137923
9: 0.9681052, 1.0227323, 0.9468634, 1.0207930, -0.0526878, 0.0758689

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0649845, upper bound: 0.0678421
time: 1.60 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0647666, upper bound: 0.0678531
time: 2.80 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0063016, 0.0034514, -0.0068033, 0.0028023, -0.0091039, 0.0102547
1: -0.0114549, 0.0142223, -0.0100450, 0.0200137, -0.0314686, 0.0242673
2: -0.0022969, 0.0209819, -0.0008929, 0.0247913, -0.0270882, 0.0218748
3: -0.0091621, 0.0018110, -0.0102327, 0.0064552, -0.0156174, 0.0120437
4: -0.0075178, 0.0106052, -0.0113958, 0.0099598, -0.0174777, 0.0220011
5: -0.0089495, 0.0127693, -0.0084287, 0.0177222, -0.0266717, 0.0211980
6: -0.0079132, 0.0039928, -0.0079717, 0.0078176, -0.0157308, 0.0119645
7: -0.0108393, 0.0096099, -0.0140777, 0.0097164, -0.0205557, 0.0236876
8: -0.0068318, 0.0055387, -0.0083563, 0.0102282, -0.0170600, 0.0138950
9: 0.9677775, 1.0228754, 0.9468241, 1.0205849, -0.0528075, 0.0760513

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0678722
time: 1.46 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0700213
time: 1.55 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0068742, 0.0029476, -0.0092855, 0.0090927
1: -0.0092294, 0.0163209, -0.0102804, 0.0216467, -0.0308761, 0.0266013
2: -0.0004811, 0.0218100, -0.0013008, 0.0254146, -0.0258956, 0.0231108
3: -0.0092587, 0.0038791, -0.0103910, 0.0080113, -0.0172700, 0.0142701
4: -0.0089070, 0.0095615, -0.0132094, 0.0103480, -0.0192550, 0.0227709
5: -0.0074959, 0.0147249, -0.0090217, 0.0197614, -0.0272573, 0.0237466
6: -0.0076797, 0.0054675, -0.0080985, 0.0094561, -0.0171358, 0.0135659
7: -0.0119043, 0.0089149, -0.0155814, 0.0101845, -0.0220888, 0.0244963
8: -0.0069978, 0.0076339, -0.0084719, 0.0118298, -0.0188276, 0.0161058
9: 0.9586030, 1.0193583, 0.9400309, 1.0213395, -0.0627365, 0.0793274

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701892, upper bound: 0.0662044
time: 1.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708206, upper bound: 0.0669953
time: 1.53 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0063058, 0.0034586, -0.0097964, 0.0085243
1: -0.0092294, 0.0163209, -0.0114675, 0.0142369, -0.0234663, 0.0277884
2: -0.0004811, 0.0218100, -0.0023040, 0.0210116, -0.0214926, 0.0241140
3: -0.0092587, 0.0038791, -0.0091753, 0.0018162, -0.0110749, 0.0130543
4: -0.0089070, 0.0095615, -0.0075261, 0.0106122, -0.0195192, 0.0170876
5: -0.0074959, 0.0147249, -0.0089659, 0.0127776, -0.0202735, 0.0236908
6: -0.0076797, 0.0054675, -0.0079179, 0.0039991, -0.0116788, 0.0133853
7: -0.0119043, 0.0089149, -0.0108489, 0.0096242, -0.0215285, 0.0197637
8: -0.0069978, 0.0076339, -0.0068596, 0.0055431, -0.0125409, 0.0144935
9: 0.9586030, 1.0193583, 0.9677531, 1.0228958, -0.0642928, 0.0516052

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701892, upper bound: 0.0662044
time: 1.56 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708206, upper bound: 0.0669953
time: 1.58 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057827, 0.0029787, -0.0067967, 0.0027754, -0.0085581, 0.0097754
1: -0.0104523, 0.0094820, -0.0101060, 0.0207606, -0.0312128, 0.0195880
2: -0.0015156, 0.0173915, -0.0011671, 0.0248293, -0.0263449, 0.0185586
3: -0.0080893, -0.0017420, -0.0102120, 0.0073227, -0.0154119, 0.0084700
4: -0.0033169, 0.0098681, -0.0125204, 0.0102099, -0.0135268, 0.0223885
5: -0.0075353, 0.0078206, -0.0087339, 0.0189557, -0.0264910, 0.0165545
6: -0.0075397, 0.0001936, -0.0080189, 0.0088134, -0.0163530, 0.0082125
7: -0.0074313, 0.0084280, -0.0149673, 0.0099363, -0.0173675, 0.0233952
8: -0.0054756, 0.0015148, -0.0081553, 0.0111491, -0.0166247, 0.0096700
9: 0.9860274, 1.0209832, 0.9430635, 1.0210063, -0.0349790, 0.0779197

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0597017, upper bound: 0.0575760
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0674339, upper bound: 0.0663575
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0074868, 0.0047480, -0.0110858, 0.0097053
1: -0.0092294, 0.0163209, -0.0113465, 0.0271213, -0.0363507, 0.0276674
2: -0.0004811, 0.0218100, -0.0018529, 0.0303596, -0.0308407, 0.0236629
3: -0.0092587, 0.0038791, -0.0116776, 0.0114324, -0.0206911, 0.0155567
4: -0.0089070, 0.0095615, -0.0164778, 0.0110528, -0.0199598, 0.0260394
5: -0.0074959, 0.0147249, -0.0103079, 0.0239427, -0.0314386, 0.0250328
6: -0.0076797, 0.0054675, -0.0084978, 0.0129430, -0.0206227, 0.0139652
7: -0.0119043, 0.0089149, -0.0198323, 0.0112676, -0.0231718, 0.0287472
8: -0.0069978, 0.0076339, -0.0103719, 0.0157998, -0.0227976, 0.0180058
9: 0.9586030, 1.0193583, 0.9233138, 1.0229756, -0.0643725, 0.0960445

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701954, upper bound: 0.0685612
time: 1.63 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708314, upper bound: 0.0696418
time: 2.14 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0063378, 0.0022185, -0.0068255, 0.0039014, -0.0102392, 0.0090440
1: -0.0092294, 0.0163209, -0.0124919, 0.0187966, -0.0280260, 0.0288128
2: -0.0004811, 0.0218100, -0.0029244, 0.0245749, -0.0250560, 0.0247344
3: -0.0092587, 0.0038791, -0.0103170, 0.0050977, -0.0143564, 0.0141961
4: -0.0089070, 0.0095615, -0.0107890, 0.0111772, -0.0200842, 0.0203506
5: -0.0074959, 0.0147249, -0.0101817, 0.0166798, -0.0241757, 0.0249066
6: -0.0076797, 0.0054675, -0.0082751, 0.0070263, -0.0147060, 0.0137426
7: -0.0119043, 0.0089149, -0.0136281, 0.0106762, -0.0225805, 0.0225430
8: -0.0069978, 0.0076339, -0.0085291, 0.0088583, -0.0158561, 0.0161630
9: 0.9586030, 1.0193583, 0.9525756, 1.0245186, -0.0659156, 0.0667827

Time for backsubstitution: 1.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701954, upper bound: 0.0685612
time: 2.40 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0708314, upper bound: 0.0696418
time: 1.35 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0057827, 0.0029787, -0.0074171, 0.0045568, -0.0103395, 0.0103957
1: -0.0104523, 0.0094820, -0.0112120, 0.0262467, -0.0366989, 0.0206940
2: -0.0015156, 0.0173915, -0.0017278, 0.0298322, -0.0313478, 0.0191193
3: -0.0080893, -0.0017420, -0.0115610, 0.0107739, -0.0188632, 0.0098189
4: -0.0033169, 0.0098681, -0.0158163, 0.0109059, -0.0142227, 0.0256845
5: -0.0075353, 0.0078206, -0.0100568, 0.0231622, -0.0306975, 0.0178774
6: -0.0075397, 0.0001936, -0.0084336, 0.0122971, -0.0198368, 0.0086271
7: -0.0074313, 0.0084280, -0.0191312, 0.0110580, -0.0184893, 0.0275591
8: -0.0054756, 0.0015148, -0.0101620, 0.0151106, -0.0205862, 0.0116768
9: 0.9860274, 1.0209832, 0.9263134, 1.0226855, -0.0366582, 0.0946698

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 159

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0597057, upper bound: 0.0588564
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0674339, upper bound: 0.0687187
time: 1.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

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

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696284, upper bound: 0.0685872
time: 2.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702553, upper bound: 0.0698043
time: 3.09 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068742, 0.0029476, -0.0068255, 0.0039014, -0.0107756, 0.0097732
1: -0.0102804, 0.0216467, -0.0124919, 0.0187966, -0.0290770, 0.0341386
2: -0.0013008, 0.0254146, -0.0029244, 0.0245749, -0.0258757, 0.0283390
3: -0.0103910, 0.0080113, -0.0103170, 0.0050977, -0.0154887, 0.0183283
4: -0.0132094, 0.0103480, -0.0107890, 0.0111772, -0.0243866, 0.0211370
5: -0.0090217, 0.0197614, -0.0101817, 0.0166798, -0.0257015, 0.0299430
6: -0.0080985, 0.0094561, -0.0082751, 0.0070263, -0.0151248, 0.0177312
7: -0.0155814, 0.0101845, -0.0136281, 0.0106762, -0.0262577, 0.0238126
8: -0.0084719, 0.0118298, -0.0085291, 0.0088583, -0.0173302, 0.0203589
9: 0.9400309, 1.0213395, 0.9525756, 1.0245186, -0.0844877, 0.0687640

Time for backsubstitution: 2.09 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696284, upper bound: 0.0685872
time: 2.48 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0702553, upper bound: 0.0698043
time: 1.46 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0063016, 0.0034514, -0.0074171, 0.0045568, -0.0108584, 0.0108685
1: -0.0114549, 0.0142223, -0.0112120, 0.0262467, -0.0377015, 0.0254343
2: -0.0022969, 0.0209819, -0.0017278, 0.0298322, -0.0321291, 0.0227097
3: -0.0091621, 0.0018110, -0.0115610, 0.0107739, -0.0199360, 0.0133720
4: -0.0075178, 0.0106052, -0.0158163, 0.0109059, -0.0184237, 0.0264216
5: -0.0089495, 0.0127693, -0.0100568, 0.0231622, -0.0321117, 0.0228261
6: -0.0079132, 0.0039928, -0.0084336, 0.0122971, -0.0202103, 0.0124264
7: -0.0108393, 0.0096099, -0.0191312, 0.0110580, -0.0218973, 0.0287410
8: -0.0068318, 0.0055387, -0.0101620, 0.0151106, -0.0219424, 0.0157007
9: 0.9677775, 1.0228754, 0.9263134, 1.0226855, -0.0549080, 0.0965620

Time for backsubstitution: 2.11 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0670587
time: 1.88 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0642977, upper bound: 0.0692036
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0063378, 0.0022185, -0.0090900, 0.0093067
1: -0.0102067, 0.0208570, -0.0092294, 0.0163209, -0.0265276, 0.0300864
2: -0.0010219, 0.0253591, -0.0004811, 0.0218100, -0.0228319, 0.0258402
3: -0.0103941, 0.0071250, -0.0092587, 0.0038791, -0.0142731, 0.0163837
4: -0.0120788, 0.0100930, -0.0089070, 0.0095615, -0.0216403, 0.0190000
5: -0.0086981, 0.0185246, -0.0074959, 0.0147249, -0.0234230, 0.0260205
6: -0.0080465, 0.0084473, -0.0076797, 0.0054675, -0.0135139, 0.0161270
7: -0.0146922, 0.0099509, -0.0119043, 0.0089149, -0.0236071, 0.0218552
8: -0.0086411, 0.0108853, -0.0069978, 0.0076339, -0.0162750, 0.0178831
9: 0.9438275, 1.0209014, 0.9586030, 1.0193583, -0.0755308, 0.0622984

Time for backsubstitution: 1.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726350, upper bound: 0.0678337
time: 1.61 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
time: 1.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0057862, 0.0029840, -0.0098555, 0.0087551
1: -0.0102067, 0.0208570, -0.0104647, 0.0095069, -0.0197137, 0.0313217
2: -0.0010219, 0.0253591, -0.0015217, 0.0174271, -0.0184491, 0.0268808
3: -0.0103941, 0.0071250, -0.0081027, -0.0017373, -0.0086568, 0.0152277
4: -0.0120788, 0.0100930, -0.0033245, 0.0098740, -0.0219528, 0.0134175
5: -0.0086981, 0.0185246, -0.0075501, 0.0078286, -0.0165267, 0.0260747
6: -0.0080465, 0.0084473, -0.0075438, 0.0001989, -0.0082453, 0.0159911
7: -0.0146922, 0.0099509, -0.0074399, 0.0084411, -0.0231333, 0.0173908
8: -0.0086411, 0.0108853, -0.0055036, 0.0015188, -0.0101599, 0.0163888
9: 0.9438275, 1.0209014, 0.9860020, 1.0210018, -0.0771743, 0.0348994

Time for backsubstitution: 2.06 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0726350, upper bound: 0.0678337
time: 2.59 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0731205, upper bound: 0.0685241
time: 1.51 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

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

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0625283, upper bound: 0.0592759
time: 1.32 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0685221, upper bound: 0.0661470
time: 1.39 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

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

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0653553, upper bound: 0.0622759
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0703105, upper bound: 0.0678950
time: 1.43 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0068715, 0.0029689, -0.0098404, 0.0098404
1: -0.0102067, 0.0208570, -0.0102067, 0.0208570, -0.0310638, 0.0310638
2: -0.0010219, 0.0253591, -0.0010219, 0.0253591, -0.0263811, 0.0263811
3: -0.0103941, 0.0071250, -0.0103941, 0.0071250, -0.0175190, 0.0175190
4: -0.0120788, 0.0100930, -0.0120788, 0.0100930, -0.0221718, 0.0221718
5: -0.0086981, 0.0185246, -0.0086981, 0.0185246, -0.0272227, 0.0272227
6: -0.0080465, 0.0084473, -0.0080465, 0.0084473, -0.0164938, 0.0164938
7: -0.0146922, 0.0099509, -0.0146922, 0.0099509, -0.0246432, 0.0246432
8: -0.0086411, 0.0108853, -0.0086411, 0.0108853, -0.0195264, 0.0195264
9: 0.9438275, 1.0209014, 0.9438275, 1.0209014, -0.0770739, 0.0770739

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0746091, upper bound: 0.0709569
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749766, upper bound: 0.0717021
time: 4.38 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0063068, 0.0034323, -0.0103038, 0.0092757
1: -0.0102067, 0.0208570, -0.0114994, 0.0137490, -0.0239557, 0.0323565
2: -0.0010219, 0.0253591, -0.0021608, 0.0209964, -0.0220183, 0.0275200
3: -0.0103941, 0.0071250, -0.0092192, 0.0010684, -0.0114624, 0.0163442
4: -0.0120788, 0.0100930, -0.0065257, 0.0104626, -0.0225414, 0.0166187
5: -0.0086981, 0.0185246, -0.0087947, 0.0117027, -0.0204008, 0.0273193
6: -0.0080465, 0.0084473, -0.0079002, 0.0031316, -0.0111780, 0.0163475
7: -0.0146922, 0.0099509, -0.0100647, 0.0095251, -0.0242174, 0.0200156
8: -0.0086411, 0.0108853, -0.0070877, 0.0047627, -0.0134038, 0.0179729
9: 0.9438275, 1.0209014, 0.9710250, 1.0226896, -0.0788621, 0.0498765

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0746091, upper bound: 0.0709569
time: 1.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0749766, upper bound: 0.0717021
time: 1.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

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

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0651576, upper bound: 0.0642792
time: 1.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0701216, upper bound: 0.0694426
time: 1.82 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

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

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0663093, upper bound: 0.0660015
time: 1.37 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0714970, upper bound: 0.0711025
time: 2.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0063378, 0.0022185, -0.0097053, 0.0110858
1: -0.0113465, 0.0271213, -0.0092294, 0.0163209, -0.0276674, 0.0363507
2: -0.0018529, 0.0303596, -0.0004811, 0.0218100, -0.0236629, 0.0308407
3: -0.0116776, 0.0114324, -0.0092587, 0.0038791, -0.0155567, 0.0206911
4: -0.0164778, 0.0110528, -0.0089070, 0.0095615, -0.0260394, 0.0199598
5: -0.0103079, 0.0239427, -0.0074959, 0.0147249, -0.0250328, 0.0314386
6: -0.0084978, 0.0129430, -0.0076797, 0.0054675, -0.0139652, 0.0206227
7: -0.0198323, 0.0112676, -0.0119043, 0.0089149, -0.0287472, 0.0231718
8: -0.0103719, 0.0157998, -0.0069978, 0.0076339, -0.0180058, 0.0227976
9: 0.9233138, 1.0229756, 0.9586030, 1.0193583, -0.0960445, 0.0643725

Time for backsubstitution: 2.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0709358, upper bound: 0.0673037
time: 1.85 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0682461
time: 1.19 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0057862, 0.0029840, -0.0104708, 0.0105341
1: -0.0113465, 0.0271213, -0.0104647, 0.0095069, -0.0208534, 0.0375860
2: -0.0018529, 0.0303596, -0.0015217, 0.0174271, -0.0192801, 0.0318813
3: -0.0116776, 0.0114324, -0.0081027, -0.0017373, -0.0099403, 0.0195351
4: -0.0164778, 0.0110528, -0.0033245, 0.0098740, -0.0263518, 0.0143773
5: -0.0103079, 0.0239427, -0.0075501, 0.0078286, -0.0181365, 0.0314927
6: -0.0084978, 0.0129430, -0.0075438, 0.0001989, -0.0086966, 0.0204868
7: -0.0198323, 0.0112676, -0.0074399, 0.0084411, -0.0282734, 0.0187074
8: -0.0103719, 0.0157998, -0.0055036, 0.0015188, -0.0118907, 0.0213033
9: 0.9233138, 1.0229756, 0.9860020, 1.0210018, -0.0976880, 0.0369735

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0709358, upper bound: 0.0673037
time: 1.65 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716066, upper bound: 0.0682461
time: 2.40 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068212, 0.0038926, -0.0062533, 0.0020826, -0.0089039, 0.0101460
1: -0.0124797, 0.0187820, -0.0090321, 0.0153494, -0.0278291, 0.0278141
2: -0.0029166, 0.0245513, -0.0003345, 0.0211109, -0.0240275, 0.0248858
3: -0.0103052, 0.0050931, -0.0090477, 0.0031019, -0.0134072, 0.0141408
4: -0.0107809, 0.0111696, -0.0081243, 0.0094109, -0.0201918, 0.0192939
5: -0.0101648, 0.0166720, -0.0071809, 0.0138042, -0.0239690, 0.0238529
6: -0.0082704, 0.0070202, -0.0075891, 0.0047445, -0.0130149, 0.0146093
7: -0.0136183, 0.0106617, -0.0112032, 0.0086332, -0.0222515, 0.0218650
8: -0.0085036, 0.0088543, -0.0066281, 0.0068659, -0.0153696, 0.0154823
9: 0.9525976, 1.0244980, 0.9621280, 1.0189878, -0.0663902, 0.0623699

Time for backsubstitution: 1.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0669314, upper bound: 0.0659973
time: 1.29 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0669314, upper bound: 0.0677821
time: 1.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0068715, 0.0029689, -0.0104557, 0.0116195
1: -0.0113465, 0.0271213, -0.0102067, 0.0208570, -0.0322035, 0.0373280
2: -0.0018529, 0.0303596, -0.0010219, 0.0253591, -0.0272121, 0.0313816
3: -0.0116776, 0.0114324, -0.0103941, 0.0071250, -0.0188026, 0.0218265
4: -0.0164778, 0.0110528, -0.0120788, 0.0100930, -0.0265708, 0.0231316
5: -0.0103079, 0.0239427, -0.0086981, 0.0185246, -0.0288325, 0.0326407
6: -0.0084978, 0.0129430, -0.0080465, 0.0084473, -0.0169450, 0.0209894
7: -0.0198323, 0.0112676, -0.0146922, 0.0099509, -0.0297833, 0.0259598
8: -0.0103719, 0.0157998, -0.0086411, 0.0108853, -0.0212571, 0.0244409
9: 0.9233138, 1.0229756, 0.9438275, 1.0209014, -0.0975876, 0.0791481

Time for backsubstitution: 2.31 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728721, upper bound: 0.0702888
time: 1.79 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0712693
time: 1.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0074868, 0.0047480, -0.0063068, 0.0034323, -0.0109191, 0.0110547
1: -0.0113465, 0.0271213, -0.0114994, 0.0137490, -0.0250955, 0.0386207
2: -0.0018529, 0.0303596, -0.0021608, 0.0209964, -0.0228493, 0.0325205
3: -0.0116776, 0.0114324, -0.0092192, 0.0010684, -0.0127460, 0.0206516
4: -0.0164778, 0.0110528, -0.0065257, 0.0104626, -0.0269404, 0.0175785
5: -0.0103079, 0.0239427, -0.0087947, 0.0117027, -0.0220106, 0.0327374
6: -0.0084978, 0.0129430, -0.0079002, 0.0031316, -0.0116293, 0.0208432
7: -0.0198323, 0.0112676, -0.0100647, 0.0095251, -0.0293575, 0.0213322
8: -0.0103719, 0.0157998, -0.0070877, 0.0047627, -0.0151345, 0.0228874
9: 0.9233138, 1.0229756, 0.9710250, 1.0226896, -0.0993758, 0.0519506

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0728721, upper bound: 0.0702888
time: 3.87 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0733960, upper bound: 0.0712693
time: 1.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0067893, 0.0038216, -0.0068027, 0.0027954, -0.0095846, 0.0106243
1: -0.0124012, 0.0186471, -0.0102529, 0.0199642, -0.0323654, 0.0289000
2: -0.0028595, 0.0243973, -0.0010808, 0.0247977, -0.0276572, 0.0254780
3: -0.0102383, 0.0050277, -0.0102580, 0.0064167, -0.0166550, 0.0152857
4: -0.0106879, 0.0111115, -0.0113449, 0.0099968, -0.0206846, 0.0224564
5: -0.0100384, 0.0165843, -0.0083817, 0.0177200, -0.0277584, 0.0249661
6: -0.0082378, 0.0069466, -0.0079685, 0.0077707, -0.0160085, 0.0149151
7: -0.0135073, 0.0105585, -0.0139359, 0.0097390, -0.0232462, 0.0244944
8: -0.0083664, 0.0088009, -0.0083175, 0.0102355, -0.0186020, 0.0171184
9: 0.9528741, 1.0243510, 0.9468634, 1.0207930, -0.0679188, 0.0774876

Time for backsubstitution: 2.04 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0684918, upper bound: 0.0686384
time: 1.36 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0681996, upper bound: 0.0686591
time: 1.61 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -0.0068212, 0.0038926, -0.0068033, 0.0028023, -0.0096236, 0.0106959
1: -0.0124797, 0.0187820, -0.0100450, 0.0200137, -0.0324935, 0.0288270
2: -0.0029166, 0.0245513, -0.0008929, 0.0247913, -0.0277080, 0.0254442
3: -0.0103052, 0.0050931, -0.0102327, 0.0064552, -0.0167605, 0.0153258
4: -0.0107809, 0.0111696, -0.0113958, 0.0099598, -0.0207407, 0.0225655
5: -0.0101648, 0.0166720, -0.0084287, 0.0177222, -0.0278869, 0.0251007
6: -0.0082704, 0.0070202, -0.0079717, 0.0078176, -0.0160880, 0.0149919
7: -0.0136183, 0.0106617, -0.0140777, 0.0097164, -0.0233347, 0.0247394
8: -0.0085036, 0.0088543, -0.0083563, 0.0102282, -0.0187318, 0.0172106
9: 0.9525976, 1.0244980, 0.9468241, 1.0205849, -0.0679873, 0.0776739

Time for backsubstitution: 2.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680717, upper bound: 0.0690937
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0680717, upper bound: 0.0708310
time: 2.19 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0068742, 0.0029476, -0.0098191, 0.0098431
1: -0.0102067, 0.0208570, -0.0102804, 0.0216467, -0.0318534, 0.0311374
2: -0.0010219, 0.0253591, -0.0013008, 0.0254146, -0.0264365, 0.0266600
3: -0.0103941, 0.0071250, -0.0103910, 0.0080113, -0.0184054, 0.0175160
4: -0.0120788, 0.0100930, -0.0132094, 0.0103480, -0.0224268, 0.0233024
5: -0.0086981, 0.0185246, -0.0090217, 0.0197614, -0.0284594, 0.0275463
6: -0.0080465, 0.0084473, -0.0080985, 0.0094561, -0.0175026, 0.0165458
7: -0.0146922, 0.0099509, -0.0155814, 0.0101845, -0.0248768, 0.0255324
8: -0.0086411, 0.0108853, -0.0084719, 0.0118298, -0.0204709, 0.0193571
9: 0.9438275, 1.0209014, 0.9400309, 1.0213395, -0.0775120, 0.0808706

Time for backsubstitution: 2.13 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716693, upper bound: 0.0662044
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0669953
time: 1.21 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0063058, 0.0034586, -0.0103301, 0.0092747
1: -0.0102067, 0.0208570, -0.0114675, 0.0142369, -0.0244436, 0.0323246
2: -0.0010219, 0.0253591, -0.0023040, 0.0210116, -0.0220335, 0.0276632
3: -0.0103941, 0.0071250, -0.0091753, 0.0018162, -0.0122103, 0.0163002
4: -0.0120788, 0.0100930, -0.0075261, 0.0106122, -0.0226910, 0.0176191
5: -0.0086981, 0.0185246, -0.0089659, 0.0127776, -0.0214757, 0.0274906
6: -0.0080465, 0.0084473, -0.0079179, 0.0039991, -0.0120456, 0.0163652
7: -0.0146922, 0.0099509, -0.0108489, 0.0096242, -0.0243165, 0.0207998
8: -0.0086411, 0.0108853, -0.0068596, 0.0055431, -0.0141842, 0.0177449
9: 0.9438275, 1.0209014, 0.9677531, 1.0228958, -0.0790683, 0.0531484

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0716693, upper bound: 0.0662044
time: 1.76 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0723091, upper bound: 0.0669953
time: 1.54 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -0.0062737, 0.0033670, -0.0067617, 0.0026812, -0.0089549, 0.0101287
1: -0.0114136, 0.0136106, -0.0102150, 0.0205190, -0.0319327, 0.0238256
2: -0.0021010, 0.0207943, -0.0013070, 0.0246151, -0.0267161, 0.0221013
3: -0.0091385, 0.0009955, -0.0101515, 0.0071523, -0.0162908, 0.0111470
4: -0.0064219, 0.0104024, -0.0123161, 0.0102000, -0.0166219, 0.0227185
5: -0.0086612, 0.0115995, -0.0085835, 0.0187829, -0.0274441, 0.0201830
6: -0.0078658, 0.0030495, -0.0079812, 0.0086179, -0.0164837, 0.0110307
7: -0.0099434, 0.0094149, -0.0147086, 0.0098542, -0.0197976, 0.0241235
8: -0.0069143, 0.0047044, -0.0079811, 0.0110159, -0.0179302, 0.0126855
9: 0.9713435, 1.0225322, 0.9435735, 1.0210730, -0.0497295, 0.0789587

Time for backsubstitution: 2.15 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0675880, upper bound: 0.0638226
time: 1.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0673898, upper bound: 0.0638461
time: 1.80 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -0.0063028, 0.0034258, -0.0067967, 0.0027754, -0.0090782, 0.0102225
1: -0.0114875, 0.0137353, -0.0101060, 0.0207606, -0.0322481, 0.0238413
2: -0.0021539, 0.0209670, -0.0011671, 0.0248293, -0.0269833, 0.0221341
3: -0.0092069, 0.0010635, -0.0102120, 0.0073227, -0.0165295, 0.0112755
4: -0.0065176, 0.0104560, -0.0125204, 0.0102099, -0.0167276, 0.0229764
5: -0.0087792, 0.0116947, -0.0087339, 0.0189557, -0.0277349, 0.0204286
6: -0.0078959, 0.0031255, -0.0080189, 0.0088134, -0.0167093, 0.0111444
7: -0.0100550, 0.0095116, -0.0149673, 0.0099363, -0.0199912, 0.0244788
8: -0.0070610, 0.0047586, -0.0081553, 0.0111491, -0.0182101, 0.0129139
9: 0.9710485, 1.0226704, 0.9430635, 1.0210063, -0.0499578, 0.0796069

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 12

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 107

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.0639245, upper bound: 0.0592931
time: 1.59 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0696636, upper bound: 0.0663575
time: 1.47 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

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

Time for backsubstitution: 2.10 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735100, upper bound: 0.0692415
time: 1.63 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740425, upper bound: 0.0699814
time: 1.57 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -0.0068715, 0.0029689, -0.0068255, 0.0039014, -0.0107729, 0.0097944
1: -0.0102067, 0.0208570, -0.0124919, 0.0187966, -0.0290034, 0.0333490
2: -0.0010219, 0.0253591, -0.0029244, 0.0245749, -0.0255968, 0.0282836
3: -0.0103941, 0.0071250, -0.0103170, 0.0050977, -0.0154918, 0.0174420
4: -0.0120788, 0.0100930, -0.0107890, 0.0111772, -0.0232560, 0.0208820
5: -0.0086981, 0.0185246, -0.0101817, 0.0166798, -0.0253779, 0.0287063
6: -0.0080465, 0.0084473, -0.0082751, 0.0070263, -0.0150728, 0.0167224
7: -0.0146922, 0.0099509, -0.0136281, 0.0106762, -0.0253685, 0.0235790
8: -0.0086411, 0.0108853, -0.0085291, 0.0088583, -0.0174995, 0.0194144
9: 0.9438275, 1.0209014, 0.9525756, 1.0245186, -0.0806911, 0.0683259

Time for backsubstitution: 2.12 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 107
type: A, layer: 1, pos: 104
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 159
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0735100, upper bound: 0.0692415
time: 1.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0740425, upper bound: 0.0699814
time: 3.09 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -0.0062737, 0.0033670, -0.0073905, 0.0045264, -0.0108001, 0.0107575
1: -0.0114136, 0.0136106, -0.0113725, 0.0259691, -0.0373827, 0.0249830
2: -0.0021010, 0.0207943, -0.0019008, 0.0297259, -0.0318269, 0.0226950
3: -0.0091385, 0.0009955, -0.0115423, 0.0105648, -0.0197033, 0.0125377
4: -0.0064219, 0.0104024, -0.0155925, 0.0109074, -0.0173293, 0.0259949
5: -0.0086612, 0.0115995, -0.0099504, 0.0229558, -0.0316170, 0.0215499
6: -0.0078658, 0.0030495, -0.0084055, 0.0121023, -0.0199681, 0.0114550
7: -0.0099434, 0.0094149, -0.0188827, 0.0110076, -0.0209510, 0.0282976
8: -0.0069143, 0.0047044, -0.0100872, 0.0149449, -0.0218592, 0.0147916
9: 0.9713435, 1.0225322, 0.9270397, 1.0228194, -0.0514759, 0.0954925

Time for backsubstitution: 2.15 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 9.59 + 591.64 = 601.23 seconds
