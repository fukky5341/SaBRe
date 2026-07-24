## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.01370172


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=3, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041471, -0.0009940, -0.0041471, -0.0009940, -0.0031530, 0.0031530)
1: (0.0173226, 0.0325174, 0.0173226, 0.0325174, -0.0101167, 0.0101167)
2: (0.0202262, 0.0305933, 0.0202262, 0.0305933, -0.0072442, 0.0072442)
3: (0.0058043, 0.0175526, 0.0058043, 0.0175526, -0.0091576, 0.0091576)
4: (-0.0184720, -0.0070115, -0.0184720, -0.0070115, -0.0094954, 0.0094954)
5: (0.0121548, 0.0266211, 0.0121548, 0.0266211, -0.0119455, 0.0119455)
6: (0.0045391, 0.0154581, 0.0045391, 0.0154581, -0.0090864, 0.0090864)
7: (-0.0230260, -0.0111398, -0.0230260, -0.0111398, -0.0103597, 0.0103597)
8: (0.0072907, 0.0191312, 0.0072907, 0.0191312, -0.0092412, 0.0092412)
9: (0.8989177, 0.9501236, 0.8989177, 0.9501236, -0.0364418, 0.0364418)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.58 = 2.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.0224800, upper bound: 0.0224800

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 187
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 187

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200999, upper bound: 0.0213068
time: 0.69 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0213067, upper bound: 0.0213068
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.54 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 9, lower bound: -0.0200999, upper bound: 0.0213068
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.54
Output dim: 9, lower bound: -0.0213067, upper bound: 0.0213068

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041215, -0.0010920, -0.0041404, -0.0010211, -0.0031004, 0.0030484
1: 0.0174430, 0.0316575, 0.0173549, 0.0322729, -0.0095909, 0.0088157
2: 0.0202439, 0.0299775, 0.0202308, 0.0304223, -0.0069097, 0.0063171
3: 0.0058958, 0.0167632, 0.0058286, 0.0173265, -0.0086230, 0.0080289
4: -0.0176357, -0.0070568, -0.0182286, -0.0070233, -0.0084825, 0.0090393
5: 0.0122690, 0.0256602, 0.0121851, 0.0263461, -0.0113779, 0.0107145
6: 0.0046071, 0.0146955, 0.0045570, 0.0152374, -0.0086241, 0.0081071
7: -0.0222719, -0.0112348, -0.0228129, -0.0111649, -0.0095019, 0.0099558
8: 0.0074112, 0.0183907, 0.0073227, 0.0189189, -0.0087118, 0.0081229
9: 0.9025575, 0.9498876, 0.8999405, 0.9500608, -0.0313094, 0.0342211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200999, upper bound: 0.0200999
time: 0.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0200999, upper bound: 0.0213068
time: 0.70 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0042008, -0.0009724, -0.0041388, -0.0010249, -0.0031759, 0.0031664
1: 0.0164073, 0.0319212, 0.0173619, 0.0323092, -0.0111948, 0.0091655
2: 0.0196588, 0.0302397, 0.0202314, 0.0304388, -0.0078755, 0.0067412
3: 0.0050006, 0.0170262, 0.0058335, 0.0173689, -0.0097518, 0.0083886
4: -0.0179647, -0.0061915, -0.0182924, -0.0070253, -0.0088136, 0.0100128
5: 0.0111131, 0.0260109, 0.0121912, 0.0264037, -0.0126483, 0.0111100
6: 0.0037587, 0.0149839, 0.0045605, 0.0152882, -0.0095784, 0.0084262
7: -0.0225591, -0.0102905, -0.0228565, -0.0111704, -0.0097751, 0.0109033
8: 0.0065277, 0.0186232, 0.0073293, 0.0189565, -0.0098599, 0.0084866
9: 0.9013567, 0.9535970, 0.8997612, 0.9500495, -0.0328716, 0.0395241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=3, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 187
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 187

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0213068, upper bound: 0.0200999
time: 0.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0213068, upper bound: 0.0213068
time: 0.77 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.83 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 9, lower bound: -0.0200999, upper bound: 0.0200999
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 9, lower bound: -0.0200999, upper bound: 0.0213068
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 9, lower bound: -0.0213068, upper bound: 0.0200999
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.83
Output dim: 9, lower bound: -0.0213068, upper bound: 0.0213068

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041215, -0.0010920, -0.0041215, -0.0010920, -0.0030296, 0.0030296
1: 0.0174430, 0.0316575, 0.0174430, 0.0316575, -0.0086965, 0.0086965
2: 0.0202439, 0.0299775, 0.0202439, 0.0299775, -0.0062917, 0.0062917
3: 0.0058958, 0.0167632, 0.0058958, 0.0167632, -0.0079331, 0.0079331
4: -0.0176357, -0.0070568, -0.0176357, -0.0070568, -0.0084218, 0.0084218
5: 0.0122690, 0.0256602, 0.0122690, 0.0256602, -0.0106004, 0.0106004
6: 0.0046071, 0.0146955, 0.0046071, 0.0146955, -0.0080338, 0.0080338
7: -0.0222719, -0.0112348, -0.0222719, -0.0112348, -0.0094056, 0.0094056
8: 0.0074112, 0.0183907, 0.0074112, 0.0183907, -0.0080099, 0.0080099
9: 0.9025575, 0.9498876, 0.9025575, 0.9498876, -0.0309629, 0.0309629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0198252
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0198834
time: 0.70 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041215, -0.0010920, -0.0042008, -0.0009724, -0.0031491, 0.0031089
1: 0.0174430, 0.0316575, 0.0164073, 0.0319212, -0.0094539, 0.0100554
2: 0.0202439, 0.0299775, 0.0196588, 0.0302397, -0.0068379, 0.0071247
3: 0.0058958, 0.0167632, 0.0050006, 0.0170262, -0.0083230, 0.0089340
4: -0.0176357, -0.0070568, -0.0179647, -0.0061915, -0.0092819, 0.0087468
5: 0.0122690, 0.0256602, 0.0111131, 0.0260109, -0.0110344, 0.0117387
6: 0.0046071, 0.0146955, 0.0037587, 0.0149839, -0.0083352, 0.0088846
7: -0.0222719, -0.0112348, -0.0225591, -0.0102905, -0.0102760, 0.0097122
8: 0.0074112, 0.0183907, 0.0065277, 0.0186232, -0.0084250, 0.0090273
9: 0.9025575, 0.9498876, 0.9013567, 0.9535970, -0.0356055, 0.0330085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0208602
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0209352
time: 0.71 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042008, -0.0009724, -0.0041215, -0.0010920, -0.0031089, 0.0031491
1: 0.0164073, 0.0319212, 0.0174430, 0.0316575, -0.0100554, 0.0094539
2: 0.0196588, 0.0302397, 0.0202439, 0.0299775, -0.0071247, 0.0068379
3: 0.0050006, 0.0170262, 0.0058958, 0.0167632, -0.0089340, 0.0083230
4: -0.0179647, -0.0061915, -0.0176357, -0.0070568, -0.0087468, 0.0092819
5: 0.0111131, 0.0260109, 0.0122690, 0.0256602, -0.0117387, 0.0110344
6: 0.0037587, 0.0149839, 0.0046071, 0.0146955, -0.0088846, 0.0083352
7: -0.0225591, -0.0102905, -0.0222719, -0.0112348, -0.0097122, 0.0102760
8: 0.0065277, 0.0186232, 0.0074112, 0.0183907, -0.0090273, 0.0084250
9: 0.9013567, 0.9535970, 0.9025575, 0.9498876, -0.0330085, 0.0356055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215
time: 0.68 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042008, -0.0009724, -0.0042008, -0.0009724, -0.0032284, 0.0032284
1: 0.0164073, 0.0319212, 0.0164073, 0.0319212, -0.0093662, 0.0093662
2: 0.0196588, 0.0302397, 0.0196588, 0.0302397, -0.0067570, 0.0067570
3: 0.0050006, 0.0170262, 0.0050006, 0.0170262, -0.0084649, 0.0084649
4: -0.0179647, -0.0061915, -0.0179647, -0.0061915, -0.0088288, 0.0088288
5: 0.0111131, 0.0260109, 0.0111131, 0.0260109, -0.0112278, 0.0112278
6: 0.0037587, 0.0149839, 0.0037587, 0.0149839, -0.0084751, 0.0084751
7: -0.0225591, -0.0102905, -0.0225591, -0.0102905, -0.0098946, 0.0098946
8: 0.0065277, 0.0186232, 0.0065277, 0.0186232, -0.0086163, 0.0086163
9: 0.9013567, 0.9535970, 0.9013567, 0.9535970, -0.0329643, 0.0329643

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215
time: 0.69 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.70 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0198252
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0198834
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0208602
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0197215, upper bound: 0.0209352
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 2.70
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0041160, -0.0011010, -0.0030014, 0.0029932
1: 0.0177250, 0.0316534, 0.0175254, 0.0316563, -0.0083628, 0.0085894
2: 0.0203672, 0.0299409, 0.0202804, 0.0299669, -0.0061071, 0.0061817
3: 0.0061374, 0.0167598, 0.0059679, 0.0167622, -0.0076375, 0.0078420
4: -0.0176199, -0.0072131, -0.0176311, -0.0071034, -0.0083175, 0.0081617
5: 0.0124980, 0.0256477, 0.0123378, 0.0256566, -0.0102661, 0.0104892
6: 0.0047825, 0.0146847, 0.0046590, 0.0146924, -0.0077735, 0.0079453
7: -0.0222562, -0.0113836, -0.0222673, -0.0112790, -0.0092901, 0.0091334
8: 0.0076742, 0.0183907, 0.0074885, 0.0183907, -0.0076999, 0.0079154
9: 0.9025774, 0.9489240, 0.9025633, 0.9496014, -0.0305869, 0.0297900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0198596
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0198596
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0041160, -0.0011012, -0.0030385, 0.0030677
1: 0.0175815, 0.0317189, 0.0175355, 0.0316565, -0.0086252, 0.0088929
2: 0.0203374, 0.0300414, 0.0202859, 0.0299669, -0.0061756, 0.0064249
3: 0.0060007, 0.0168510, 0.0059702, 0.0167622, -0.0078430, 0.0079751
4: -0.0176961, -0.0071435, -0.0176309, -0.0071098, -0.0083826, 0.0083218
5: 0.0123361, 0.0257415, 0.0123450, 0.0256564, -0.0105430, 0.0105877
6: 0.0046862, 0.0147623, 0.0046668, 0.0146922, -0.0079534, 0.0080192
7: -0.0223200, -0.0112477, -0.0222671, -0.0112847, -0.0093387, 0.0093753
8: 0.0075015, 0.0184785, 0.0074931, 0.0183907, -0.0079497, 0.0080898
9: 0.9022455, 0.9492757, 0.9025635, 0.9495751, -0.0311799, 0.0304267

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0199176
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0199176
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0041952, -0.0009845, -0.0031179, 0.0030724
1: 0.0177250, 0.0316534, 0.0164897, 0.0319202, -0.0091204, 0.0099500
2: 0.0203672, 0.0299409, 0.0196937, 0.0302294, -0.0066545, 0.0070155
3: 0.0061374, 0.0167598, 0.0050717, 0.0170252, -0.0080274, 0.0088398
4: -0.0176199, -0.0072131, -0.0179599, -0.0062375, -0.0091760, 0.0084868
5: 0.0124980, 0.0256477, 0.0111818, 0.0260071, -0.0107000, 0.0116214
6: 0.0047825, 0.0146847, 0.0038111, 0.0149807, -0.0080750, 0.0087937
7: -0.0222562, -0.0113836, -0.0225543, -0.0103348, -0.0101574, 0.0094404
8: 0.0076742, 0.0183907, 0.0066044, 0.0186232, -0.0081150, 0.0089330
9: 0.9025774, 0.9489240, 0.9013627, 0.9533098, -0.0352272, 0.0318354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0208602
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0208602
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0041951, -0.0009848, -0.0031549, 0.0031469
1: 0.0175815, 0.0317189, 0.0165025, 0.0319202, -0.0093828, 0.0102437
2: 0.0203374, 0.0300414, 0.0197025, 0.0302291, -0.0067229, 0.0072508
3: 0.0060007, 0.0168510, 0.0050803, 0.0170252, -0.0082329, 0.0089676
4: -0.0176961, -0.0071435, -0.0179596, -0.0062453, -0.0092367, 0.0086466
5: 0.0123361, 0.0257415, 0.0111916, 0.0260069, -0.0109768, 0.0117209
6: 0.0046862, 0.0147623, 0.0038193, 0.0149805, -0.0082549, 0.0088647
7: -0.0223200, -0.0112477, -0.0225541, -0.0103410, -0.0102046, 0.0096817
8: 0.0075015, 0.0184785, 0.0066149, 0.0186232, -0.0083648, 0.0090995
9: 0.9022455, 0.9492757, 0.9013630, 0.9532805, -0.0357786, 0.0324721

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0209352
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0209352
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041952, -0.0009845, -0.0041024, -0.0011228, -0.0030724, 0.0031179
1: 0.0164897, 0.0319202, 0.0177250, 0.0316534, -0.0099500, 0.0091204
2: 0.0196937, 0.0302294, 0.0203672, 0.0299409, -0.0070155, 0.0066545
3: 0.0050717, 0.0170252, 0.0061374, 0.0167598, -0.0088398, 0.0080274
4: -0.0179599, -0.0062375, -0.0176199, -0.0072131, -0.0084868, 0.0091760
5: 0.0111818, 0.0260071, 0.0124980, 0.0256477, -0.0116214, 0.0107000
6: 0.0038111, 0.0149807, 0.0047825, 0.0146847, -0.0087937, 0.0080750
7: -0.0225543, -0.0103348, -0.0222562, -0.0113836, -0.0094404, 0.0101574
8: 0.0066044, 0.0186232, 0.0076742, 0.0183907, -0.0089330, 0.0081150
9: 0.9013627, 0.9533098, 0.9025774, 0.9489240, -0.0318354, 0.0352272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0196924
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041951, -0.0009848, -0.0041397, -0.0010483, -0.0031469, 0.0031549
1: 0.0165025, 0.0319202, 0.0175815, 0.0317189, -0.0102437, 0.0093828
2: 0.0197025, 0.0302291, 0.0203374, 0.0300414, -0.0072508, 0.0067229
3: 0.0050803, 0.0170252, 0.0060007, 0.0168510, -0.0089676, 0.0082329
4: -0.0179596, -0.0062453, -0.0176961, -0.0071435, -0.0086466, 0.0092367
5: 0.0111916, 0.0260069, 0.0123361, 0.0257415, -0.0117209, 0.0109768
6: 0.0038193, 0.0149805, 0.0046862, 0.0147623, -0.0088647, 0.0082549
7: -0.0225541, -0.0103410, -0.0223200, -0.0112477, -0.0096817, 0.0102046
8: 0.0066149, 0.0186232, 0.0075015, 0.0184785, -0.0090995, 0.0083648
9: 0.9013630, 0.9532805, 0.9022455, 0.9492757, -0.0324721, 0.0357786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0196924
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041952, -0.0009845, -0.0041816, -0.0010133, -0.0031819, 0.0031971
1: 0.0164897, 0.0319202, 0.0166869, 0.0319175, -0.0092592, 0.0090304
2: 0.0196937, 0.0302294, 0.0197779, 0.0302042, -0.0066450, 0.0065710
3: 0.0050717, 0.0170252, 0.0052423, 0.0170228, -0.0083739, 0.0081696
4: -0.0179599, -0.0062375, -0.0179484, -0.0063458, -0.0085670, 0.0087240
5: 0.0111818, 0.0260071, 0.0113442, 0.0259980, -0.0111150, 0.0108903
6: 0.0038111, 0.0149807, 0.0039356, 0.0149728, -0.0083870, 0.0082156
7: -0.0225543, -0.0103348, -0.0225430, -0.0104393, -0.0096209, 0.0097791
8: 0.0066044, 0.0186232, 0.0067857, 0.0186232, -0.0085206, 0.0083062
9: 0.9013627, 0.9533098, 0.9013770, 0.9526234, -0.0317938, 0.0325907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0196924
time: 0.70 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041951, -0.0009848, -0.0042187, -0.0009181, -0.0032770, 0.0032339
1: 0.0165025, 0.0319202, 0.0165520, 0.0319708, -0.0095584, 0.0092774
2: 0.0197025, 0.0302291, 0.0197592, 0.0302962, -0.0068767, 0.0066358
3: 0.0050803, 0.0170252, 0.0051241, 0.0171198, -0.0085050, 0.0083601
4: -0.0179596, -0.0062453, -0.0180029, -0.0062789, -0.0087142, 0.0087915
5: 0.0111916, 0.0260069, 0.0111924, 0.0260893, -0.0112127, 0.0111463
6: 0.0038193, 0.0149805, 0.0038457, 0.0150357, -0.0084644, 0.0083859
7: -0.0225541, -0.0103410, -0.0225917, -0.0103089, -0.0098440, 0.0098275
8: 0.0066149, 0.0186232, 0.0066378, 0.0187207, -0.0086928, 0.0085346
9: 0.9013630, 0.9532805, 0.9010318, 0.9529719, -0.0323633, 0.0331607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0196924
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215
time: 0.70 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.78 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0198596
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0198596
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0199176
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0198596, upper bound: 0.0199176
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0208602
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0208602
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0209352
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0196924, upper bound: 0.0209352
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0196924
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0196924
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0196924
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0208602, upper bound: 0.0197215
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0196924
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.78
Output dim: 9, lower bound: -0.0209352, upper bound: 0.0197215

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0041024, -0.0011228, -0.0029796, 0.0029796
1: 0.0177250, 0.0316534, 0.0177250, 0.0316534, -0.0083568, 0.0083569
2: 0.0203672, 0.0299409, 0.0203672, 0.0299409, -0.0060651, 0.0060651
3: 0.0061374, 0.0167598, 0.0061374, 0.0167598, -0.0076349, 0.0076349
4: -0.0176199, -0.0072131, -0.0176199, -0.0072131, -0.0081408, 0.0081408
5: 0.0124980, 0.0256477, 0.0124980, 0.0256477, -0.0102564, 0.0102564
6: 0.0047825, 0.0146847, 0.0047825, 0.0146847, -0.0077648, 0.0077648
7: -0.0222562, -0.0113836, -0.0222562, -0.0113836, -0.0091061, 0.0091061
8: 0.0076742, 0.0183907, 0.0076742, 0.0183907, -0.0076989, 0.0076989
9: 0.9025774, 0.9489240, 0.9025774, 0.9489240, -0.0297745, 0.0297745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0194558, upper bound: 0.0191976
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0197052, upper bound: 0.0196882
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0041397, -0.0010483, -0.0030542, 0.0030169
1: 0.0177250, 0.0316534, 0.0175815, 0.0317189, -0.0086488, 0.0086409
2: 0.0203672, 0.0299409, 0.0203374, 0.0300414, -0.0063051, 0.0061145
3: 0.0061374, 0.0167598, 0.0060007, 0.0168510, -0.0077651, 0.0078053
4: -0.0176199, -0.0072131, -0.0176961, -0.0071435, -0.0082319, 0.0082098
5: 0.0124980, 0.0256477, 0.0123361, 0.0257415, -0.0103614, 0.0104619
6: 0.0047825, 0.0146847, 0.0046862, 0.0147623, -0.0078426, 0.0078899
7: -0.0222562, -0.0113836, -0.0223200, -0.0112477, -0.0092831, 0.0091611
8: 0.0076742, 0.0183907, 0.0075015, 0.0184785, -0.0078661, 0.0079329
9: 0.9025774, 0.9489240, 0.9022455, 0.9492757, -0.0302910, 0.0303423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0192130, upper bound: 0.0194409
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0197052, upper bound: 0.0196882
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0041024, -0.0011228, -0.0030169, 0.0030542
1: 0.0175815, 0.0317189, 0.0177250, 0.0316534, -0.0086409, 0.0086488
2: 0.0203374, 0.0300414, 0.0203672, 0.0299409, -0.0061145, 0.0063051
3: 0.0060007, 0.0168510, 0.0061374, 0.0167598, -0.0078053, 0.0077651
4: -0.0176961, -0.0071435, -0.0176199, -0.0072131, -0.0082098, 0.0082319
5: 0.0123361, 0.0257415, 0.0124980, 0.0256477, -0.0104619, 0.0103614
6: 0.0046862, 0.0147623, 0.0047825, 0.0146847, -0.0078899, 0.0078426
7: -0.0223200, -0.0112477, -0.0222562, -0.0113836, -0.0091611, 0.0092831
8: 0.0075015, 0.0184785, 0.0076742, 0.0183907, -0.0079329, 0.0078661
9: 0.9022455, 0.9492757, 0.9025774, 0.9489240, -0.0303423, 0.0302910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0194409, upper bound: 0.0192562
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0196882, upper bound: 0.0197450
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0041397, -0.0010483, -0.0030915, 0.0030915
1: 0.0175815, 0.0317189, 0.0175815, 0.0317189, -0.0086757, 0.0086757
2: 0.0203374, 0.0300414, 0.0203374, 0.0300414, -0.0062757, 0.0062757
3: 0.0060007, 0.0168510, 0.0060007, 0.0168510, -0.0078468, 0.0078468
4: -0.0176961, -0.0071435, -0.0176961, -0.0071435, -0.0083038, 0.0083038
5: 0.0123361, 0.0257415, 0.0123361, 0.0257415, -0.0105380, 0.0105380
6: 0.0046862, 0.0147623, 0.0046862, 0.0147623, -0.0079476, 0.0079476
7: -0.0223200, -0.0112477, -0.0223200, -0.0112477, -0.0093489, 0.0093489
8: 0.0075015, 0.0184785, 0.0075015, 0.0184785, -0.0079628, 0.0079628
9: 0.9022455, 0.9492757, 0.9022455, 0.9492757, -0.0304397, 0.0304397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191976, upper bound: 0.0194986
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0196882, upper bound: 0.0197450
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0041816, -0.0010133, -0.0030891, 0.0030588
1: 0.0177250, 0.0316534, 0.0166869, 0.0319175, -0.0091151, 0.0097189
2: 0.0203672, 0.0299409, 0.0197779, 0.0302042, -0.0066160, 0.0068951
3: 0.0061374, 0.0167598, 0.0052423, 0.0170228, -0.0080248, 0.0086190
4: -0.0176199, -0.0072131, -0.0179484, -0.0063458, -0.0089916, 0.0084668
5: 0.0124980, 0.0256477, 0.0113442, 0.0259980, -0.0106903, 0.0113721
6: 0.0047825, 0.0146847, 0.0039356, 0.0149728, -0.0080666, 0.0086030
7: -0.0222562, -0.0113836, -0.0225430, -0.0104393, -0.0099619, 0.0094143
8: 0.0076742, 0.0183907, 0.0067857, 0.0186232, -0.0081141, 0.0087055
9: 0.9025774, 0.9489240, 0.9013770, 0.9526234, -0.0343649, 0.0318200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193032, upper bound: 0.0201477
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195174, upper bound: 0.0206923
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0042187, -0.0009181, -0.0031844, 0.0030959
1: 0.0177250, 0.0316534, 0.0165520, 0.0319708, -0.0092414, 0.0099498
2: 0.0203672, 0.0299409, 0.0197592, 0.0302962, -0.0067968, 0.0069300
3: 0.0061374, 0.0167598, 0.0051241, 0.0171198, -0.0081465, 0.0087732
4: -0.0176199, -0.0072131, -0.0180029, -0.0062789, -0.0090694, 0.0085548
5: 0.0124980, 0.0256477, 0.0111924, 0.0260893, -0.0108200, 0.0115587
6: 0.0047825, 0.0146847, 0.0038457, 0.0150357, -0.0081653, 0.0087123
7: -0.0222562, -0.0113836, -0.0225917, -0.0103089, -0.0101220, 0.0094766
8: 0.0076742, 0.0183907, 0.0066378, 0.0187207, -0.0082345, 0.0089088
9: 0.9025774, 0.9489240, 0.9010318, 0.9529719, -0.0347801, 0.0323305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0193032, upper bound: 0.0201477
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195174, upper bound: 0.0206923
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0041816, -0.0010133, -0.0031264, 0.0031334
1: 0.0175815, 0.0317189, 0.0166869, 0.0319175, -0.0093991, 0.0100108
2: 0.0203374, 0.0300414, 0.0197779, 0.0302042, -0.0066654, 0.0071351
3: 0.0060007, 0.0168510, 0.0052423, 0.0170228, -0.0081952, 0.0087492
4: -0.0176961, -0.0071435, -0.0179484, -0.0063458, -0.0090606, 0.0085580
5: 0.0123361, 0.0257415, 0.0113442, 0.0259980, -0.0108957, 0.0114771
6: 0.0046862, 0.0147623, 0.0039356, 0.0149728, -0.0081917, 0.0086808
7: -0.0223200, -0.0112477, -0.0225430, -0.0104393, -0.0100169, 0.0095913
8: 0.0075015, 0.0184785, 0.0067857, 0.0186232, -0.0083480, 0.0088727
9: 0.9022455, 0.9492757, 0.9013770, 0.9526234, -0.0349327, 0.0323365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0192955, upper bound: 0.0202180
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195113, upper bound: 0.0207628
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0042187, -0.0009181, -0.0032217, 0.0031705
1: 0.0175815, 0.0317189, 0.0165520, 0.0319708, -0.0094059, 0.0099983
2: 0.0203374, 0.0300414, 0.0197592, 0.0302962, -0.0068019, 0.0070930
3: 0.0060007, 0.0168510, 0.0051241, 0.0171198, -0.0082382, 0.0087978
4: -0.0176961, -0.0071435, -0.0180029, -0.0062789, -0.0091225, 0.0086312
5: 0.0123361, 0.0257415, 0.0111924, 0.0260893, -0.0109753, 0.0116024
6: 0.0046862, 0.0147623, 0.0038457, 0.0150357, -0.0082514, 0.0087522
7: -0.0223200, -0.0112477, -0.0225917, -0.0103089, -0.0101623, 0.0096585
8: 0.0075015, 0.0184785, 0.0066378, 0.0187207, -0.0083773, 0.0089326
9: 0.9022455, 0.9492757, 0.9010318, 0.9529719, -0.0349245, 0.0324978

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0192955, upper bound: 0.0202180
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0195113, upper bound: 0.0207628
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0041024, -0.0011228, -0.0030588, 0.0030891
1: 0.0166869, 0.0319175, 0.0177250, 0.0316534, -0.0097189, 0.0091151
2: 0.0197779, 0.0302042, 0.0203672, 0.0299409, -0.0068951, 0.0066160
3: 0.0052423, 0.0170228, 0.0061374, 0.0167598, -0.0086190, 0.0080248
4: -0.0179484, -0.0063458, -0.0176199, -0.0072131, -0.0084668, 0.0089916
5: 0.0113442, 0.0259980, 0.0124980, 0.0256477, -0.0113721, 0.0106903
6: 0.0039356, 0.0149728, 0.0047825, 0.0146847, -0.0086030, 0.0080666
7: -0.0225430, -0.0104393, -0.0222562, -0.0113836, -0.0094143, 0.0099619
8: 0.0067857, 0.0186232, 0.0076742, 0.0183907, -0.0087055, 0.0081141
9: 0.9013770, 0.9526234, 0.9025774, 0.9489240, -0.0318200, 0.0343649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0193032
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195174
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042187, -0.0009181, -0.0041024, -0.0011228, -0.0030959, 0.0031844
1: 0.0165520, 0.0319708, 0.0177250, 0.0316534, -0.0099498, 0.0092414
2: 0.0197592, 0.0302962, 0.0203672, 0.0299409, -0.0069300, 0.0067968
3: 0.0051241, 0.0171198, 0.0061374, 0.0167598, -0.0087732, 0.0081465
4: -0.0180029, -0.0062789, -0.0176199, -0.0072131, -0.0085548, 0.0090694
5: 0.0111924, 0.0260893, 0.0124980, 0.0256477, -0.0115587, 0.0108200
6: 0.0038457, 0.0150357, 0.0047825, 0.0146847, -0.0087123, 0.0081653
7: -0.0225917, -0.0103089, -0.0222562, -0.0113836, -0.0094766, 0.0101220
8: 0.0066378, 0.0187207, 0.0076742, 0.0183907, -0.0089088, 0.0082345
9: 0.9010318, 0.9529719, 0.9025774, 0.9489240, -0.0323306, 0.0347801

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0193381
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0041397, -0.0010483, -0.0031334, 0.0031264
1: 0.0166869, 0.0319175, 0.0175815, 0.0317189, -0.0100108, 0.0093991
2: 0.0197779, 0.0302042, 0.0203374, 0.0300414, -0.0071351, 0.0066654
3: 0.0052423, 0.0170228, 0.0060007, 0.0168510, -0.0087492, 0.0081952
4: -0.0179484, -0.0063458, -0.0176961, -0.0071435, -0.0085580, 0.0090606
5: 0.0113442, 0.0259980, 0.0123361, 0.0257415, -0.0114771, 0.0108957
6: 0.0039356, 0.0149728, 0.0046862, 0.0147623, -0.0086808, 0.0081917
7: -0.0225430, -0.0104393, -0.0223200, -0.0112477, -0.0095913, 0.0100169
8: 0.0067857, 0.0186232, 0.0075015, 0.0184785, -0.0088727, 0.0083480
9: 0.9013770, 0.9526234, 0.9022455, 0.9492757, -0.0323365, 0.0349327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0192955
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195113
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042187, -0.0009181, -0.0041397, -0.0010483, -0.0031705, 0.0032217
1: 0.0165520, 0.0319708, 0.0175815, 0.0317189, -0.0099983, 0.0094058
2: 0.0197592, 0.0302962, 0.0203374, 0.0300414, -0.0070930, 0.0068019
3: 0.0051241, 0.0171198, 0.0060007, 0.0168510, -0.0087978, 0.0082382
4: -0.0180029, -0.0062789, -0.0176961, -0.0071435, -0.0086312, 0.0091225
5: 0.0111924, 0.0260893, 0.0123361, 0.0257415, -0.0116024, 0.0109753
6: 0.0038457, 0.0150357, 0.0046862, 0.0147623, -0.0087522, 0.0082514
7: -0.0225917, -0.0103089, -0.0223200, -0.0112477, -0.0096585, 0.0101623
8: 0.0066378, 0.0187207, 0.0075015, 0.0184785, -0.0089326, 0.0083773
9: 0.9010318, 0.9529719, 0.9022455, 0.9492757, -0.0324978, 0.0349245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0193381
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0041816, -0.0010133, -0.0031683, 0.0031683
1: 0.0166869, 0.0319175, 0.0166869, 0.0319175, -0.0090243, 0.0090243
2: 0.0197779, 0.0302042, 0.0197779, 0.0302042, -0.0065278, 0.0065278
3: 0.0052423, 0.0170228, 0.0052423, 0.0170228, -0.0081669, 0.0081669
4: -0.0179484, -0.0063458, -0.0179484, -0.0063458, -0.0085464, 0.0085464
5: 0.0113442, 0.0259980, 0.0113442, 0.0259980, -0.0108805, 0.0108805
6: 0.0039356, 0.0149728, 0.0039356, 0.0149728, -0.0082071, 0.0082071
7: -0.0225430, -0.0104393, -0.0225430, -0.0104393, -0.0095939, 0.0095939
8: 0.0067857, 0.0186232, 0.0067857, 0.0186232, -0.0083053, 0.0083053
9: 0.9013770, 0.9526234, 0.9013770, 0.9526234, -0.0317781, 0.0317781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204940, upper bound: 0.0191088
time: 0.66 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195174
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0042187, -0.0009181, -0.0041816, -0.0010133, -0.0032054, 0.0032636
1: 0.0165520, 0.0319708, 0.0166869, 0.0319175, -0.0092925, 0.0093156
2: 0.0197592, 0.0302962, 0.0197779, 0.0302042, -0.0065755, 0.0067567
3: 0.0051241, 0.0171198, 0.0052423, 0.0170228, -0.0083271, 0.0082971
4: -0.0180029, -0.0062789, -0.0179484, -0.0063458, -0.0086174, 0.0086341
5: 0.0111924, 0.0260893, 0.0113442, 0.0259980, -0.0110769, 0.0109846
6: 0.0038457, 0.0150357, 0.0039356, 0.0149728, -0.0083245, 0.0082893
7: -0.0225917, -0.0103089, -0.0225430, -0.0104393, -0.0096496, 0.0097616
8: 0.0066378, 0.0187207, 0.0067857, 0.0186232, -0.0085259, 0.0084700
9: 0.9010318, 0.9529719, 0.9013770, 0.9526234, -0.0323325, 0.0322645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204940, upper bound: 0.0191260
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0042187, -0.0009181, -0.0032636, 0.0032054
1: 0.0166869, 0.0319175, 0.0165520, 0.0319708, -0.0093156, 0.0092925
2: 0.0197779, 0.0302042, 0.0197592, 0.0302962, -0.0067567, 0.0065755
3: 0.0052423, 0.0170228, 0.0051241, 0.0171198, -0.0082971, 0.0083271
4: -0.0179484, -0.0063458, -0.0180029, -0.0062789, -0.0086341, 0.0086174
5: 0.0113442, 0.0259980, 0.0111924, 0.0260893, -0.0109846, 0.0110769
6: 0.0039356, 0.0149728, 0.0038457, 0.0150357, -0.0082893, 0.0083245
7: -0.0225430, -0.0104393, -0.0225917, -0.0103089, -0.0097616, 0.0096496
8: 0.0067857, 0.0186232, 0.0066378, 0.0187207, -0.0084700, 0.0085259
9: 0.9013770, 0.9526234, 0.9010318, 0.9529719, -0.0322645, 0.0323325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0192955
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195113
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0042187, -0.0009181, -0.0042187, -0.0009181, -0.0033007, 0.0033007
1: 0.0165520, 0.0319708, 0.0165520, 0.0319708, -0.0093270, 0.0093270
2: 0.0197592, 0.0302962, 0.0197592, 0.0302962, -0.0067272, 0.0067272
3: 0.0051241, 0.0171198, 0.0051241, 0.0171198, -0.0083642, 0.0083642
4: -0.0180029, -0.0062789, -0.0180029, -0.0062789, -0.0086965, 0.0086965
5: 0.0111924, 0.0260893, 0.0111924, 0.0260893, -0.0111413, 0.0111413
6: 0.0038457, 0.0150357, 0.0038457, 0.0150357, -0.0083804, 0.0083804
7: -0.0225917, -0.0103089, -0.0225917, -0.0103089, -0.0098180, 0.0098180
8: 0.0066378, 0.0187207, 0.0066378, 0.0187207, -0.0085484, 0.0085484
9: 0.9010318, 0.9529719, 0.9010318, 0.9529719, -0.0323815, 0.0323815

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 219

## Relational analysis of IS_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0204940, upper bound: 0.0191260
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.66 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0194558, upper bound: 0.0191976
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0197052, upper bound: 0.0196882
IS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0192130, upper bound: 0.0194409
IS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0197052, upper bound: 0.0196882
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0194409, upper bound: 0.0192562
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0196882, upper bound: 0.0197450
IS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0191976, upper bound: 0.0194986
IS_A1_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0196882, upper bound: 0.0197450
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0193032, upper bound: 0.0201477
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0195174, upper bound: 0.0206923
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0193032, upper bound: 0.0201477
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0195174, upper bound: 0.0206923
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0192955, upper bound: 0.0202180
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0195113, upper bound: 0.0207628
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0192955, upper bound: 0.0202180
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0195113, upper bound: 0.0207628
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0193032
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195174
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0193381
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0192955
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195113
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0193381
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0204940, upper bound: 0.0191088
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195174
IS_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0204940, upper bound: 0.0191260
IS_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476
IS_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0201477, upper bound: 0.0192955
IS_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195113
IS_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0204940, upper bound: 0.0191260
IS_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 2.66
Output dim: 9, lower bound: -0.0206923, upper bound: 0.0195476

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040950, -0.0011180, -0.0040865, -0.0011470, -0.0029480, 0.0029685
1: 0.0180102, 0.0318097, 0.0178614, 0.0316527, -0.0080516, 0.0083379
2: 0.0205854, 0.0300432, 0.0204404, 0.0299116, -0.0057675, 0.0060788
3: 0.0062686, 0.0168785, 0.0062230, 0.0167586, -0.0073974, 0.0076357
4: -0.0177240, -0.0073628, -0.0176149, -0.0072804, -0.0081699, 0.0079128
5: 0.0126354, 0.0257630, 0.0125946, 0.0256433, -0.0100189, 0.0102574
6: 0.0048994, 0.0147838, 0.0048514, 0.0146812, -0.0075583, 0.0077805
7: -0.0223695, -0.0115517, -0.0222509, -0.0114797, -0.0091105, 0.0088727
8: 0.0077603, 0.0184851, 0.0077587, 0.0183907, -0.0075066, 0.0076638
9: 0.9020864, 0.9481564, 0.9025850, 0.9485730, -0.0298344, 0.0286708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171883, upper bound: 0.0175610
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0177182, upper bound: 0.0173856
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0041024, -0.0011228, -0.0029700, 0.0029644
1: 0.0178103, 0.0316530, 0.0177250, 0.0316534, -0.0081800, 0.0083547
2: 0.0204141, 0.0299231, 0.0203672, 0.0299409, -0.0059418, 0.0060358
3: 0.0061951, 0.0167591, 0.0061374, 0.0167598, -0.0075599, 0.0076345
4: -0.0176166, -0.0072584, -0.0176199, -0.0072131, -0.0081393, 0.0080797
5: 0.0125631, 0.0256449, 0.0124980, 0.0256477, -0.0101705, 0.0102551
6: 0.0048292, 0.0146824, 0.0047825, 0.0146847, -0.0077069, 0.0077637
7: -0.0222527, -0.0114462, -0.0222562, -0.0113836, -0.0091047, 0.0090154
8: 0.0077319, 0.0183907, 0.0076742, 0.0183907, -0.0076173, 0.0076998
9: 0.9025822, 0.9486928, 0.9025774, 0.9489240, -0.0297722, 0.0294399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0179775, upper bound: 0.0174208
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0181550, upper bound: 0.0181550
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0040865, -0.0011470, -0.0041334, -0.0010424, -0.0030441, 0.0029864
1: 0.0178614, 0.0316527, 0.0178618, 0.0318816, -0.0086041, 0.0083456
2: 0.0204404, 0.0299116, 0.0205547, 0.0301407, -0.0063153, 0.0058175
3: 0.0062230, 0.0167586, 0.0061322, 0.0169652, -0.0077556, 0.0075786
4: -0.0176149, -0.0072804, -0.0177937, -0.0072913, -0.0080041, 0.0082339
5: 0.0125946, 0.0256433, 0.0124724, 0.0258513, -0.0103499, 0.0102270
6: 0.0048514, 0.0146812, 0.0048014, 0.0148554, -0.0078527, 0.0076862
7: -0.0222509, -0.0114797, -0.0224280, -0.0114140, -0.0090500, 0.0091642
8: 0.0077587, 0.0183907, 0.0075854, 0.0185713, -0.0078338, 0.0077467
9: 0.9025850, 0.9485730, 0.9017551, 0.9485164, -0.0292058, 0.0303578

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165393, upper bound: 0.0167171
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159813, upper bound: 0.0166630
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0041301, -0.0010666, -0.0030358, 0.0030072
1: 0.0177250, 0.0316534, 0.0176665, 0.0317185, -0.0086464, 0.0084696
2: 0.0203672, 0.0299409, 0.0203836, 0.0300238, -0.0062761, 0.0059886
3: 0.0061374, 0.0167598, 0.0060571, 0.0168502, -0.0077654, 0.0077325
4: -0.0176199, -0.0072131, -0.0176928, -0.0071885, -0.0081706, 0.0082082
5: 0.0124980, 0.0256477, 0.0124001, 0.0257386, -0.0103600, 0.0103761
6: 0.0047825, 0.0146847, 0.0047325, 0.0147601, -0.0078413, 0.0078327
7: -0.0222562, -0.0113836, -0.0223166, -0.0113103, -0.0091913, 0.0091595
8: 0.0076742, 0.0183907, 0.0075582, 0.0184785, -0.0078652, 0.0078519
9: 0.9025774, 0.9489240, 0.9022505, 0.9490474, -0.0299596, 0.0303396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176707, upper bound: 0.0175101
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175323, upper bound: 0.0175101
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041334, -0.0010424, -0.0040865, -0.0011470, -0.0029864, 0.0030441
1: 0.0178618, 0.0318816, 0.0178614, 0.0316527, -0.0083456, 0.0086041
2: 0.0205547, 0.0301407, 0.0204404, 0.0299116, -0.0058175, 0.0063153
3: 0.0061322, 0.0169652, 0.0062230, 0.0167586, -0.0075786, 0.0077556
4: -0.0177937, -0.0072913, -0.0176149, -0.0072804, -0.0082339, 0.0080041
5: 0.0124724, 0.0258513, 0.0125946, 0.0256433, -0.0102270, 0.0103499
6: 0.0048014, 0.0148554, 0.0048514, 0.0146812, -0.0076862, 0.0078527
7: -0.0224280, -0.0114140, -0.0222509, -0.0114797, -0.0091642, 0.0090500
8: 0.0075854, 0.0185713, 0.0077587, 0.0183907, -0.0077467, 0.0078338
9: 0.9017551, 0.9485164, 0.9025850, 0.9485730, -0.0303578, 0.0292058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0167171, upper bound: 0.0165393
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166630, upper bound: 0.0159813
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041301, -0.0010666, -0.0041024, -0.0011228, -0.0030072, 0.0030358
1: 0.0176665, 0.0317185, 0.0177250, 0.0316534, -0.0084696, 0.0086464
2: 0.0203836, 0.0300238, 0.0203672, 0.0299409, -0.0059886, 0.0062761
3: 0.0060571, 0.0168502, 0.0061374, 0.0167598, -0.0077325, 0.0077654
4: -0.0176928, -0.0071885, -0.0176199, -0.0072131, -0.0082082, 0.0081706
5: 0.0124001, 0.0257386, 0.0124980, 0.0256477, -0.0103761, 0.0103600
6: 0.0047325, 0.0147601, 0.0047825, 0.0146847, -0.0078327, 0.0078413
7: -0.0223166, -0.0113103, -0.0222562, -0.0113836, -0.0091594, 0.0091913
8: 0.0075582, 0.0184785, 0.0076742, 0.0183907, -0.0078519, 0.0078652
9: 0.9022505, 0.9490474, 0.9025774, 0.9489240, -0.0303396, 0.0299597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175101, upper bound: 0.0176707
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175101, upper bound: 0.0175323
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041237, -0.0010769, -0.0041334, -0.0010424, -0.0030813, 0.0030565
1: 0.0177171, 0.0317182, 0.0178618, 0.0318816, -0.0086493, 0.0083769
2: 0.0204105, 0.0300124, 0.0205547, 0.0301407, -0.0062882, 0.0059804
3: 0.0060851, 0.0168498, 0.0061322, 0.0169652, -0.0078470, 0.0076094
4: -0.0176912, -0.0072106, -0.0177937, -0.0072913, -0.0080733, 0.0083333
5: 0.0124329, 0.0257371, 0.0124724, 0.0258513, -0.0105380, 0.0102981
6: 0.0047545, 0.0147589, 0.0048014, 0.0148554, -0.0079642, 0.0077415
7: -0.0223148, -0.0113433, -0.0224280, -0.0114140, -0.0091135, 0.0093551
8: 0.0075853, 0.0184785, 0.0075854, 0.0185713, -0.0079266, 0.0077687
9: 0.9022529, 0.9489262, 0.9017551, 0.9485164, -0.0293325, 0.0305039

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160533, upper bound: 0.0170270
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159679, upper bound: 0.0166630
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0041301, -0.0010666, -0.0030731, 0.0030818
1: 0.0175815, 0.0317189, 0.0176665, 0.0317185, -0.0086734, 0.0085046
2: 0.0203374, 0.0300414, 0.0203836, 0.0300238, -0.0062460, 0.0061558
3: 0.0060007, 0.0168510, 0.0060571, 0.0168502, -0.0078474, 0.0077723
4: -0.0176961, -0.0071435, -0.0176928, -0.0071885, -0.0082425, 0.0083022
5: 0.0123361, 0.0257415, 0.0124001, 0.0257386, -0.0105366, 0.0104517
6: 0.0046862, 0.0147623, 0.0047325, 0.0147601, -0.0079464, 0.0078895
7: -0.0223200, -0.0112477, -0.0223166, -0.0113103, -0.0092596, 0.0093475
8: 0.0075015, 0.0184785, 0.0075582, 0.0184785, -0.0079632, 0.0078808
9: 0.9022455, 0.9492757, 0.9022505, 0.9490474, -0.0301073, 0.0304372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175756, upper bound: 0.0175214
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175101, upper bound: 0.0175214
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0040950, -0.0011180, -0.0041660, -0.0010444, -0.0030507, 0.0030480
1: 0.0180102, 0.0318097, 0.0168204, 0.0319168, -0.0088098, 0.0097071
2: 0.0205854, 0.0300432, 0.0198502, 0.0301768, -0.0063251, 0.0069124
3: 0.0062686, 0.0168785, 0.0053290, 0.0170216, -0.0077873, 0.0086247
4: -0.0177240, -0.0073628, -0.0179438, -0.0064135, -0.0090244, 0.0082382
5: 0.0126354, 0.0257630, 0.0114423, 0.0259942, -0.0104523, 0.0113769
6: 0.0048994, 0.0147838, 0.0040060, 0.0149697, -0.0078597, 0.0086230
7: -0.0223695, -0.0115517, -0.0225382, -0.0105332, -0.0099704, 0.0091802
8: 0.0077603, 0.0184851, 0.0068723, 0.0186232, -0.0079216, 0.0086755
9: 0.9020864, 0.9481564, 0.9013841, 0.9522769, -0.0344469, 0.0307152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175856, upper bound: 0.0175257
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175403, upper bound: 0.0182954
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0041816, -0.0010133, -0.0030795, 0.0030436
1: 0.0178103, 0.0316530, 0.0166869, 0.0319175, -0.0089277, 0.0097167
2: 0.0204141, 0.0299231, 0.0197779, 0.0302042, -0.0064918, 0.0068659
3: 0.0061951, 0.0167591, 0.0052423, 0.0170228, -0.0079498, 0.0086206
4: -0.0176166, -0.0072584, -0.0179484, -0.0063458, -0.0089901, 0.0084040
5: 0.0125631, 0.0256449, 0.0113442, 0.0259980, -0.0106043, 0.0113708
6: 0.0048292, 0.0146824, 0.0039356, 0.0149728, -0.0080087, 0.0086021
7: -0.0222527, -0.0114462, -0.0225430, -0.0104393, -0.0099605, 0.0093189
8: 0.0077319, 0.0183907, 0.0067857, 0.0186232, -0.0080325, 0.0087101
9: 0.9025822, 0.9486928, 0.9013770, 0.9526234, -0.0343625, 0.0314890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0177756, upper bound: 0.0182765
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0179548, upper bound: 0.0191510
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040950, -0.0011180, -0.0042033, -0.0009539, -0.0031411, 0.0030853
1: 0.0180102, 0.0318097, 0.0166832, 0.0319700, -0.0089359, 0.0099385
2: 0.0205854, 0.0300432, 0.0198314, 0.0302691, -0.0065049, 0.0069483
3: 0.0062686, 0.0168785, 0.0052076, 0.0171186, -0.0079090, 0.0087805
4: -0.0177240, -0.0073628, -0.0179981, -0.0063460, -0.0091024, 0.0083259
5: 0.0126354, 0.0257630, 0.0112891, 0.0260852, -0.0105819, 0.0115662
6: 0.0048994, 0.0147838, 0.0039154, 0.0150324, -0.0079583, 0.0087329
7: -0.0223695, -0.0115517, -0.0225869, -0.0104013, -0.0101311, 0.0092423
8: 0.0077603, 0.0184851, 0.0067198, 0.0187207, -0.0080422, 0.0088811
9: 0.9020864, 0.9481564, 0.9010386, 0.9526324, -0.0348658, 0.0312259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168891, upper bound: 0.0167734
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165747, upper bound: 0.0167143
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0042187, -0.0009181, -0.0031748, 0.0030807
1: 0.0178103, 0.0316530, 0.0165520, 0.0319708, -0.0090624, 0.0099476
2: 0.0204141, 0.0299231, 0.0197592, 0.0302962, -0.0066767, 0.0069008
3: 0.0061951, 0.0167591, 0.0051241, 0.0171198, -0.0080715, 0.0087745
4: -0.0176166, -0.0072584, -0.0180029, -0.0062789, -0.0090679, 0.0084922
5: 0.0125631, 0.0256449, 0.0111924, 0.0260893, -0.0107338, 0.0115574
6: 0.0048292, 0.0146824, 0.0038457, 0.0150357, -0.0081074, 0.0087115
7: -0.0222527, -0.0114462, -0.0225917, -0.0103089, -0.0101207, 0.0093797
8: 0.0077319, 0.0183907, 0.0066378, 0.0187207, -0.0081529, 0.0089143
9: 0.9025822, 0.9486928, 0.9010318, 0.9529719, -0.0347777, 0.0319949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0174846, upper bound: 0.0184965
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0173562, upper bound: 0.0184965
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041334, -0.0010424, -0.0041660, -0.0010444, -0.0030890, 0.0031236
1: 0.0178618, 0.0318816, 0.0168204, 0.0319168, -0.0091038, 0.0099733
2: 0.0205547, 0.0301407, 0.0198502, 0.0301768, -0.0063751, 0.0071489
3: 0.0061322, 0.0169652, 0.0053290, 0.0170216, -0.0079685, 0.0087446
4: -0.0177937, -0.0072913, -0.0179438, -0.0064135, -0.0090885, 0.0083296
5: 0.0124724, 0.0258513, 0.0114423, 0.0259942, -0.0106604, 0.0114694
6: 0.0048014, 0.0148554, 0.0040060, 0.0149697, -0.0079876, 0.0086953
7: -0.0224280, -0.0114140, -0.0225382, -0.0105332, -0.0100240, 0.0093575
8: 0.0075854, 0.0185713, 0.0068723, 0.0186232, -0.0081617, 0.0088455
9: 0.9017551, 0.9485164, 0.9013841, 0.9522769, -0.0349703, 0.0312502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165990, upper bound: 0.0174100
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165601, upper bound: 0.0167347
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041301, -0.0010666, -0.0041816, -0.0010133, -0.0031167, 0.0031150
1: 0.0176665, 0.0317185, 0.0166869, 0.0319175, -0.0092173, 0.0100085
2: 0.0203836, 0.0300238, 0.0197779, 0.0302042, -0.0065386, 0.0071062
3: 0.0060571, 0.0168502, 0.0052423, 0.0170228, -0.0081224, 0.0087515
4: -0.0176928, -0.0071885, -0.0179484, -0.0063458, -0.0090590, 0.0084949
5: 0.0124001, 0.0257386, 0.0113442, 0.0259980, -0.0108099, 0.0114757
6: 0.0047325, 0.0147601, 0.0039356, 0.0149728, -0.0081345, 0.0086798
7: -0.0223166, -0.0113103, -0.0225430, -0.0104393, -0.0100153, 0.0094949
8: 0.0075582, 0.0184785, 0.0067857, 0.0186232, -0.0082671, 0.0088755
9: 0.9022505, 0.9490474, 0.9013770, 0.9526234, -0.0349299, 0.0320088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0173539, upper bound: 0.0186859
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0173539, upper bound: 0.0185032
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041334, -0.0010424, -0.0042033, -0.0009539, -0.0031795, 0.0031609
1: 0.0178618, 0.0318816, 0.0166832, 0.0319700, -0.0091071, 0.0099827
2: 0.0205547, 0.0301407, 0.0198314, 0.0302691, -0.0065132, 0.0071071
3: 0.0061322, 0.0169652, 0.0052076, 0.0171186, -0.0080006, 0.0088039
4: -0.0177937, -0.0072913, -0.0179981, -0.0063460, -0.0091555, 0.0084002
5: 0.0124724, 0.0258513, 0.0112891, 0.0260852, -0.0107349, 0.0116083
6: 0.0048014, 0.0148554, 0.0039154, 0.0150324, -0.0080450, 0.0087716
7: -0.0224280, -0.0114140, -0.0225869, -0.0104013, -0.0101733, 0.0094224
8: 0.0075854, 0.0185713, 0.0067198, 0.0187207, -0.0081832, 0.0089030
9: 0.9017551, 0.9485164, 0.9010386, 0.9526324, -0.0350055, 0.0313897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0168582, upper bound: 0.0167660
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0165600, upper bound: 0.0167122
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041301, -0.0010666, -0.0042187, -0.0009181, -0.0032120, 0.0031521
1: 0.0176665, 0.0317185, 0.0165520, 0.0319708, -0.0092245, 0.0099960
2: 0.0203836, 0.0300238, 0.0197592, 0.0302962, -0.0066837, 0.0070633
3: 0.0060571, 0.0168502, 0.0051241, 0.0171198, -0.0081636, 0.0088001
4: -0.0176928, -0.0071885, -0.0180029, -0.0062789, -0.0091209, 0.0085682
5: 0.0124001, 0.0257386, 0.0111924, 0.0260893, -0.0108890, 0.0116011
6: 0.0047325, 0.0147601, 0.0038457, 0.0150357, -0.0081933, 0.0087515
7: -0.0223166, -0.0113103, -0.0225917, -0.0103089, -0.0101609, 0.0095653
8: 0.0075582, 0.0184785, 0.0066378, 0.0187207, -0.0082953, 0.0089377
9: 0.9022505, 0.9490474, 0.9010318, 0.9529719, -0.0349221, 0.0321693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0174345, upper bound: 0.0184975
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0173539, upper bound: 0.0184975
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041660, -0.0010444, -0.0040950, -0.0011180, -0.0030480, 0.0030507
1: 0.0168204, 0.0319168, 0.0180102, 0.0318097, -0.0097072, 0.0088098
2: 0.0198502, 0.0301768, 0.0205854, 0.0300432, -0.0069124, 0.0063251
3: 0.0053290, 0.0170216, 0.0062686, 0.0168785, -0.0086247, 0.0077873
4: -0.0179438, -0.0064135, -0.0177240, -0.0073628, -0.0082382, 0.0090244
5: 0.0114423, 0.0259942, 0.0126354, 0.0257630, -0.0113769, 0.0104523
6: 0.0040060, 0.0149697, 0.0048994, 0.0147838, -0.0086230, 0.0078597
7: -0.0225382, -0.0105332, -0.0223695, -0.0115517, -0.0091802, 0.0099704
8: 0.0068723, 0.0186232, 0.0077603, 0.0184851, -0.0086755, 0.0079216
9: 0.9013841, 0.9522769, 0.9020864, 0.9481564, -0.0307152, 0.0344469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0175257, upper bound: 0.0175856
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0182954, upper bound: 0.0175403
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0040929, -0.0011381, -0.0030436, 0.0030795
1: 0.0166869, 0.0319175, 0.0178103, 0.0316530, -0.0097167, 0.0089277
2: 0.0197779, 0.0302042, 0.0204141, 0.0299231, -0.0068659, 0.0064918
3: 0.0052423, 0.0170228, 0.0061951, 0.0167591, -0.0086206, 0.0079498
4: -0.0179484, -0.0063458, -0.0176166, -0.0072584, -0.0084040, 0.0089901
5: 0.0113442, 0.0259980, 0.0125631, 0.0256449, -0.0113708, 0.0106043
6: 0.0039356, 0.0149728, 0.0048292, 0.0146824, -0.0086021, 0.0080087
7: -0.0225430, -0.0104393, -0.0222527, -0.0114462, -0.0093189, 0.0099605
8: 0.0067857, 0.0186232, 0.0077319, 0.0183907, -0.0087101, 0.0080325
9: 0.9013770, 0.9526234, 0.9025822, 0.9486928, -0.0314890, 0.0343625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0182765, upper bound: 0.0177756
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191510, upper bound: 0.0179548
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042033, -0.0009539, -0.0040950, -0.0011180, -0.0030853, 0.0031411
1: 0.0166832, 0.0319700, 0.0180102, 0.0318097, -0.0099385, 0.0089359
2: 0.0198314, 0.0302691, 0.0205854, 0.0300432, -0.0069483, 0.0065049
3: 0.0052076, 0.0171186, 0.0062686, 0.0168785, -0.0087805, 0.0079090
4: -0.0179981, -0.0063460, -0.0177240, -0.0073628, -0.0083259, 0.0091024
5: 0.0112891, 0.0260852, 0.0126354, 0.0257630, -0.0115662, 0.0105819
6: 0.0039154, 0.0150324, 0.0048994, 0.0147838, -0.0087329, 0.0079583
7: -0.0225869, -0.0104013, -0.0223695, -0.0115517, -0.0092423, 0.0101311
8: 0.0067198, 0.0187207, 0.0077603, 0.0184851, -0.0088811, 0.0080422
9: 0.9010386, 0.9526324, 0.9020864, 0.9481564, -0.0312259, 0.0348657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0167734, upper bound: 0.0168891
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0167143, upper bound: 0.0165747
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042187, -0.0009181, -0.0040929, -0.0011381, -0.0030807, 0.0031748
1: 0.0165520, 0.0319708, 0.0178103, 0.0316530, -0.0099476, 0.0090624
2: 0.0197592, 0.0302962, 0.0204141, 0.0299231, -0.0069008, 0.0066767
3: 0.0051241, 0.0171198, 0.0061951, 0.0167591, -0.0087745, 0.0080715
4: -0.0180029, -0.0062789, -0.0176166, -0.0072584, -0.0084922, 0.0090679
5: 0.0111924, 0.0260893, 0.0125631, 0.0256449, -0.0115574, 0.0107338
6: 0.0038457, 0.0150357, 0.0048292, 0.0146824, -0.0087115, 0.0081074
7: -0.0225917, -0.0103089, -0.0222527, -0.0114462, -0.0093797, 0.0101207
8: 0.0066378, 0.0187207, 0.0077319, 0.0183907, -0.0089143, 0.0081529
9: 0.9010318, 0.9529719, 0.9025822, 0.9486928, -0.0319949, 0.0347777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173562
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041660, -0.0010444, -0.0041334, -0.0010424, -0.0031236, 0.0030890
1: 0.0168204, 0.0319168, 0.0178618, 0.0318816, -0.0099733, 0.0091038
2: 0.0198502, 0.0301768, 0.0205547, 0.0301407, -0.0071489, 0.0063751
3: 0.0053290, 0.0170216, 0.0061322, 0.0169652, -0.0087446, 0.0079685
4: -0.0179438, -0.0064135, -0.0177937, -0.0072913, -0.0083296, 0.0090885
5: 0.0114423, 0.0259942, 0.0124724, 0.0258513, -0.0114694, 0.0106604
6: 0.0040060, 0.0149697, 0.0048014, 0.0148554, -0.0086953, 0.0079876
7: -0.0225382, -0.0105332, -0.0224280, -0.0114140, -0.0093575, 0.0100240
8: 0.0068723, 0.0186232, 0.0075854, 0.0185713, -0.0088455, 0.0081617
9: 0.9013841, 0.9522769, 0.9017551, 0.9485164, -0.0312502, 0.0349703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0174100, upper bound: 0.0165990
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0167347, upper bound: 0.0165601
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0041301, -0.0010666, -0.0031150, 0.0031167
1: 0.0166869, 0.0319175, 0.0176665, 0.0317185, -0.0100085, 0.0092173
2: 0.0197779, 0.0302042, 0.0203836, 0.0300238, -0.0071062, 0.0065386
3: 0.0052423, 0.0170228, 0.0060571, 0.0168502, -0.0087515, 0.0081224
4: -0.0179484, -0.0063458, -0.0176928, -0.0071885, -0.0084949, 0.0090590
5: 0.0113442, 0.0259980, 0.0124001, 0.0257386, -0.0114757, 0.0108099
6: 0.0039356, 0.0149728, 0.0047325, 0.0147601, -0.0086798, 0.0081345
7: -0.0225430, -0.0104393, -0.0223166, -0.0113103, -0.0094949, 0.0100153
8: 0.0067857, 0.0186232, 0.0075582, 0.0184785, -0.0088755, 0.0082671
9: 0.9013770, 0.9526234, 0.9022505, 0.9490474, -0.0320088, 0.0349299

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0186859, upper bound: 0.0173539
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0185032, upper bound: 0.0173539
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0042033, -0.0009539, -0.0041334, -0.0010424, -0.0031609, 0.0031795
1: 0.0166832, 0.0319700, 0.0178618, 0.0318816, -0.0099827, 0.0091071
2: 0.0198314, 0.0302691, 0.0205547, 0.0301407, -0.0071071, 0.0065132
3: 0.0052076, 0.0171186, 0.0061322, 0.0169652, -0.0088039, 0.0080006
4: -0.0179981, -0.0063460, -0.0177937, -0.0072913, -0.0084002, 0.0091555
5: 0.0112891, 0.0260852, 0.0124724, 0.0258513, -0.0116083, 0.0107349
6: 0.0039154, 0.0150324, 0.0048014, 0.0148554, -0.0087716, 0.0080450
7: -0.0225869, -0.0104013, -0.0224280, -0.0114140, -0.0094224, 0.0101733
8: 0.0067198, 0.0187207, 0.0075854, 0.0185713, -0.0089030, 0.0081832
9: 0.9010386, 0.9526324, 0.9017551, 0.9485164, -0.0313897, 0.0350055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0167660, upper bound: 0.0168891
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0167122, upper bound: 0.0165600
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0042187, -0.0009181, -0.0041301, -0.0010666, -0.0031521, 0.0032120
1: 0.0165520, 0.0319708, 0.0176665, 0.0317185, -0.0099960, 0.0092245
2: 0.0197592, 0.0302962, 0.0203836, 0.0300238, -0.0070633, 0.0066837
3: 0.0051241, 0.0171198, 0.0060571, 0.0168502, -0.0088001, 0.0081636
4: -0.0180029, -0.0062789, -0.0176928, -0.0071885, -0.0085682, 0.0091209
5: 0.0111924, 0.0260893, 0.0124001, 0.0257386, -0.0116011, 0.0108890
6: 0.0038457, 0.0150357, 0.0047325, 0.0147601, -0.0087515, 0.0081933
7: -0.0225917, -0.0103089, -0.0223166, -0.0113103, -0.0095653, 0.0101609
8: 0.0066378, 0.0187207, 0.0075582, 0.0184785, -0.0089377, 0.0082953
9: 0.9010318, 0.9529719, 0.9022505, 0.9490474, -0.0321693, 0.0349221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173540
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0041796, -0.0009954, -0.0041660, -0.0010444, -0.0031352, 0.0031707
1: 0.0169396, 0.0320711, 0.0168204, 0.0319168, -0.0087464, 0.0090258
2: 0.0199922, 0.0303141, 0.0198502, 0.0301768, -0.0062326, 0.0065661
3: 0.0053648, 0.0171358, 0.0053290, 0.0170216, -0.0079463, 0.0081801
4: -0.0180430, -0.0064906, -0.0179438, -0.0064135, -0.0085768, 0.0083244
5: 0.0114660, 0.0261084, 0.0114423, 0.0259942, -0.0106658, 0.0108895
6: 0.0040463, 0.0150575, 0.0040060, 0.0149697, -0.0080109, 0.0082258
7: -0.0226535, -0.0105855, -0.0225382, -0.0105332, -0.0096014, 0.0093784
8: 0.0068574, 0.0187140, 0.0068723, 0.0186232, -0.0081334, 0.0082788
9: 0.9008898, 0.9519049, 0.9013841, 0.9522769, -0.0318876, 0.0307093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0181021, upper bound: 0.0174497
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0187325, upper bound: 0.0172721
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0041719, -0.0010331, -0.0041816, -0.0010133, -0.0031586, 0.0031486
1: 0.0167731, 0.0319171, 0.0166869, 0.0319175, -0.0088553, 0.0090220
2: 0.0198245, 0.0301870, 0.0197779, 0.0302042, -0.0064069, 0.0064982
3: 0.0052983, 0.0170220, 0.0052423, 0.0170228, -0.0080920, 0.0081659
4: -0.0179452, -0.0063906, -0.0179484, -0.0063458, -0.0085448, 0.0084841
5: 0.0114082, 0.0259954, 0.0113442, 0.0259980, -0.0107959, 0.0108791
6: 0.0039812, 0.0149707, 0.0039356, 0.0149728, -0.0081499, 0.0082059
7: -0.0225397, -0.0105001, -0.0225430, -0.0104393, -0.0095924, 0.0095009
8: 0.0068417, 0.0186232, 0.0067857, 0.0186232, -0.0082250, 0.0083028
9: 0.9013817, 0.9523923, 0.9013770, 0.9526234, -0.0317755, 0.0314450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0190101, upper bound: 0.0172568
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0191510, upper bound: 0.0179548
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0042168, -0.0009004, -0.0041660, -0.0010444, -0.0031725, 0.0032657
1: 0.0167986, 0.0321350, 0.0168204, 0.0319168, -0.0090175, 0.0092820
2: 0.0199715, 0.0304100, 0.0198502, 0.0301768, -0.0062807, 0.0067890
3: 0.0052387, 0.0172312, 0.0053290, 0.0170216, -0.0081157, 0.0083070
4: -0.0181021, -0.0064220, -0.0179438, -0.0064135, -0.0086456, 0.0084124
5: 0.0113158, 0.0262001, 0.0114423, 0.0259942, -0.0108613, 0.0109865
6: 0.0039559, 0.0151269, 0.0040060, 0.0149697, -0.0081313, 0.0083053
7: -0.0227067, -0.0104531, -0.0225382, -0.0105332, -0.0096553, 0.0095459
8: 0.0067000, 0.0188095, 0.0068723, 0.0186232, -0.0083605, 0.0084510
9: 0.9005122, 0.9522641, 0.9013841, 0.9522769, -0.0324111, 0.0312145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176977, upper bound: 0.0164987
time: 0.71 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176387, upper bound: 0.0159702
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0042091, -0.0009410, -0.0041816, -0.0010133, -0.0031957, 0.0032406
1: 0.0166393, 0.0319703, 0.0166869, 0.0319175, -0.0091256, 0.0093132
2: 0.0198059, 0.0302791, 0.0197779, 0.0302042, -0.0064525, 0.0067276
3: 0.0051798, 0.0171190, 0.0052423, 0.0170228, -0.0082550, 0.0082957
4: -0.0179996, -0.0063238, -0.0179484, -0.0063458, -0.0086157, 0.0085715
5: 0.0112560, 0.0260865, 0.0113442, 0.0259980, -0.0109926, 0.0109831
6: 0.0038914, 0.0150334, 0.0039356, 0.0149728, -0.0082674, 0.0082880
7: -0.0225884, -0.0103696, -0.0225430, -0.0104393, -0.0096479, 0.0096680
8: 0.0066936, 0.0187207, 0.0067857, 0.0186232, -0.0084461, 0.0084664
9: 0.9010364, 0.9527404, 0.9013770, 0.9526234, -0.0323296, 0.0319352

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173562
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041660, -0.0010444, -0.0042168, -0.0009004, -0.0032657, 0.0031725
1: 0.0168204, 0.0319168, 0.0167986, 0.0321350, -0.0092820, 0.0090175
2: 0.0198502, 0.0301768, 0.0199715, 0.0304100, -0.0067890, 0.0062807
3: 0.0053290, 0.0170216, 0.0052387, 0.0172312, -0.0083070, 0.0081157
4: -0.0179438, -0.0064135, -0.0181021, -0.0064220, -0.0084124, 0.0086456
5: 0.0114423, 0.0259942, 0.0113158, 0.0262001, -0.0109865, 0.0108613
6: 0.0040060, 0.0149697, 0.0039559, 0.0151269, -0.0083053, 0.0081313
7: -0.0225382, -0.0105332, -0.0227067, -0.0104531, -0.0095459, 0.0096553
8: 0.0068723, 0.0186232, 0.0067000, 0.0188095, -0.0084510, 0.0083605
9: 0.9013841, 0.9522769, 0.9005122, 0.9522641, -0.0312146, 0.0324111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0174100, upper bound: 0.0165990
time: 0.68 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0167347, upper bound: 0.0165601
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0042091, -0.0009410, -0.0032406, 0.0031957
1: 0.0166869, 0.0319175, 0.0166393, 0.0319703, -0.0093132, 0.0091256
2: 0.0197779, 0.0302042, 0.0198059, 0.0302791, -0.0067276, 0.0064525
3: 0.0052423, 0.0170228, 0.0051798, 0.0171190, -0.0082957, 0.0082550
4: -0.0179484, -0.0063458, -0.0179996, -0.0063238, -0.0085715, 0.0086157
5: 0.0113442, 0.0259980, 0.0112560, 0.0260865, -0.0109831, 0.0109926
6: 0.0039356, 0.0149728, 0.0038914, 0.0150334, -0.0082880, 0.0082674
7: -0.0225430, -0.0104393, -0.0225884, -0.0103696, -0.0096680, 0.0096479
8: 0.0067857, 0.0186232, 0.0066936, 0.0187207, -0.0084664, 0.0084461
9: 0.9013770, 0.9526234, 0.9010364, 0.9527404, -0.0319352, 0.0323296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0186859, upper bound: 0.0173539
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0185032, upper bound: 0.0173539
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0042168, -0.0009004, -0.0042033, -0.0009539, -0.0032629, 0.0033029
1: 0.0167986, 0.0321350, 0.0166832, 0.0319700, -0.0090464, 0.0093241
2: 0.0199715, 0.0304100, 0.0198314, 0.0302691, -0.0064357, 0.0067666
3: 0.0052387, 0.0172312, 0.0052076, 0.0171186, -0.0081427, 0.0083776
4: -0.0181021, -0.0064220, -0.0179981, -0.0063460, -0.0087271, 0.0084720
5: 0.0113158, 0.0262001, 0.0112891, 0.0260852, -0.0109197, 0.0111504
6: 0.0039559, 0.0151269, 0.0039154, 0.0150324, -0.0081835, 0.0084000
7: -0.0227067, -0.0104531, -0.0225869, -0.0104013, -0.0098272, 0.0095968
8: 0.0067000, 0.0188095, 0.0067198, 0.0187207, -0.0083727, 0.0085219
9: 0.9005122, 0.9522641, 0.9010386, 0.9526324, -0.0324841, 0.0313065

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0180083, upper bound: 0.0160331
time: 0.73 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0176378, upper bound: 0.0159555
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0042091, -0.0009410, -0.0042187, -0.0009181, -0.0032910, 0.0032777
1: 0.0166393, 0.0319703, 0.0165520, 0.0319708, -0.0091569, 0.0093247
2: 0.0198059, 0.0302791, 0.0197592, 0.0302962, -0.0066111, 0.0066976
3: 0.0051798, 0.0171190, 0.0051241, 0.0171198, -0.0082907, 0.0083645
4: -0.0179996, -0.0063238, -0.0180029, -0.0062789, -0.0086949, 0.0086341
5: 0.0112560, 0.0260865, 0.0111924, 0.0260893, -0.0110562, 0.0111399
6: 0.0038914, 0.0150334, 0.0038457, 0.0150357, -0.0083230, 0.0083791
7: -0.0225884, -0.0103696, -0.0225917, -0.0103089, -0.0098164, 0.0097250
8: 0.0066936, 0.0187207, 0.0066378, 0.0187207, -0.0084681, 0.0085473
9: 0.9010364, 0.9527404, 0.9010318, 0.9529719, -0.0323789, 0.0320478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B2_A2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
time: 0.73 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173540
time: 0.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 2.88 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0171883, upper bound: 0.0175610
IS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0177182, upper bound: 0.0173856
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0179775, upper bound: 0.0174208
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0181550, upper bound: 0.0181550
IS_A1_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0165393, upper bound: 0.0167171
IS_A1_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0159813, upper bound: 0.0166630
IS_A1_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0176707, upper bound: 0.0175101
IS_A1_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175323, upper bound: 0.0175101
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0167171, upper bound: 0.0165393
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0166630, upper bound: 0.0159813
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175101, upper bound: 0.0176707
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175101, upper bound: 0.0175323
IS_A1_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0160533, upper bound: 0.0170270
IS_A1_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0159679, upper bound: 0.0166630
IS_A1_B1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175756, upper bound: 0.0175214
IS_A1_B1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175101, upper bound: 0.0175214
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175856, upper bound: 0.0175257
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175403, upper bound: 0.0182954
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0177756, upper bound: 0.0182765
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0179548, upper bound: 0.0191510
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0168891, upper bound: 0.0167734
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0165747, upper bound: 0.0167143
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0174846, upper bound: 0.0184965
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0173562, upper bound: 0.0184965
IS_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0165990, upper bound: 0.0174100
IS_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0165601, upper bound: 0.0167347
IS_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0173539, upper bound: 0.0186859
IS_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0173539, upper bound: 0.0185032
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0168582, upper bound: 0.0167660
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0165600, upper bound: 0.0167122
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0174345, upper bound: 0.0184975
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0173539, upper bound: 0.0184975
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0175257, upper bound: 0.0175856
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0182954, upper bound: 0.0175403
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0182765, upper bound: 0.0177756
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0191510, upper bound: 0.0179548
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0167734, upper bound: 0.0168891
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0167143, upper bound: 0.0165747
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173562
IS_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0174100, upper bound: 0.0165990
IS_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0167347, upper bound: 0.0165601
IS_A2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0186859, upper bound: 0.0173539
IS_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0185032, upper bound: 0.0173539
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0167660, upper bound: 0.0168891
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0167122, upper bound: 0.0165600
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173540
IS_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0181021, upper bound: 0.0174497
IS_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0187325, upper bound: 0.0172721
IS_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0190101, upper bound: 0.0172568
IS_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0191510, upper bound: 0.0179548
IS_A2_B2_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0176977, upper bound: 0.0164987
IS_A2_B2_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0176387, upper bound: 0.0159702
IS_A2_B2_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
IS_A2_B2_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173562
IS_A2_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0174100, upper bound: 0.0165990
IS_A2_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0167347, upper bound: 0.0165601
IS_A2_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0186859, upper bound: 0.0173539
IS_A2_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0185032, upper bound: 0.0173539
IS_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0180083, upper bound: 0.0160331
IS_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0176378, upper bound: 0.0159555
IS_A2_B2_B2_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0174846
IS_A2_B2_B2_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 2.88
Output dim: 9, lower bound: -0.0184965, upper bound: 0.0173540

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0037531, -0.0014536, -0.0040193, -0.0012104, -0.0025427, 0.0025658
1: 0.0191233, 0.0319055, 0.0180629, 0.0316527, -0.0061972, 0.0079932
2: 0.0207164, 0.0294683, 0.0204661, 0.0297985, -0.0052662, 0.0050386
3: 0.0071643, 0.0171908, 0.0063948, 0.0167543, -0.0058547, 0.0069849
4: -0.0178486, -0.0077817, -0.0175979, -0.0073644, -0.0073963, 0.0070234
5: 0.0137739, 0.0259752, 0.0128097, 0.0256280, -0.0081358, 0.0093686
6: 0.0055729, 0.0149748, 0.0049815, 0.0146696, -0.0063314, 0.0070062
7: -0.0224668, -0.0125670, -0.0222331, -0.0116745, -0.0082150, 0.0072363
8: 0.0089621, 0.0188062, 0.0079845, 0.0183907, -0.0056591, 0.0072149
9: 0.9010233, 0.9460711, 0.9026111, 0.9481710, -0.0278234, 0.0245714

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145575, upper bound: 0.0148610
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0141975
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0039902, -0.0012034, -0.0040693, -0.0011639, -0.0028262, 0.0028659
1: 0.0186371, 0.0318097, 0.0179511, 0.0316527, -0.0074474, 0.0082272
2: 0.0208021, 0.0299066, 0.0204694, 0.0298889, -0.0055431, 0.0057703
3: 0.0070426, 0.0168759, 0.0063310, 0.0167582, -0.0059701, 0.0075950
4: -0.0177137, -0.0079080, -0.0176132, -0.0073551, -0.0081548, 0.0067230
5: 0.0134865, 0.0257539, 0.0127167, 0.0256418, -0.0085363, 0.0102056
6: 0.0055432, 0.0147769, 0.0049417, 0.0146801, -0.0062228, 0.0077545
7: -0.0223589, -0.0122311, -0.0222492, -0.0115763, -0.0090680, 0.0076704
8: 0.0086056, 0.0184851, 0.0078818, 0.0183907, -0.0063212, 0.0075864
9: 0.9021025, 0.9456860, 0.9025875, 0.9482280, -0.0297255, 0.0240718

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151263, upper bound: 0.0141871
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148435, upper bound: 0.0140992
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039513, -0.0012484, -0.0036952, -0.0013653, -0.0025860, 0.0024467
1: 0.0182448, 0.0316530, 0.0189931, 0.0317726, -0.0074643, 0.0061811
2: 0.0204716, 0.0296964, 0.0205489, 0.0293141, -0.0047390, 0.0052550
3: 0.0065657, 0.0167496, 0.0072348, 0.0171822, -0.0066729, 0.0057586
4: -0.0175795, -0.0074428, -0.0178438, -0.0077787, -0.0070415, 0.0072504
5: 0.0130288, 0.0256112, 0.0138806, 0.0259592, -0.0090029, 0.0079926
6: 0.0051114, 0.0146572, 0.0056247, 0.0149805, -0.0067868, 0.0062712
7: -0.0222138, -0.0118633, -0.0224293, -0.0126310, -0.0071487, 0.0078938
8: 0.0082168, 0.0183907, 0.0090965, 0.0188012, -0.0068393, 0.0054881
9: 0.9026397, 0.9478173, 0.9011671, 0.9462265, -0.0246959, 0.0268653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152647
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152410
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0039966, -0.0012092, -0.0028837, 0.0028586
1: 0.0178103, 0.0316530, 0.0182774, 0.0316534, -0.0081407, 0.0077708
2: 0.0204141, 0.0299231, 0.0205461, 0.0297974, -0.0056334, 0.0058717
3: 0.0061951, 0.0167591, 0.0067967, 0.0167568, -0.0075592, 0.0062151
4: -0.0176166, -0.0072584, -0.0176080, -0.0076590, -0.0069716, 0.0080782
5: 0.0125631, 0.0256449, 0.0132203, 0.0256370, -0.0101686, 0.0087868
6: 0.0048292, 0.0146824, 0.0053210, 0.0146766, -0.0077057, 0.0064429
7: -0.0222527, -0.0114462, -0.0222437, -0.0119716, -0.0079163, 0.0090145
8: 0.0077319, 0.0183907, 0.0084095, 0.0183907, -0.0076026, 0.0065094
9: 0.9025822, 0.9486928, 0.9025958, 0.9468501, -0.0252737, 0.0294354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160988, upper bound: 0.0161597
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160988, upper bound: 0.0160988
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040865, -0.0011470, -0.0039229, -0.0012480, -0.0028385, 0.0027759
1: 0.0178614, 0.0316527, 0.0185970, 0.0318816, -0.0085177, 0.0071419
2: 0.0204404, 0.0299116, 0.0207043, 0.0297991, -0.0056848, 0.0055988
3: 0.0062230, 0.0167586, 0.0067706, 0.0169469, -0.0077502, 0.0067408
4: -0.0176149, -0.0072804, -0.0177230, -0.0076762, -0.0074946, 0.0082228
5: 0.0125946, 0.0256433, 0.0133270, 0.0257886, -0.0103367, 0.0090980
6: 0.0048514, 0.0146812, 0.0053205, 0.0148050, -0.0078434, 0.0070080
7: -0.0222509, -0.0114797, -0.0223524, -0.0121252, -0.0080784, 0.0091562
8: 0.0077587, 0.0183907, 0.0084368, 0.0185713, -0.0078176, 0.0066488
9: 0.9025850, 0.9485730, 0.9018676, 0.9468139, -0.0269011, 0.0303197

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138747, upper bound: 0.0148278
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145360, upper bound: 0.0148031
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040377, -0.0011925, -0.0038799, -0.0007271, -0.0033106, 0.0026874
1: 0.0181046, 0.0316527, 0.0190932, 0.0319850, -0.0084295, 0.0068294
2: 0.0205399, 0.0298361, 0.0212186, 0.0300048, -0.0058034, 0.0053053
3: 0.0064463, 0.0167540, 0.0074311, 0.0174914, -0.0084890, 0.0065502
4: -0.0175966, -0.0074780, -0.0185093, -0.0084806, -0.0072906, 0.0094355
5: 0.0128980, 0.0256267, 0.0141970, 0.0265745, -0.0114242, 0.0088459
6: 0.0050616, 0.0146687, 0.0060287, 0.0154904, -0.0088412, 0.0068218
7: -0.0222317, -0.0117537, -0.0229394, -0.0129332, -0.0078311, 0.0101129
8: 0.0080083, 0.0183907, 0.0089897, 0.0190317, -0.0082722, 0.0064644
9: 0.9026133, 0.9478071, 0.8998247, 0.9437917, -0.0260781, 0.0335994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041024, -0.0011228, -0.0039175, -0.0012568, -0.0028456, 0.0027947
1: 0.0177250, 0.0316534, 0.0184070, 0.0317185, -0.0085669, 0.0072560
2: 0.0203672, 0.0299409, 0.0205439, 0.0296770, -0.0056493, 0.0057670
3: 0.0061374, 0.0167598, 0.0067016, 0.0168319, -0.0077593, 0.0068890
4: -0.0176199, -0.0072131, -0.0176213, -0.0075809, -0.0076531, 0.0081966
5: 0.0124980, 0.0256477, 0.0132559, 0.0256738, -0.0103446, 0.0092374
6: 0.0047825, 0.0146847, 0.0052512, 0.0147106, -0.0078301, 0.0071472
7: -0.0222562, -0.0113836, -0.0222407, -0.0120308, -0.0082117, 0.0091490
8: 0.0076742, 0.0183907, 0.0084054, 0.0184785, -0.0078503, 0.0067464
9: 0.9025774, 0.9489240, 0.9023612, 0.9473296, -0.0276342, 0.0302994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152527, upper bound: 0.0156279
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161630, upper bound: 0.0160080
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040532, -0.0011783, -0.0038730, -0.0007367, -0.0033165, 0.0026947
1: 0.0179724, 0.0316534, 0.0188819, 0.0318254, -0.0084742, 0.0069475
2: 0.0204672, 0.0298642, 0.0210367, 0.0298818, -0.0057891, 0.0054803
3: 0.0063648, 0.0167552, 0.0073453, 0.0173764, -0.0084853, 0.0067099
4: -0.0176015, -0.0074123, -0.0184107, -0.0083567, -0.0074580, 0.0093920
5: 0.0128037, 0.0256310, 0.0141068, 0.0264618, -0.0114058, 0.0089920
6: 0.0049949, 0.0146720, 0.0059454, 0.0153905, -0.0088135, 0.0069688
7: -0.0222368, -0.0116612, -0.0228296, -0.0128086, -0.0079705, 0.0100900
8: 0.0079268, 0.0183907, 0.0089450, 0.0189295, -0.0082897, 0.0065701
9: 0.9026060, 0.9481464, 0.9003108, 0.9443833, -0.0268488, 0.0334944

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152076, upper bound: 0.0156279
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160300, upper bound: 0.0160080
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0039229, -0.0012480, -0.0040865, -0.0011470, -0.0027759, 0.0028385
1: 0.0185970, 0.0318816, 0.0178614, 0.0316527, -0.0071419, 0.0085177
2: 0.0207043, 0.0297991, 0.0204404, 0.0299116, -0.0055988, 0.0056848
3: 0.0067706, 0.0169469, 0.0062230, 0.0167586, -0.0067408, 0.0077502
4: -0.0177230, -0.0076762, -0.0176149, -0.0072804, -0.0082228, 0.0074946
5: 0.0133270, 0.0257886, 0.0125946, 0.0256433, -0.0090980, 0.0103368
6: 0.0053205, 0.0148050, 0.0048514, 0.0146812, -0.0070080, 0.0078433
7: -0.0223524, -0.0121252, -0.0222509, -0.0114797, -0.0091562, 0.0080784
8: 0.0084368, 0.0185713, 0.0077587, 0.0183907, -0.0066488, 0.0078176
9: 0.9018676, 0.9468139, 0.9025850, 0.9485730, -0.0303197, 0.0269011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148278, upper bound: 0.0138747
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148031, upper bound: 0.0145360
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0038799, -0.0007271, -0.0040377, -0.0011925, -0.0026874, 0.0033106
1: 0.0190932, 0.0319850, 0.0181046, 0.0316527, -0.0068294, 0.0084295
2: 0.0212186, 0.0300048, 0.0205399, 0.0298361, -0.0053053, 0.0058034
3: 0.0074311, 0.0174914, 0.0064463, 0.0167540, -0.0065502, 0.0084890
4: -0.0185093, -0.0084806, -0.0175966, -0.0074780, -0.0094355, 0.0072906
5: 0.0141970, 0.0265745, 0.0128980, 0.0256267, -0.0088459, 0.0114242
6: 0.0060287, 0.0154904, 0.0050616, 0.0146687, -0.0068218, 0.0088412
7: -0.0229394, -0.0129332, -0.0222317, -0.0117537, -0.0101129, 0.0078311
8: 0.0089897, 0.0190317, 0.0080083, 0.0183907, -0.0064644, 0.0082722
9: 0.8998247, 0.9437917, 0.9026132, 0.9478070, -0.0335994, 0.0260781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0039175, -0.0012568, -0.0041024, -0.0011228, -0.0027947, 0.0028456
1: 0.0184070, 0.0317185, 0.0177250, 0.0316534, -0.0072560, 0.0085669
2: 0.0205439, 0.0296770, 0.0203672, 0.0299409, -0.0057670, 0.0056493
3: 0.0067016, 0.0168319, 0.0061374, 0.0167598, -0.0068890, 0.0077593
4: -0.0176213, -0.0075809, -0.0176199, -0.0072131, -0.0081966, 0.0076531
5: 0.0132559, 0.0256738, 0.0124980, 0.0256477, -0.0092374, 0.0103446
6: 0.0052512, 0.0147106, 0.0047825, 0.0146847, -0.0071472, 0.0078301
7: -0.0222407, -0.0120308, -0.0222562, -0.0113836, -0.0091490, 0.0082117
8: 0.0084054, 0.0184785, 0.0076742, 0.0183907, -0.0067464, 0.0078503
9: 0.9023612, 0.9473296, 0.9025774, 0.9489240, -0.0302995, 0.0276342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152527
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0161630
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0038730, -0.0007367, -0.0040532, -0.0011783, -0.0026947, 0.0033165
1: 0.0188819, 0.0318254, 0.0179723, 0.0316534, -0.0069475, 0.0084742
2: 0.0210367, 0.0298818, 0.0204672, 0.0298642, -0.0054803, 0.0057891
3: 0.0073453, 0.0173764, 0.0063648, 0.0167552, -0.0067099, 0.0084853
4: -0.0184107, -0.0083567, -0.0176015, -0.0074123, -0.0093920, 0.0074580
5: 0.0141068, 0.0264618, 0.0128037, 0.0256310, -0.0089920, 0.0114058
6: 0.0059454, 0.0153905, 0.0049949, 0.0146720, -0.0069688, 0.0088135
7: -0.0228296, -0.0128086, -0.0222368, -0.0116612, -0.0100900, 0.0079705
8: 0.0089450, 0.0189295, 0.0079268, 0.0183907, -0.0065701, 0.0082897
9: 0.9003108, 0.9443833, 0.9026060, 0.9481464, -0.0334944, 0.0268488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152076
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0160300
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039113, -0.0012590, -0.0041334, -0.0010424, -0.0028689, 0.0028744
1: 0.0184532, 0.0317182, 0.0178618, 0.0318816, -0.0073797, 0.0082990
2: 0.0205670, 0.0296673, 0.0205547, 0.0301407, -0.0060417, 0.0053434
3: 0.0067300, 0.0168314, 0.0061322, 0.0169652, -0.0069493, 0.0076047
4: -0.0176194, -0.0076024, -0.0177937, -0.0072913, -0.0080646, 0.0077808
5: 0.0132893, 0.0256722, 0.0124724, 0.0258513, -0.0093462, 0.0102866
6: 0.0052751, 0.0147093, 0.0048014, 0.0148554, -0.0072414, 0.0077340
7: -0.0222387, -0.0120623, -0.0224280, -0.0114140, -0.0091077, 0.0083466
8: 0.0084345, 0.0184785, 0.0075854, 0.0185713, -0.0067650, 0.0077546
9: 0.9023639, 0.9472150, 0.9017551, 0.9485164, -0.0293048, 0.0279644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141666, upper bound: 0.0146558
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140765, upper bound: 0.0151040
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038671, -0.0007374, -0.0040380, -0.0011736, -0.0026935, 0.0033005
1: 0.0189299, 0.0318250, 0.0183975, 0.0318816, -0.0071429, 0.0080145
2: 0.0210653, 0.0298724, 0.0207701, 0.0299851, -0.0056429, 0.0055080
3: 0.0073738, 0.0173759, 0.0066198, 0.0169562, -0.0068104, 0.0083143
4: -0.0184090, -0.0083828, -0.0177587, -0.0076988, -0.0092629, 0.0075845
5: 0.0141427, 0.0264602, 0.0131168, 0.0258200, -0.0091177, 0.0112557
6: 0.0059706, 0.0153892, 0.0052454, 0.0148302, -0.0070751, 0.0086925
7: -0.0228279, -0.0128462, -0.0223905, -0.0119826, -0.0099365, 0.0081052
8: 0.0089742, 0.0189295, 0.0081314, 0.0185713, -0.0066431, 0.0081278
9: 0.9003136, 0.9442567, 0.9018110, 0.9468588, -0.0326510, 0.0273610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140835, upper bound: 0.0144218
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041397, -0.0010483, -0.0039175, -0.0012568, -0.0028829, 0.0028692
1: 0.0175815, 0.0317189, 0.0184070, 0.0317185, -0.0085956, 0.0072267
2: 0.0203374, 0.0300414, 0.0205439, 0.0296770, -0.0056055, 0.0059083
3: 0.0060007, 0.0168510, 0.0067016, 0.0168319, -0.0078427, 0.0068755
4: -0.0176961, -0.0071435, -0.0176213, -0.0075809, -0.0076893, 0.0082936
5: 0.0123361, 0.0257415, 0.0132559, 0.0256738, -0.0105251, 0.0092600
6: 0.0046862, 0.0147623, 0.0052512, 0.0147106, -0.0079388, 0.0071670
7: -0.0223200, -0.0112477, -0.0222407, -0.0120308, -0.0082494, 0.0093417
8: 0.0075015, 0.0184785, 0.0084054, 0.0184785, -0.0079490, 0.0067178
9: 0.9022455, 0.9492757, 0.9023612, 0.9473296, -0.0275738, 0.0304096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152357, upper bound: 0.0156406
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160932, upper bound: 0.0160120
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0040390, -0.0011787, -0.0038730, -0.0007367, -0.0033023, 0.0026942
1: 0.0181217, 0.0317189, 0.0188819, 0.0318254, -0.0082930, 0.0069830
2: 0.0205511, 0.0298796, 0.0210367, 0.0298818, -0.0057546, 0.0055012
3: 0.0064951, 0.0168415, 0.0073453, 0.0173764, -0.0085390, 0.0067304
4: -0.0176588, -0.0075544, -0.0184107, -0.0083567, -0.0074881, 0.0094831
5: 0.0129846, 0.0257077, 0.0141068, 0.0264618, -0.0114829, 0.0090239
6: 0.0051331, 0.0147366, 0.0059454, 0.0153905, -0.0088911, 0.0069946
7: -0.0222805, -0.0118204, -0.0228296, -0.0128086, -0.0080066, 0.0101602
8: 0.0080526, 0.0184785, 0.0089450, 0.0189295, -0.0083091, 0.0065906
9: 0.9023031, 0.9476138, 0.9003108, 0.9443833, -0.0269434, 0.0337178

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152067, upper bound: 0.0156406
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0160120
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040213, -0.0012021, -0.0037763, -0.0013821, -0.0026392, 0.0025742
1: 0.0182399, 0.0318097, 0.0180819, 0.0320361, -0.0084347, 0.0078871
2: 0.0206130, 0.0299110, 0.0200002, 0.0295901, -0.0053079, 0.0064186
3: 0.0064615, 0.0168736, 0.0063578, 0.0174726, -0.0073983, 0.0070898
4: -0.0177049, -0.0074526, -0.0181796, -0.0069277, -0.0081569, 0.0077798
5: 0.0128772, 0.0257460, 0.0127861, 0.0263298, -0.0098658, 0.0094448
6: 0.0050432, 0.0147708, 0.0048096, 0.0152701, -0.0073922, 0.0073926
7: -0.0223496, -0.0117647, -0.0227282, -0.0117322, -0.0082342, 0.0085508
8: 0.0080172, 0.0184851, 0.0082174, 0.0190545, -0.0075932, 0.0067906
9: 0.9021165, 0.9477148, 0.8998722, 0.9498303, -0.0305709, 0.0296687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151044, upper bound: 0.0143119
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147146, upper bound: 0.0141498
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040875, -0.0011261, -0.0040582, -0.0011684, -0.0029191, 0.0029321
1: 0.0180513, 0.0318097, 0.0174437, 0.0319168, -0.0087377, 0.0090997
2: 0.0205986, 0.0300332, 0.0200498, 0.0300308, -0.0060465, 0.0067118
3: 0.0063188, 0.0168783, 0.0060326, 0.0170190, -0.0077680, 0.0072518
4: -0.0177232, -0.0073970, -0.0179337, -0.0068965, -0.0078485, 0.0082288
5: 0.0126911, 0.0257623, 0.0122186, 0.0259860, -0.0104267, 0.0098988
6: 0.0049405, 0.0147833, 0.0045791, 0.0149628, -0.0078452, 0.0073062
7: -0.0223687, -0.0115963, -0.0225280, -0.0111824, -0.0087387, 0.0091584
8: 0.0078157, 0.0184851, 0.0076451, 0.0186232, -0.0078802, 0.0075334
9: 0.9020876, 0.9479977, 0.9014003, 0.9500028, -0.0301606, 0.0306557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147599, upper bound: 0.0155186
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147207, upper bound: 0.0148894
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039885, -0.0012303, -0.0037817, -0.0013522, -0.0026363, 0.0025514
1: 0.0181288, 0.0316530, 0.0179879, 0.0320369, -0.0084006, 0.0078401
2: 0.0204560, 0.0297529, 0.0199461, 0.0296071, -0.0054289, 0.0062742
3: 0.0064672, 0.0167523, 0.0063291, 0.0174736, -0.0074292, 0.0070081
4: -0.0175898, -0.0073929, -0.0181868, -0.0069071, -0.0080760, 0.0078858
5: 0.0129028, 0.0256206, 0.0127493, 0.0263337, -0.0098728, 0.0093652
6: 0.0050356, 0.0146642, 0.0047853, 0.0152778, -0.0074414, 0.0073125
7: -0.0222247, -0.0117531, -0.0227331, -0.0117005, -0.0081735, 0.0085818
8: 0.0080857, 0.0183907, 0.0081835, 0.0190545, -0.0075463, 0.0067431
9: 0.9026237, 0.9480532, 0.8998659, 0.9499702, -0.0302322, 0.0300898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156764, upper bound: 0.0160950
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155044, upper bound: 0.0160950
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0040643, -0.0011620, -0.0029309, 0.0029263
1: 0.0178103, 0.0316530, 0.0173192, 0.0319175, -0.0088875, 0.0091002
2: 0.0204141, 0.0299231, 0.0199713, 0.0300403, -0.0061893, 0.0067018
3: 0.0061951, 0.0167591, 0.0059393, 0.0170196, -0.0079485, 0.0072255
4: -0.0176166, -0.0072584, -0.0179360, -0.0068119, -0.0078335, 0.0084006
5: 0.0125631, 0.0256449, 0.0121168, 0.0259878, -0.0106007, 0.0098842
6: 0.0048292, 0.0146824, 0.0044993, 0.0149644, -0.0080059, 0.0072903
7: -0.0222527, -0.0114462, -0.0225303, -0.0110853, -0.0087308, 0.0093158
8: 0.0077319, 0.0183907, 0.0075605, 0.0186232, -0.0080194, 0.0075099
9: 0.9025822, 0.9486928, 0.9013969, 0.9504046, -0.0300720, 0.0314804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0171563
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0170393
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0040950, -0.0011180, -0.0039965, -0.0012290, -0.0028660, 0.0028785
1: 0.0180102, 0.0318097, 0.0174309, 0.0319700, -0.0088578, 0.0089063
2: 0.0205854, 0.0300432, 0.0199892, 0.0299374, -0.0059437, 0.0067723
3: 0.0062686, 0.0168785, 0.0058399, 0.0171019, -0.0079018, 0.0080492
4: -0.0177240, -0.0073628, -0.0179321, -0.0067604, -0.0086640, 0.0083029
5: 0.0126354, 0.0257630, 0.0121674, 0.0260261, -0.0105602, 0.0105541
6: 0.0048994, 0.0147838, 0.0044545, 0.0149842, -0.0079415, 0.0081303
7: -0.0223695, -0.0115517, -0.0225180, -0.0111404, -0.0092564, 0.0092235
8: 0.0077603, 0.0184851, 0.0075324, 0.0187207, -0.0080267, 0.0079174
9: 0.9020864, 0.9481564, 0.9011388, 0.9508425, -0.0329671, 0.0311831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145409, upper bound: 0.0148680
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149400, upper bound: 0.0148081
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040472, -0.0011748, -0.0039194, -0.0007418, -0.0033054, 0.0027446
1: 0.0182576, 0.0318097, 0.0180084, 0.0320908, -0.0087648, 0.0084830
2: 0.0206857, 0.0299679, 0.0204835, 0.0301331, -0.0060236, 0.0063893
3: 0.0064943, 0.0168740, 0.0066032, 0.0175991, -0.0084405, 0.0077065
4: -0.0177065, -0.0075584, -0.0187164, -0.0075759, -0.0082802, 0.0092486
5: 0.0129445, 0.0257472, 0.0131342, 0.0267923, -0.0113326, 0.0100361
6: 0.0051109, 0.0147718, 0.0052162, 0.0156573, -0.0087004, 0.0077716
7: -0.0223511, -0.0118262, -0.0231297, -0.0120178, -0.0087668, 0.0099671
8: 0.0080123, 0.0184851, 0.0082220, 0.0191380, -0.0083387, 0.0075732
9: 0.9021142, 0.9473943, 0.8990721, 0.9477046, -0.0316369, 0.0335165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143276, upper bound: 0.0147842
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0040113, -0.0012185, -0.0028744, 0.0028732
1: 0.0178103, 0.0316530, 0.0173008, 0.0319708, -0.0089823, 0.0089130
2: 0.0204141, 0.0299231, 0.0199189, 0.0299590, -0.0061035, 0.0067233
3: 0.0061951, 0.0167591, 0.0057589, 0.0171031, -0.0080643, 0.0080396
4: -0.0176166, -0.0072584, -0.0179369, -0.0066953, -0.0086266, 0.0084689
5: 0.0125631, 0.0256449, 0.0120718, 0.0260303, -0.0107121, 0.0105415
6: 0.0048292, 0.0146824, 0.0043865, 0.0149877, -0.0080906, 0.0081043
7: -0.0222527, -0.0114462, -0.0225231, -0.0110487, -0.0092434, 0.0093602
8: 0.0077319, 0.0183907, 0.0074503, 0.0187207, -0.0081374, 0.0079484
9: 0.9025822, 0.9486928, 0.9011317, 0.9511734, -0.0328653, 0.0319520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157059, upper bound: 0.0160780
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159484, upper bound: 0.0169337
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040448, -0.0011867, -0.0039342, -0.0007390, -0.0033058, 0.0027475
1: 0.0180511, 0.0316530, 0.0178741, 0.0320916, -0.0088735, 0.0084896
2: 0.0205105, 0.0298485, 0.0204091, 0.0301557, -0.0061666, 0.0063393
3: 0.0064163, 0.0167545, 0.0065163, 0.0176003, -0.0086025, 0.0076951
4: -0.0175987, -0.0074536, -0.0187208, -0.0075050, -0.0082448, 0.0094201
5: 0.0128616, 0.0256285, 0.0130327, 0.0267961, -0.0114890, 0.0100264
6: 0.0050366, 0.0146701, 0.0051435, 0.0156603, -0.0088513, 0.0077466
7: -0.0222338, -0.0117165, -0.0231343, -0.0119207, -0.0087561, 0.0101139
8: 0.0079778, 0.0183907, 0.0081343, 0.0191380, -0.0084503, 0.0076004
9: 0.9026101, 0.9479352, 0.8990651, 0.9480608, -0.0315486, 0.0342689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150554, upper bound: 0.0166586
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158301, upper bound: 0.0169337
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0039229, -0.0012480, -0.0041660, -0.0010444, -0.0028786, 0.0029180
1: 0.0185970, 0.0318816, 0.0168204, 0.0319168, -0.0079002, 0.0098869
2: 0.0207043, 0.0297991, 0.0198502, 0.0301768, -0.0061564, 0.0065184
3: 0.0067706, 0.0169469, 0.0053290, 0.0170216, -0.0071306, 0.0087392
4: -0.0177230, -0.0076762, -0.0179438, -0.0064135, -0.0090774, 0.0078201
5: 0.0133270, 0.0257886, 0.0114423, 0.0259942, -0.0095314, 0.0114563
6: 0.0053205, 0.0148050, 0.0040060, 0.0149697, -0.0073094, 0.0086859
7: -0.0223524, -0.0121252, -0.0225382, -0.0105332, -0.0100160, 0.0083860
8: 0.0084368, 0.0185713, 0.0068723, 0.0186232, -0.0070638, 0.0088293
9: 0.9018676, 0.9468139, 0.9013841, 0.9522769, -0.0349322, 0.0289454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147293, upper bound: 0.0146145
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146669, upper bound: 0.0154260
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0038799, -0.0007271, -0.0041197, -0.0011126, -0.0027673, 0.0033925
1: 0.0190932, 0.0319850, 0.0170546, 0.0319168, -0.0075612, 0.0098018
2: 0.0212186, 0.0300048, 0.0199383, 0.0301097, -0.0058661, 0.0066225
3: 0.0074311, 0.0174914, 0.0055455, 0.0170175, -0.0069037, 0.0094635
4: -0.0185093, -0.0084806, -0.0179279, -0.0065862, -0.0102679, 0.0075874
5: 0.0141970, 0.0265745, 0.0117266, 0.0259810, -0.0092420, 0.0125221
6: 0.0060287, 0.0154904, 0.0041981, 0.0149587, -0.0070933, 0.0096611
7: -0.0229394, -0.0129332, -0.0225219, -0.0107785, -0.0109579, 0.0081152
8: 0.0089897, 0.0190317, 0.0071120, 0.0186232, -0.0068410, 0.0092690
9: 0.8998247, 0.9437917, 0.9014099, 0.9515868, -0.0381107, 0.0279633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0039175, -0.0012568, -0.0041816, -0.0010133, -0.0029042, 0.0029248
1: 0.0184070, 0.0317185, 0.0166869, 0.0319175, -0.0080037, 0.0099290
2: 0.0205439, 0.0296770, 0.0197779, 0.0302042, -0.0063169, 0.0064793
3: 0.0067016, 0.0168319, 0.0052423, 0.0170228, -0.0072790, 0.0087455
4: -0.0176213, -0.0075809, -0.0179484, -0.0063458, -0.0090474, 0.0079774
5: 0.0132559, 0.0256738, 0.0113442, 0.0259980, -0.0096712, 0.0114603
6: 0.0052512, 0.0147106, 0.0039356, 0.0149728, -0.0074490, 0.0086686
7: -0.0222407, -0.0120308, -0.0225430, -0.0104393, -0.0100048, 0.0085152
8: 0.0084054, 0.0184785, 0.0067857, 0.0186232, -0.0071616, 0.0088606
9: 0.9023612, 0.9473296, 0.9013770, 0.9526234, -0.0348898, 0.0296833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0161309
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0171183
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0038730, -0.0007367, -0.0041358, -0.0010883, -0.0027847, 0.0033992
1: 0.0188819, 0.0318254, 0.0169186, 0.0319175, -0.0076681, 0.0098448
2: 0.0210367, 0.0298818, 0.0198644, 0.0301375, -0.0060344, 0.0066071
3: 0.0073453, 0.0173764, 0.0054565, 0.0170187, -0.0070636, 0.0094597
4: -0.0184107, -0.0083567, -0.0179328, -0.0065160, -0.0102239, 0.0077537
5: 0.0141068, 0.0264618, 0.0116256, 0.0259851, -0.0093889, 0.0125049
6: 0.0059454, 0.0153905, 0.0041257, 0.0149621, -0.0072406, 0.0096339
7: -0.0228296, -0.0128086, -0.0225269, -0.0106819, -0.0109371, 0.0082503
8: 0.0089450, 0.0189295, 0.0070227, 0.0186232, -0.0069468, 0.0092929
9: 0.9003108, 0.9443833, 0.9014025, 0.9519429, -0.0380082, 0.0287385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160741
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169503
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041334, -0.0010424, -0.0039965, -0.0012290, -0.0029044, 0.0029541
1: 0.0178618, 0.0318816, 0.0174309, 0.0319700, -0.0090316, 0.0089272
2: 0.0205547, 0.0301407, 0.0199892, 0.0299374, -0.0059647, 0.0069127
3: 0.0061322, 0.0169652, 0.0058399, 0.0171019, -0.0079934, 0.0080450
4: -0.0177937, -0.0072913, -0.0179321, -0.0067604, -0.0086978, 0.0083825
5: 0.0124724, 0.0258513, 0.0121674, 0.0260261, -0.0107152, 0.0105866
6: 0.0048014, 0.0148554, 0.0044545, 0.0149842, -0.0080303, 0.0081554
7: -0.0224280, -0.0114140, -0.0225180, -0.0111404, -0.0092903, 0.0094056
8: 0.0075854, 0.0185713, 0.0075324, 0.0187207, -0.0081680, 0.0078994
9: 0.9017551, 0.9485164, 0.9011388, 0.9508425, -0.0329594, 0.0313434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145373, upper bound: 0.0148672
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149293, upper bound: 0.0147974
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0040452, -0.0011675, -0.0039194, -0.0007418, -0.0033034, 0.0027519
1: 0.0183560, 0.0318816, 0.0180084, 0.0320908, -0.0087291, 0.0085400
2: 0.0207523, 0.0299972, 0.0204835, 0.0301331, -0.0060366, 0.0064224
3: 0.0065814, 0.0169569, 0.0066032, 0.0175991, -0.0084936, 0.0077192
4: -0.0177614, -0.0076669, -0.0187164, -0.0075759, -0.0083013, 0.0093425
5: 0.0130665, 0.0258224, 0.0131342, 0.0267923, -0.0114161, 0.0100552
6: 0.0052102, 0.0148321, 0.0052162, 0.0156573, -0.0087729, 0.0077888
7: -0.0223934, -0.0119391, -0.0231297, -0.0120178, -0.0087961, 0.0100511
8: 0.0080894, 0.0185713, 0.0082220, 0.0191380, -0.0083810, 0.0075883
9: 0.9018067, 0.9469921, 0.8990721, 0.9477046, -0.0317049, 0.0337495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143224, upper bound: 0.0147827
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041301, -0.0010666, -0.0040113, -0.0012185, -0.0029116, 0.0029446
1: 0.0176665, 0.0317185, 0.0173008, 0.0319708, -0.0091474, 0.0089388
2: 0.0203836, 0.0300238, 0.0199189, 0.0299590, -0.0061190, 0.0068672
3: 0.0060571, 0.0168502, 0.0057589, 0.0171031, -0.0081564, 0.0080378
4: -0.0176928, -0.0071885, -0.0179369, -0.0066953, -0.0086603, 0.0085507
5: 0.0124001, 0.0257386, 0.0120718, 0.0260303, -0.0108694, 0.0105760
6: 0.0047325, 0.0147601, 0.0043865, 0.0149877, -0.0081786, 0.0081323
7: -0.0223166, -0.0113103, -0.0225231, -0.0110487, -0.0092762, 0.0095484
8: 0.0075582, 0.0184785, 0.0074503, 0.0187207, -0.0082801, 0.0079315
9: 0.9022505, 0.9490474, 0.9011317, 0.9511734, -0.0328614, 0.0321238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156478, upper bound: 0.0160741
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159204, upper bound: 0.0169334
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0040397, -0.0011797, -0.0039342, -0.0007390, -0.0033007, 0.0027545
1: 0.0181492, 0.0317185, 0.0178741, 0.0320916, -0.0088400, 0.0085537
2: 0.0205732, 0.0298794, 0.0204091, 0.0301557, -0.0061804, 0.0063695
3: 0.0064984, 0.0168417, 0.0065163, 0.0176003, -0.0086534, 0.0077078
4: -0.0176593, -0.0075580, -0.0187208, -0.0075050, -0.0082657, 0.0095098
5: 0.0129824, 0.0257084, 0.0130327, 0.0267961, -0.0115722, 0.0100459
6: 0.0051321, 0.0147370, 0.0051435, 0.0156603, -0.0089228, 0.0077636
7: -0.0222811, -0.0118233, -0.0231343, -0.0119207, -0.0087844, 0.0101999
8: 0.0080520, 0.0184785, 0.0081343, 0.0191380, -0.0084935, 0.0076168
9: 0.9023018, 0.9475615, 0.8990651, 0.9480608, -0.0316193, 0.0345066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150554, upper bound: 0.0166659
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169334
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0037763, -0.0013821, -0.0040213, -0.0012021, -0.0025742, 0.0026392
1: 0.0180819, 0.0320361, 0.0182399, 0.0318097, -0.0078871, 0.0084347
2: 0.0200002, 0.0295901, 0.0206130, 0.0299110, -0.0064186, 0.0053079
3: 0.0063578, 0.0174726, 0.0064615, 0.0168736, -0.0070898, 0.0073983
4: -0.0181796, -0.0069277, -0.0177049, -0.0074526, -0.0077798, 0.0081569
5: 0.0127861, 0.0263298, 0.0128772, 0.0257460, -0.0094448, 0.0098658
6: 0.0048096, 0.0152701, 0.0050432, 0.0147708, -0.0073926, 0.0073922
7: -0.0227282, -0.0117322, -0.0223496, -0.0117647, -0.0085508, 0.0082342
8: 0.0082174, 0.0190545, 0.0080172, 0.0184851, -0.0067906, 0.0075932
9: 0.8998722, 0.9498303, 0.9021165, 0.9477148, -0.0296687, 0.0305709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143119, upper bound: 0.0151044
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0147146
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040582, -0.0011684, -0.0040875, -0.0011261, -0.0029321, 0.0029191
1: 0.0174437, 0.0319168, 0.0180513, 0.0318097, -0.0090997, 0.0087377
2: 0.0200498, 0.0300308, 0.0205986, 0.0300332, -0.0067118, 0.0060464
3: 0.0060326, 0.0170190, 0.0063188, 0.0168783, -0.0072518, 0.0077680
4: -0.0179337, -0.0068965, -0.0177232, -0.0073970, -0.0082288, 0.0078485
5: 0.0122186, 0.0259860, 0.0126911, 0.0257623, -0.0098988, 0.0104267
6: 0.0045791, 0.0149628, 0.0049405, 0.0147833, -0.0073062, 0.0078452
7: -0.0225280, -0.0111824, -0.0223687, -0.0115963, -0.0091584, 0.0087387
8: 0.0076451, 0.0186232, 0.0078157, 0.0184851, -0.0075334, 0.0078802
9: 0.9014003, 0.9500028, 0.9020877, 0.9479977, -0.0306557, 0.0301606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155186, upper bound: 0.0147599
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148894, upper bound: 0.0147207
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0037817, -0.0013522, -0.0039885, -0.0012303, -0.0025514, 0.0026363
1: 0.0179879, 0.0320369, 0.0181288, 0.0316530, -0.0078401, 0.0084006
2: 0.0199461, 0.0296071, 0.0204560, 0.0297529, -0.0062742, 0.0054289
3: 0.0063291, 0.0174736, 0.0064672, 0.0167523, -0.0070081, 0.0074292
4: -0.0181868, -0.0069071, -0.0175898, -0.0073929, -0.0078858, 0.0080760
5: 0.0127493, 0.0263337, 0.0129028, 0.0256206, -0.0093652, 0.0098728
6: 0.0047853, 0.0152778, 0.0050356, 0.0146642, -0.0073125, 0.0074414
7: -0.0227331, -0.0117005, -0.0222247, -0.0117531, -0.0085818, 0.0081735
8: 0.0081835, 0.0190545, 0.0080857, 0.0183907, -0.0067431, 0.0075463
9: 0.8998659, 0.9499702, 0.9026237, 0.9480532, -0.0300898, 0.0302322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0156764
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0155044
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0011620, -0.0040929, -0.0011381, -0.0029263, 0.0029309
1: 0.0173192, 0.0319175, 0.0178103, 0.0316530, -0.0091002, 0.0088875
2: 0.0199713, 0.0300403, 0.0204141, 0.0299231, -0.0067018, 0.0061893
3: 0.0059393, 0.0170196, 0.0061951, 0.0167591, -0.0072255, 0.0079485
4: -0.0179360, -0.0068119, -0.0176166, -0.0072584, -0.0084006, 0.0078335
5: 0.0121168, 0.0259878, 0.0125631, 0.0256449, -0.0098842, 0.0106007
6: 0.0044993, 0.0149644, 0.0048292, 0.0146824, -0.0072903, 0.0080059
7: -0.0225303, -0.0110853, -0.0222527, -0.0114462, -0.0093158, 0.0087308
8: 0.0075605, 0.0186232, 0.0077319, 0.0183907, -0.0075099, 0.0080194
9: 0.9013969, 0.9504046, 0.9025822, 0.9486928, -0.0314804, 0.0300720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171563, upper bound: 0.0159041
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159041
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039965, -0.0012290, -0.0040950, -0.0011180, -0.0028785, 0.0028660
1: 0.0174309, 0.0319700, 0.0180102, 0.0318097, -0.0089063, 0.0088578
2: 0.0199892, 0.0299374, 0.0205854, 0.0300432, -0.0067723, 0.0059437
3: 0.0058399, 0.0171019, 0.0062686, 0.0168785, -0.0080492, 0.0079018
4: -0.0179321, -0.0067604, -0.0177240, -0.0073628, -0.0083029, 0.0086640
5: 0.0121674, 0.0260261, 0.0126354, 0.0257630, -0.0105541, 0.0105602
6: 0.0044545, 0.0149842, 0.0048994, 0.0147838, -0.0081303, 0.0079415
7: -0.0225180, -0.0111404, -0.0223695, -0.0115517, -0.0092235, 0.0092564
8: 0.0075324, 0.0187207, 0.0077603, 0.0184851, -0.0079174, 0.0080267
9: 0.9011388, 0.9508425, 0.9020864, 0.9481564, -0.0311831, 0.0329671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148680, upper bound: 0.0145409
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148081, upper bound: 0.0149400
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039194, -0.0007418, -0.0040472, -0.0011748, -0.0027446, 0.0033054
1: 0.0180084, 0.0320908, 0.0182576, 0.0318097, -0.0084830, 0.0087648
2: 0.0204835, 0.0301331, 0.0206857, 0.0299679, -0.0063893, 0.0060236
3: 0.0066032, 0.0175991, 0.0064943, 0.0168740, -0.0077065, 0.0084405
4: -0.0187164, -0.0075759, -0.0177065, -0.0075584, -0.0092486, 0.0082802
5: 0.0131342, 0.0267923, 0.0129445, 0.0257472, -0.0100361, 0.0113326
6: 0.0052162, 0.0156573, 0.0051109, 0.0147718, -0.0077716, 0.0087004
7: -0.0231297, -0.0120178, -0.0223511, -0.0118262, -0.0099671, 0.0087668
8: 0.0082220, 0.0191380, 0.0080123, 0.0184851, -0.0075732, 0.0083387
9: 0.8990721, 0.9477046, 0.9021142, 0.9473943, -0.0335165, 0.0316369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143276
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040113, -0.0012185, -0.0040929, -0.0011381, -0.0028732, 0.0028744
1: 0.0173008, 0.0319708, 0.0178103, 0.0316530, -0.0089130, 0.0089823
2: 0.0199189, 0.0299590, 0.0204141, 0.0299231, -0.0067233, 0.0061035
3: 0.0057589, 0.0171031, 0.0061951, 0.0167591, -0.0080396, 0.0080643
4: -0.0179369, -0.0066953, -0.0176166, -0.0072584, -0.0084689, 0.0086266
5: 0.0120718, 0.0260303, 0.0125631, 0.0256449, -0.0105415, 0.0107121
6: 0.0043865, 0.0149877, 0.0048292, 0.0146824, -0.0081043, 0.0080906
7: -0.0225231, -0.0110487, -0.0222527, -0.0114462, -0.0093602, 0.0092434
8: 0.0074503, 0.0187207, 0.0077319, 0.0183907, -0.0079484, 0.0081374
9: 0.9011317, 0.9511734, 0.9025822, 0.9486928, -0.0319520, 0.0328653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160780, upper bound: 0.0157059
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0159484
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039342, -0.0007390, -0.0040448, -0.0011867, -0.0027475, 0.0033058
1: 0.0178741, 0.0320916, 0.0180511, 0.0316530, -0.0084897, 0.0088735
2: 0.0204091, 0.0301557, 0.0205105, 0.0298485, -0.0063393, 0.0061666
3: 0.0065163, 0.0176003, 0.0064163, 0.0167545, -0.0076951, 0.0086026
4: -0.0187208, -0.0075050, -0.0175987, -0.0074536, -0.0094201, 0.0082448
5: 0.0130327, 0.0267961, 0.0128616, 0.0256285, -0.0100264, 0.0114890
6: 0.0051435, 0.0156603, 0.0050366, 0.0146701, -0.0077466, 0.0088513
7: -0.0231343, -0.0119207, -0.0222338, -0.0117165, -0.0101139, 0.0087561
8: 0.0081343, 0.0191380, 0.0079779, 0.0183907, -0.0076004, 0.0084503
9: 0.8990651, 0.9480608, 0.9026101, 0.9479351, -0.0342689, 0.0315486

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0158301
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041660, -0.0010444, -0.0039229, -0.0012480, -0.0029180, 0.0028786
1: 0.0168204, 0.0319168, 0.0185970, 0.0318816, -0.0098869, 0.0079002
2: 0.0198502, 0.0301768, 0.0207043, 0.0297991, -0.0065184, 0.0061564
3: 0.0053290, 0.0170216, 0.0067706, 0.0169469, -0.0087392, 0.0071306
4: -0.0179438, -0.0064135, -0.0177230, -0.0076762, -0.0078201, 0.0090774
5: 0.0114423, 0.0259942, 0.0133270, 0.0257886, -0.0114563, 0.0095314
6: 0.0040060, 0.0149697, 0.0053205, 0.0148050, -0.0086859, 0.0073094
7: -0.0225382, -0.0105332, -0.0223524, -0.0121252, -0.0083860, 0.0100160
8: 0.0068723, 0.0186232, 0.0084368, 0.0185713, -0.0088293, 0.0070638
9: 0.9013841, 0.9522769, 0.9018676, 0.9468139, -0.0289454, 0.0349322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146145, upper bound: 0.0147293
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154260, upper bound: 0.0146669
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041197, -0.0011125, -0.0038799, -0.0007271, -0.0033925, 0.0027673
1: 0.0170545, 0.0319168, 0.0190932, 0.0319850, -0.0098018, 0.0075612
2: 0.0199383, 0.0301097, 0.0212186, 0.0300048, -0.0066225, 0.0058661
3: 0.0055455, 0.0170175, 0.0074311, 0.0174914, -0.0094635, 0.0069037
4: -0.0179279, -0.0065862, -0.0185093, -0.0084806, -0.0075874, 0.0102679
5: 0.0117266, 0.0259810, 0.0141970, 0.0265745, -0.0125221, 0.0092420
6: 0.0041981, 0.0149587, 0.0060287, 0.0154904, -0.0096611, 0.0070933
7: -0.0225219, -0.0107786, -0.0229394, -0.0129332, -0.0081152, 0.0109580
8: 0.0071119, 0.0186232, 0.0089897, 0.0190317, -0.0092691, 0.0068410
9: 0.9014100, 0.9515868, 0.8998247, 0.9437917, -0.0279633, 0.0381107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140702, upper bound: 0.0146479
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147606, upper bound: 0.0146180
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0039175, -0.0012568, -0.0029248, 0.0029042
1: 0.0166869, 0.0319175, 0.0184070, 0.0317185, -0.0099290, 0.0080037
2: 0.0197779, 0.0302042, 0.0205439, 0.0296770, -0.0064793, 0.0063169
3: 0.0052423, 0.0170228, 0.0067016, 0.0168319, -0.0087455, 0.0072790
4: -0.0179484, -0.0063458, -0.0176213, -0.0075809, -0.0079774, 0.0090474
5: 0.0113442, 0.0259980, 0.0132559, 0.0256738, -0.0114603, 0.0096712
6: 0.0039356, 0.0149728, 0.0052512, 0.0147106, -0.0086686, 0.0074490
7: -0.0225430, -0.0104393, -0.0222407, -0.0120308, -0.0085152, 0.0100048
8: 0.0067857, 0.0186232, 0.0084054, 0.0184785, -0.0088606, 0.0071616
9: 0.9013770, 0.9526234, 0.9023612, 0.9473296, -0.0296833, 0.0348898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161309, upper bound: 0.0154774
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171183, upper bound: 0.0158176
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041358, -0.0010882, -0.0038730, -0.0007367, -0.0033992, 0.0027847
1: 0.0169186, 0.0319175, 0.0188819, 0.0318254, -0.0098448, 0.0076681
2: 0.0198644, 0.0301375, 0.0210367, 0.0298818, -0.0066071, 0.0060344
3: 0.0054565, 0.0170187, 0.0073453, 0.0173764, -0.0094597, 0.0070636
4: -0.0179328, -0.0065159, -0.0184107, -0.0083567, -0.0077537, 0.0102239
5: 0.0116256, 0.0259851, 0.0141068, 0.0264618, -0.0125049, 0.0093889
6: 0.0041257, 0.0149621, 0.0059454, 0.0153905, -0.0096339, 0.0072406
7: -0.0225269, -0.0106819, -0.0228296, -0.0128086, -0.0082503, 0.0109371
8: 0.0070227, 0.0186232, 0.0089450, 0.0189295, -0.0092929, 0.0069468
9: 0.9014025, 0.9519429, 0.9003108, 0.9443833, -0.0287385, 0.0380082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0154774
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169503, upper bound: 0.0158176
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039965, -0.0012290, -0.0041334, -0.0010424, -0.0029541, 0.0029044
1: 0.0174309, 0.0319700, 0.0178618, 0.0318816, -0.0089272, 0.0090316
2: 0.0199892, 0.0299374, 0.0205547, 0.0301407, -0.0069127, 0.0059647
3: 0.0058399, 0.0171019, 0.0061322, 0.0169652, -0.0080450, 0.0079934
4: -0.0179321, -0.0067604, -0.0177937, -0.0072913, -0.0083825, 0.0086978
5: 0.0121674, 0.0260261, 0.0124724, 0.0258513, -0.0105866, 0.0107152
6: 0.0044545, 0.0149842, 0.0048014, 0.0148554, -0.0081554, 0.0080303
7: -0.0225180, -0.0111404, -0.0224280, -0.0114140, -0.0094056, 0.0092903
8: 0.0075324, 0.0187207, 0.0075854, 0.0185713, -0.0078994, 0.0081680
9: 0.9011388, 0.9508425, 0.9017551, 0.9485164, -0.0313434, 0.0329594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148672, upper bound: 0.0145409
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147974, upper bound: 0.0149400
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039194, -0.0007418, -0.0040452, -0.0011675, -0.0027519, 0.0033034
1: 0.0180084, 0.0320908, 0.0183560, 0.0318816, -0.0085400, 0.0087291
2: 0.0204835, 0.0301331, 0.0207523, 0.0299972, -0.0064224, 0.0060366
3: 0.0066032, 0.0175991, 0.0065814, 0.0169569, -0.0077192, 0.0084936
4: -0.0187164, -0.0075759, -0.0177614, -0.0076669, -0.0093425, 0.0083013
5: 0.0131342, 0.0267923, 0.0130665, 0.0258224, -0.0100552, 0.0114161
6: 0.0052162, 0.0156573, 0.0052102, 0.0148321, -0.0077888, 0.0087729
7: -0.0231297, -0.0120178, -0.0223934, -0.0119391, -0.0100511, 0.0087961
8: 0.0082220, 0.0191380, 0.0080894, 0.0185713, -0.0075883, 0.0083810
9: 0.8990721, 0.9477046, 0.9018067, 0.9469921, -0.0337495, 0.0317049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147827, upper bound: 0.0143224
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147325, upper bound: 0.0146182
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0040113, -0.0012185, -0.0041301, -0.0010666, -0.0029446, 0.0029116
1: 0.0173008, 0.0319708, 0.0176665, 0.0317185, -0.0089388, 0.0091474
2: 0.0199189, 0.0299590, 0.0203836, 0.0300238, -0.0068672, 0.0061190
3: 0.0057589, 0.0171031, 0.0060571, 0.0168502, -0.0080378, 0.0081564
4: -0.0179369, -0.0066953, -0.0176928, -0.0071885, -0.0085507, 0.0086603
5: 0.0120718, 0.0260303, 0.0124001, 0.0257386, -0.0105760, 0.0108694
6: 0.0043865, 0.0149877, 0.0047325, 0.0147601, -0.0081323, 0.0081786
7: -0.0225231, -0.0110487, -0.0223166, -0.0113103, -0.0095484, 0.0092762
8: 0.0074503, 0.0187207, 0.0075582, 0.0184785, -0.0079315, 0.0082801
9: 0.9011317, 0.9511734, 0.9022505, 0.9490474, -0.0321238, 0.0328614

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0157059
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0159484
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039342, -0.0007390, -0.0040397, -0.0011797, -0.0027545, 0.0033007
1: 0.0178741, 0.0320916, 0.0181492, 0.0317185, -0.0085537, 0.0088400
2: 0.0204091, 0.0301557, 0.0205732, 0.0298794, -0.0063695, 0.0061804
3: 0.0065163, 0.0176003, 0.0064984, 0.0168417, -0.0077078, 0.0086534
4: -0.0187208, -0.0075050, -0.0176593, -0.0075580, -0.0095098, 0.0082657
5: 0.0130327, 0.0267961, 0.0129824, 0.0257084, -0.0100459, 0.0115722
6: 0.0051435, 0.0156603, 0.0051321, 0.0147370, -0.0077636, 0.0089228
7: -0.0231343, -0.0119207, -0.0222811, -0.0118233, -0.0101999, 0.0087844
8: 0.0081343, 0.0191380, 0.0080520, 0.0184785, -0.0076168, 0.0084935
9: 0.8990651, 0.9480608, 0.9023018, 0.9475615, -0.0345066, 0.0316193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0158177
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0038468, -0.0014314, -0.0041029, -0.0011424, -0.0027044, 0.0026715
1: 0.0180705, 0.0321710, 0.0170118, 0.0319168, -0.0068840, 0.0086836
2: 0.0201124, 0.0297486, 0.0198732, 0.0300703, -0.0057442, 0.0054986
3: 0.0062394, 0.0174815, 0.0054987, 0.0170177, -0.0064082, 0.0074986
4: -0.0181898, -0.0069059, -0.0179289, -0.0064955, -0.0077833, 0.0074432
5: 0.0126183, 0.0263471, 0.0116541, 0.0259819, -0.0088063, 0.0099848
6: 0.0047192, 0.0152706, 0.0041328, 0.0149595, -0.0067960, 0.0074333
7: -0.0227687, -0.0116142, -0.0225229, -0.0107173, -0.0087004, 0.0077670
8: 0.0080180, 0.0190597, 0.0070920, 0.0186232, -0.0062883, 0.0078050
9: 0.8997339, 0.9498649, 0.9014083, 0.9518887, -0.0297565, 0.0266044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154310, upper bound: 0.0148146
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153595, upper bound: 0.0141754
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0040601, -0.0011499, -0.0041460, -0.0010715, -0.0029886, 0.0029960
1: 0.0176607, 0.0320711, 0.0169281, 0.0319168, -0.0080625, 0.0088991
2: 0.0202256, 0.0301516, 0.0198828, 0.0301505, -0.0059928, 0.0062108
3: 0.0061848, 0.0171328, 0.0054490, 0.0170212, -0.0064270, 0.0081349
4: -0.0180311, -0.0070538, -0.0179420, -0.0064928, -0.0085604, 0.0070922
5: 0.0123696, 0.0260979, 0.0115735, 0.0259927, -0.0090989, 0.0108301
6: 0.0047141, 0.0150493, 0.0041023, 0.0149684, -0.0066172, 0.0081967
7: -0.0226414, -0.0113383, -0.0225364, -0.0106439, -0.0095507, 0.0081226
8: 0.0077540, 0.0187140, 0.0070047, 0.0186232, -0.0068448, 0.0081937
9: 0.9009085, 0.9492470, 0.9013871, 0.9518969, -0.0317691, 0.0258180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161620, upper bound: 0.0141554
time: 0.67 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158468, upper bound: 0.0140758
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040341, -0.0012072, -0.0037817, -0.0013522, -0.0026819, 0.0025745
1: 0.0171962, 0.0319171, 0.0179879, 0.0320369, -0.0081377, 0.0068282
2: 0.0198761, 0.0299649, 0.0199461, 0.0296071, -0.0051538, 0.0057216
3: 0.0056675, 0.0170134, 0.0063291, 0.0174736, -0.0071838, 0.0062970
4: -0.0179120, -0.0065741, -0.0181868, -0.0069071, -0.0074550, 0.0076342
5: 0.0118746, 0.0259680, 0.0127493, 0.0263337, -0.0096177, 0.0086390
6: 0.0042628, 0.0149480, 0.0047853, 0.0152778, -0.0072115, 0.0067222
7: -0.0225058, -0.0109087, -0.0227331, -0.0117005, -0.0076611, 0.0083752
8: 0.0073156, 0.0186232, 0.0081835, 0.0190545, -0.0074262, 0.0060898
9: 0.9014355, 0.9515424, 0.8998659, 0.9499702, -0.0266946, 0.0287759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166864, upper bound: 0.0151155
time: 0.73 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166864, upper bound: 0.0150795
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041719, -0.0010331, -0.0040643, -0.0011620, -0.0030100, 0.0030313
1: 0.0167731, 0.0319171, 0.0173192, 0.0319175, -0.0088129, 0.0083787
2: 0.0198245, 0.0301870, 0.0199713, 0.0300403, -0.0060608, 0.0063295
3: 0.0052983, 0.0170220, 0.0059393, 0.0170196, -0.0080911, 0.0066688
4: -0.0179452, -0.0063906, -0.0179360, -0.0068119, -0.0073373, 0.0084821
5: 0.0114082, 0.0259954, 0.0121168, 0.0259878, -0.0107937, 0.0093405
6: 0.0039812, 0.0149707, 0.0044993, 0.0149644, -0.0081483, 0.0068323
7: -0.0225397, -0.0105001, -0.0225303, -0.0110853, -0.0083622, 0.0094997
8: 0.0068417, 0.0186232, 0.0075605, 0.0186232, -0.0082098, 0.0070277
9: 0.9013817, 0.9523923, 0.9013969, 0.9504046, -0.0270157, 0.0314393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159891
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159041
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0040135, -0.0012112, -0.0041660, -0.0010444, -0.0029691, 0.0029548
1: 0.0175464, 0.0321350, 0.0168204, 0.0319168, -0.0078427, 0.0091997
2: 0.0201260, 0.0300737, 0.0198502, 0.0301768, -0.0060647, 0.0061508
3: 0.0058647, 0.0172148, 0.0053290, 0.0170216, -0.0072949, 0.0083014
4: -0.0180374, -0.0068304, -0.0179438, -0.0064135, -0.0086323, 0.0079072
5: 0.0121886, 0.0261421, 0.0114423, 0.0259942, -0.0097573, 0.0109721
6: 0.0044880, 0.0150788, 0.0040060, 0.0149697, -0.0074643, 0.0082943
7: -0.0226382, -0.0111891, -0.0225382, -0.0105332, -0.0096463, 0.0085988
8: 0.0075119, 0.0188095, 0.0068723, 0.0186232, -0.0072912, 0.0084363
9: 0.9006134, 0.9505134, 0.9013841, 0.9522769, -0.0323755, 0.0289474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B1_A2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157601, upper bound: 0.0138293
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157875, upper bound: 0.0144901
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0039371, -0.0007310, -0.0041139, -0.0011204, -0.0028166, 0.0033829
1: 0.0181546, 0.0322504, 0.0170852, 0.0319168, -0.0074105, 0.0091080
2: 0.0206313, 0.0302710, 0.0199500, 0.0301008, -0.0057616, 0.0061887
3: 0.0066471, 0.0177078, 0.0055736, 0.0170169, -0.0070426, 0.0090467
4: -0.0188108, -0.0076603, -0.0179258, -0.0066090, -0.0098724, 0.0076889
5: 0.0131745, 0.0269073, 0.0117637, 0.0259793, -0.0094188, 0.0120772
6: 0.0052647, 0.0157472, 0.0042232, 0.0149573, -0.0072410, 0.0093106
7: -0.0232462, -0.0120834, -0.0225197, -0.0108107, -0.0106115, 0.0082692
8: 0.0082178, 0.0192310, 0.0071432, 0.0186232, -0.0070013, 0.0088947
9: 0.8985667, 0.9472936, 0.9014133, 0.9514952, -0.0357467, 0.0280082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156761, upper bound: 0.0134165
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B1_A2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157285, upper bound: 0.0139730
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040022, -0.0012256, -0.0041816, -0.0010133, -0.0029888, 0.0029560
1: 0.0173884, 0.0319703, 0.0166869, 0.0319175, -0.0079400, 0.0092351
2: 0.0199662, 0.0299456, 0.0197779, 0.0302042, -0.0062328, 0.0060955
3: 0.0058134, 0.0171023, 0.0052423, 0.0170228, -0.0074280, 0.0082898
4: -0.0179337, -0.0067381, -0.0179484, -0.0063458, -0.0086028, 0.0080579
5: 0.0121338, 0.0260275, 0.0113442, 0.0259980, -0.0098728, 0.0109677
6: 0.0044304, 0.0149854, 0.0039356, 0.0149728, -0.0075919, 0.0082769
7: -0.0225198, -0.0111087, -0.0225430, -0.0104393, -0.0096372, 0.0087105
8: 0.0075046, 0.0187207, 0.0067857, 0.0186232, -0.0073643, 0.0084513
9: 0.9011363, 0.9509469, 0.9013770, 0.9526234, -0.0322911, 0.0296438

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150996
time: 0.72 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0159484
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0039250, -0.0007410, -0.0041290, -0.0010980, -0.0028270, 0.0033880
1: 0.0179611, 0.0320911, 0.0169543, 0.0319175, -0.0075215, 0.0091386
2: 0.0204560, 0.0301416, 0.0198781, 0.0301270, -0.0059354, 0.0061471
3: 0.0065737, 0.0175995, 0.0054897, 0.0170181, -0.0071910, 0.0090294
4: -0.0187179, -0.0075488, -0.0179304, -0.0065425, -0.0098248, 0.0078489
5: 0.0130968, 0.0267937, 0.0116693, 0.0259830, -0.0095526, 0.0120514
6: 0.0051896, 0.0156583, 0.0041554, 0.0149604, -0.0073796, 0.0092788
7: -0.0231313, -0.0119814, -0.0225243, -0.0107197, -0.0105826, 0.0083959
8: 0.0081919, 0.0191380, 0.0070594, 0.0186232, -0.0070947, 0.0089013
9: 0.8990695, 0.9478247, 0.9014063, 0.9518356, -0.0355697, 0.0287487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0158301
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0041660, -0.0010444, -0.0040135, -0.0012112, -0.0029548, 0.0029691
1: 0.0168204, 0.0319168, 0.0175464, 0.0321350, -0.0091997, 0.0078427
2: 0.0198502, 0.0301768, 0.0201260, 0.0300737, -0.0061508, 0.0060647
3: 0.0053290, 0.0170216, 0.0058647, 0.0172148, -0.0083014, 0.0072949
4: -0.0179438, -0.0064135, -0.0180374, -0.0068304, -0.0079072, 0.0086323
5: 0.0114423, 0.0259942, 0.0121886, 0.0261421, -0.0109721, 0.0097573
6: 0.0040060, 0.0149697, 0.0044880, 0.0150788, -0.0082943, 0.0074643
7: -0.0225382, -0.0105332, -0.0226382, -0.0111891, -0.0085988, 0.0096463
8: 0.0068723, 0.0186232, 0.0075119, 0.0188095, -0.0084363, 0.0072912
9: 0.9013841, 0.9522769, 0.9006134, 0.9505134, -0.0289474, 0.0323755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146145, upper bound: 0.0147293
time: 0.70 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154260, upper bound: 0.0146669
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041139, -0.0011205, -0.0039371, -0.0007310, -0.0033829, 0.0028166
1: 0.0170852, 0.0319168, 0.0181546, 0.0322504, -0.0091080, 0.0074105
2: 0.0199500, 0.0301008, 0.0206313, 0.0302710, -0.0061887, 0.0057616
3: 0.0055736, 0.0170169, 0.0066471, 0.0177078, -0.0090467, 0.0070426
4: -0.0179258, -0.0066090, -0.0188108, -0.0076603, -0.0076889, 0.0098724
5: 0.0117637, 0.0259793, 0.0131745, 0.0269073, -0.0120772, 0.0094188
6: 0.0042232, 0.0149573, 0.0052647, 0.0157472, -0.0093106, 0.0072410
7: -0.0225197, -0.0108107, -0.0232462, -0.0120834, -0.0082692, 0.0106115
8: 0.0071432, 0.0186232, 0.0082178, 0.0192310, -0.0088946, 0.0070013
9: 0.9014133, 0.9514952, 0.8985667, 0.9472936, -0.0280082, 0.0357467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: B, layer: 3, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140702, upper bound: 0.0146479
time: 0.69 seconds

## Relational analysis of IS_A2_B2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147606, upper bound: 0.0146180
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0041816, -0.0010133, -0.0040022, -0.0012256, -0.0029560, 0.0029888
1: 0.0166869, 0.0319175, 0.0173884, 0.0319703, -0.0092351, 0.0079400
2: 0.0197779, 0.0302042, 0.0199662, 0.0299456, -0.0060955, 0.0062328
3: 0.0052423, 0.0170228, 0.0058134, 0.0171023, -0.0082898, 0.0074280
4: -0.0179484, -0.0063458, -0.0179337, -0.0067381, -0.0080579, 0.0086028
5: 0.0113442, 0.0259980, 0.0121338, 0.0260275, -0.0109677, 0.0098728
6: 0.0039356, 0.0149728, 0.0044304, 0.0149854, -0.0082769, 0.0075919
7: -0.0225430, -0.0104393, -0.0225198, -0.0111087, -0.0087105, 0.0096372
8: 0.0067857, 0.0186232, 0.0075046, 0.0187207, -0.0084513, 0.0073643
9: 0.9013770, 0.9526234, 0.9011363, 0.9509469, -0.0296438, 0.0322911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161309, upper bound: 0.0154774
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171183, upper bound: 0.0158176
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0041290, -0.0010980, -0.0039250, -0.0007410, -0.0033880, 0.0028270
1: 0.0169543, 0.0319175, 0.0179611, 0.0320911, -0.0091386, 0.0075215
2: 0.0198781, 0.0301270, 0.0204560, 0.0301416, -0.0061471, 0.0059354
3: 0.0054897, 0.0170181, 0.0065737, 0.0175995, -0.0090294, 0.0071910
4: -0.0179304, -0.0065425, -0.0187179, -0.0075488, -0.0078489, 0.0098248
5: 0.0116693, 0.0259830, 0.0130968, 0.0267937, -0.0120514, 0.0095526
6: 0.0041554, 0.0149604, 0.0051896, 0.0156583, -0.0092788, 0.0073796
7: -0.0225243, -0.0107197, -0.0231313, -0.0119814, -0.0083959, 0.0105826
8: 0.0070594, 0.0186232, 0.0081919, 0.0191380, -0.0089013, 0.0070947
9: 0.9014063, 0.9518356, 0.8990695, 0.9478247, -0.0287487, 0.0355697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0154774
time: 0.76 seconds

## Relational analysis of IS_A2_B2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169503, upper bound: 0.0158176
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0042168, -0.0009004, -0.0039965, -0.0012290, -0.0029878, 0.0030961
1: 0.0167986, 0.0321350, 0.0174309, 0.0319700, -0.0089734, 0.0080799
2: 0.0199715, 0.0304100, 0.0199892, 0.0299374, -0.0057942, 0.0065235
3: 0.0052387, 0.0172312, 0.0058399, 0.0171019, -0.0081379, 0.0074935
4: -0.0181021, -0.0064220, -0.0179321, -0.0067604, -0.0081780, 0.0084622
5: 0.0113158, 0.0262001, 0.0121674, 0.0260261, -0.0109077, 0.0099749
6: 0.0039559, 0.0151269, 0.0044545, 0.0149842, -0.0081752, 0.0076858
7: -0.0227067, -0.0104531, -0.0225180, -0.0111404, -0.0088339, 0.0095903
8: 0.0067000, 0.0188095, 0.0075324, 0.0187207, -0.0083592, 0.0073854
9: 0.9005122, 0.9522641, 0.9011388, 0.9508425, -0.0299687, 0.0312761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155563, upper bound: 0.0141608
time: 0.74 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160926, upper bound: 0.0140402
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041142, -0.0010959, -0.0039194, -0.0007418, -0.0033723, 0.0028235
1: 0.0173691, 0.0321350, 0.0180084, 0.0320908, -0.0086689, 0.0077311
2: 0.0201836, 0.0302540, 0.0204835, 0.0301331, -0.0058563, 0.0061053
3: 0.0057602, 0.0172224, 0.0066032, 0.0175991, -0.0088481, 0.0073043
4: -0.0180671, -0.0068335, -0.0187164, -0.0075759, -0.0079824, 0.0096865
5: 0.0119913, 0.0261689, 0.0131342, 0.0267923, -0.0118875, 0.0096881
6: 0.0044155, 0.0151010, 0.0052162, 0.0156573, -0.0091517, 0.0074900
7: -0.0226699, -0.0110364, -0.0231297, -0.0120178, -0.0085347, 0.0104193
8: 0.0072814, 0.0188095, 0.0082220, 0.0191380, -0.0087227, 0.0071744
9: 0.9005666, 0.9505700, 0.8990721, 0.9477046, -0.0293282, 0.0346742

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 8

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152806, upper bound: 0.0140804
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B2_A2_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157253, upper bound: 0.0139595
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0040022, -0.0012256, -0.0042187, -0.0009181, -0.0030841, 0.0029931
1: 0.0173884, 0.0319703, 0.0165520, 0.0319708, -0.0079124, 0.0092516
2: 0.0199662, 0.0299456, 0.0197592, 0.0302962, -0.0063667, 0.0060542
3: 0.0058134, 0.0171023, 0.0051241, 0.0171198, -0.0074069, 0.0083596
4: -0.0179337, -0.0067381, -0.0180029, -0.0062789, -0.0086852, 0.0080842
5: 0.0121338, 0.0260275, 0.0111924, 0.0260893, -0.0098813, 0.0111279
6: 0.0044304, 0.0149854, 0.0038457, 0.0150357, -0.0076077, 0.0083709
7: -0.0225198, -0.0111087, -0.0225917, -0.0103089, -0.0098099, 0.0087327
8: 0.0075046, 0.0187207, 0.0066378, 0.0187207, -0.0073295, 0.0085335
9: 0.9011363, 0.9509469, 0.9010318, 0.9529719, -0.0323486, 0.0295372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150996
time: 0.75 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0159484
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_B2_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0039250, -0.0007410, -0.0041115, -0.0011128, -0.0028122, 0.0033705
1: 0.0179611, 0.0320911, 0.0171277, 0.0319708, -0.0075543, 0.0089193
2: 0.0204560, 0.0301416, 0.0199721, 0.0301320, -0.0059376, 0.0061011
3: 0.0065737, 0.0175995, 0.0056580, 0.0171103, -0.0072127, 0.0090562
4: -0.0187179, -0.0075488, -0.0179654, -0.0066962, -0.0099009, 0.0078833
5: 0.0130968, 0.0267937, 0.0118784, 0.0260560, -0.0095875, 0.0120939
6: 0.0051896, 0.0156583, 0.0043143, 0.0150086, -0.0074063, 0.0093393
7: -0.0231313, -0.0119814, -0.0225527, -0.0109005, -0.0106279, 0.0084313
8: 0.0081919, 0.0191380, 0.0072276, 0.0187207, -0.0071139, 0.0088783
9: 0.8990695, 0.9478247, 0.9010882, 0.9512573, -0.0357051, 0.0288667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
time: 0.77 seconds

## Relational analysis of IS_A2_B2_B2_A2_A2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0158177
time: 0.78 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.06 seconds
IS_A1_B1_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0145575, upper bound: 0.0148610
IS_A1_B1_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0141975
IS_A1_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0151263, upper bound: 0.0141871
IS_A1_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0148435, upper bound: 0.0140992
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152647
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152410
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160988, upper bound: 0.0161597
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160988, upper bound: 0.0160988
IS_A1_B1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0138747, upper bound: 0.0148278
IS_A1_B1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0145360, upper bound: 0.0148031
IS_A1_B1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
IS_A1_B1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
IS_A1_B1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0152527, upper bound: 0.0156279
IS_A1_B1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0161630, upper bound: 0.0160080
IS_A1_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0152076, upper bound: 0.0156279
IS_A1_B1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160300, upper bound: 0.0160080
IS_A1_B1_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0148278, upper bound: 0.0138747
IS_A1_B1_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0148031, upper bound: 0.0145360
IS_A1_B1_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
IS_A1_B1_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
IS_A1_B1_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152527
IS_A1_B1_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0161630
IS_A1_B1_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152076
IS_A1_B1_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0160300
IS_A1_B1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0141666, upper bound: 0.0146558
IS_A1_B1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0140765, upper bound: 0.0151040
IS_A1_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0140835, upper bound: 0.0144218
IS_A1_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
IS_A1_B1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0152357, upper bound: 0.0156406
IS_A1_B1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160932, upper bound: 0.0160120
IS_A1_B1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0152067, upper bound: 0.0156406
IS_A1_B1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0160120
IS_A1_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0151044, upper bound: 0.0143119
IS_A1_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147146, upper bound: 0.0141498
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147599, upper bound: 0.0155186
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147207, upper bound: 0.0148894
IS_A1_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0156764, upper bound: 0.0160950
IS_A1_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0155044, upper bound: 0.0160950
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0171563
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0170393
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0145409, upper bound: 0.0148680
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0149400, upper bound: 0.0148081
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0143276, upper bound: 0.0147842
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
IS_A1_B2_A1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0157059, upper bound: 0.0160780
IS_A1_B2_A1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0159484, upper bound: 0.0169337
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0150554, upper bound: 0.0166586
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0158301, upper bound: 0.0169337
IS_A1_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147293, upper bound: 0.0146145
IS_A1_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0146669, upper bound: 0.0154260
IS_A1_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
IS_A1_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
IS_A1_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0161309
IS_A1_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0171183
IS_A1_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160741
IS_A1_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169503
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0145373, upper bound: 0.0148672
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0149293, upper bound: 0.0147974
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0143224, upper bound: 0.0147827
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
IS_A1_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0156478, upper bound: 0.0160741
IS_A1_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0159204, upper bound: 0.0169334
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0150554, upper bound: 0.0166659
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169334
IS_A2_B1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0143119, upper bound: 0.0151044
IS_A2_B1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0147146
IS_A2_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0155186, upper bound: 0.0147599
IS_A2_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0148894, upper bound: 0.0147207
IS_A2_B1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0156764
IS_A2_B1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0155044
IS_A2_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0171563, upper bound: 0.0159041
IS_A2_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159041
IS_A2_B1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0148680, upper bound: 0.0145409
IS_A2_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0148081, upper bound: 0.0149400
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143276
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
IS_A2_B1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160780, upper bound: 0.0157059
IS_A2_B1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0159484
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0158301
IS_A2_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0146145, upper bound: 0.0147293
IS_A2_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0154260, upper bound: 0.0146669
IS_A2_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0140702, upper bound: 0.0146479
IS_A2_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147606, upper bound: 0.0146180
IS_A2_B1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0161309, upper bound: 0.0154774
IS_A2_B1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0171183, upper bound: 0.0158176
IS_A2_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0154774
IS_A2_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169503, upper bound: 0.0158176
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0148672, upper bound: 0.0145409
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147974, upper bound: 0.0149400
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147827, upper bound: 0.0143224
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147325, upper bound: 0.0146182
IS_A2_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0157059
IS_A2_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0159484
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0158177
IS_A2_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0154310, upper bound: 0.0148146
IS_A2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0153595, upper bound: 0.0141754
IS_A2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0161620, upper bound: 0.0141554
IS_A2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0158468, upper bound: 0.0140758
IS_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166864, upper bound: 0.0151155
IS_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166864, upper bound: 0.0150795
IS_A2_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159891
IS_A2_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159041
IS_A2_B2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0157601, upper bound: 0.0138293
IS_A2_B2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0157875, upper bound: 0.0144901
IS_A2_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0156761, upper bound: 0.0134165
IS_A2_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0157285, upper bound: 0.0139730
IS_A2_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150996
IS_A2_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0159484
IS_A2_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0158301
IS_A2_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0146145, upper bound: 0.0147293
IS_A2_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0154260, upper bound: 0.0146669
IS_A2_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0140702, upper bound: 0.0146479
IS_A2_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0147606, upper bound: 0.0146180
IS_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0161309, upper bound: 0.0154774
IS_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0171183, upper bound: 0.0158176
IS_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0154774
IS_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169503, upper bound: 0.0158176
IS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0155563, upper bound: 0.0141608
IS_A2_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0160926, upper bound: 0.0140402
IS_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0152806, upper bound: 0.0140804
IS_A2_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0157253, upper bound: 0.0139595
IS_A2_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150996
IS_A2_B2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0159484
IS_A2_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.06
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0158177

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0035738, -0.0014883, -0.0040865, -0.0011470, -0.0024267, 0.0025697
1: 0.0196329, 0.0319055, 0.0178614, 0.0316527, -0.0053600, 0.0083000
2: 0.0208356, 0.0291895, 0.0204404, 0.0299116, -0.0052710, 0.0047498
3: 0.0076653, 0.0171740, 0.0062230, 0.0167586, -0.0051631, 0.0072794
4: -0.0177815, -0.0080994, -0.0176149, -0.0072804, -0.0075460, 0.0065890
5: 0.0144538, 0.0259178, 0.0125946, 0.0256433, -0.0071925, 0.0097057
6: 0.0059960, 0.0149267, 0.0048514, 0.0146812, -0.0057517, 0.0072233
7: -0.0223971, -0.0131271, -0.0222509, -0.0114797, -0.0085132, 0.0064355
8: 0.0096054, 0.0188062, 0.0077587, 0.0183907, -0.0048300, 0.0075687
9: 0.9011216, 0.9447060, 0.9025850, 0.9485730, -0.0285678, 0.0226390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145575, upper bound: 0.0148610
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145575, upper bound: 0.0148610
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034896, -0.0009321, -0.0039909, -0.0012241, -0.0022655, 0.0030021
1: 0.0201048, 0.0320241, 0.0183443, 0.0316527, -0.0052331, 0.0080496
2: 0.0213636, 0.0293564, 0.0206397, 0.0297703, -0.0048726, 0.0051435
3: 0.0083841, 0.0177582, 0.0066644, 0.0167493, -0.0050014, 0.0079799
4: -0.0186157, -0.0089252, -0.0175785, -0.0076653, -0.0087759, 0.0063809
5: 0.0153890, 0.0267465, 0.0131900, 0.0256102, -0.0069108, 0.0106887
6: 0.0067258, 0.0156697, 0.0052645, 0.0146564, -0.0055682, 0.0082102
7: -0.0229931, -0.0140121, -0.0222126, -0.0120144, -0.0094098, 0.0061304
8: 0.0102399, 0.0192998, 0.0082527, 0.0183907, -0.0047229, 0.0079214
9: 0.8989591, 0.9414968, 0.9026415, 0.9470575, -0.0318462, 0.0219474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0141949
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0141975
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039902, -0.0012034, -0.0039082, -0.0012659, -0.0027242, 0.0027048
1: 0.0186371, 0.0318097, 0.0184666, 0.0316527, -0.0073910, 0.0072462
2: 0.0208021, 0.0299066, 0.0205695, 0.0296304, -0.0050405, 0.0055788
3: 0.0070426, 0.0168759, 0.0067574, 0.0167439, -0.0059607, 0.0068975
4: -0.0177137, -0.0079080, -0.0175573, -0.0076062, -0.0077309, 0.0067097
5: 0.0134865, 0.0257539, 0.0133010, 0.0255914, -0.0085185, 0.0092829
6: 0.0055432, 0.0147769, 0.0052868, 0.0146420, -0.0062104, 0.0071944
7: -0.0223589, -0.0122311, -0.0221901, -0.0120746, -0.0082925, 0.0076607
8: 0.0086056, 0.0184851, 0.0084686, 0.0183907, -0.0063104, 0.0066892
9: 0.9021025, 0.9456860, 0.9026748, 0.9471400, -0.0277611, 0.0240229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149742, upper bound: 0.0137180
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149742, upper bound: 0.0139274
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0038959, -0.0012514, -0.0038172, -0.0007483, -0.0030669, 0.0025658
1: 0.0190933, 0.0318097, 0.0190400, 0.0317771, -0.0071438, 0.0068355
2: 0.0210002, 0.0297746, 0.0210736, 0.0297738, -0.0050739, 0.0051894
3: 0.0074854, 0.0168665, 0.0074522, 0.0173516, -0.0066738, 0.0066856
4: -0.0176773, -0.0082776, -0.0183815, -0.0084225, -0.0074987, 0.0079122
5: 0.0140686, 0.0257214, 0.0142597, 0.0264360, -0.0094781, 0.0089207
6: 0.0059448, 0.0147520, 0.0060303, 0.0153710, -0.0071874, 0.0069666
7: -0.0223207, -0.0127521, -0.0228020, -0.0129793, -0.0079137, 0.0084934
8: 0.0091033, 0.0184851, 0.0090929, 0.0189046, -0.0066727, 0.0064455
9: 0.9021602, 0.9441911, 0.9004279, 0.9440972, -0.0270607, 0.0272941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0139860
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148435, upper bound: 0.0140992
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0039145, -0.0012636, -0.0036952, -0.0013653, -0.0025492, 0.0024316
1: 0.0184181, 0.0316530, 0.0189931, 0.0317726, -0.0071900, 0.0061441
2: 0.0205435, 0.0296403, 0.0205489, 0.0293141, -0.0046655, 0.0051434
3: 0.0067288, 0.0167444, 0.0072348, 0.0171822, -0.0065862, 0.0057573
4: -0.0175591, -0.0075858, -0.0178438, -0.0077787, -0.0070392, 0.0071736
5: 0.0132685, 0.0255931, 0.0138806, 0.0259592, -0.0087977, 0.0079894
6: 0.0052648, 0.0146432, 0.0056247, 0.0149805, -0.0067053, 0.0062692
7: -0.0221921, -0.0120437, -0.0224293, -0.0126310, -0.0071472, 0.0077382
8: 0.0084396, 0.0183907, 0.0090965, 0.0188012, -0.0066615, 0.0054787
9: 0.9026718, 0.9472603, 0.9011671, 0.9462265, -0.0246881, 0.0265310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152647
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154616, upper bound: 0.0152647
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0038231, -0.0007478, -0.0035990, -0.0013831, -0.0024400, 0.0028512
1: 0.0189903, 0.0317775, 0.0193642, 0.0317726, -0.0067659, 0.0060202
2: 0.0210448, 0.0297834, 0.0207534, 0.0291780, -0.0043779, 0.0052317
3: 0.0074231, 0.0173521, 0.0076666, 0.0171714, -0.0063633, 0.0064789
4: -0.0183832, -0.0083964, -0.0178021, -0.0081580, -0.0082082, 0.0069312
5: 0.0142238, 0.0264376, 0.0144532, 0.0259221, -0.0084200, 0.0089665
6: 0.0060052, 0.0153723, 0.0060265, 0.0149501, -0.0064639, 0.0072167
7: -0.0228039, -0.0129404, -0.0223840, -0.0131398, -0.0079646, 0.0073540
8: 0.0090632, 0.0189046, 0.0095645, 0.0188012, -0.0064123, 0.0059244
9: 0.9004252, 0.9442251, 0.9012308, 0.9447064, -0.0280127, 0.0257838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152410
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154616, upper bound: 0.0152410
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039145, -0.0012636, -0.0039966, -0.0012092, -0.0027053, 0.0027330
1: 0.0184181, 0.0316530, 0.0182774, 0.0316534, -0.0070889, 0.0077097
2: 0.0205435, 0.0296403, 0.0205461, 0.0297974, -0.0054358, 0.0053233
3: 0.0067288, 0.0167444, 0.0067967, 0.0167568, -0.0068228, 0.0062054
4: -0.0175591, -0.0075858, -0.0176080, -0.0076590, -0.0069579, 0.0076402
5: 0.0132685, 0.0255931, 0.0132203, 0.0256370, -0.0091958, 0.0087685
6: 0.0052648, 0.0146432, 0.0053210, 0.0146766, -0.0071203, 0.0064301
7: -0.0221921, -0.0120437, -0.0222437, -0.0119716, -0.0079063, 0.0081959
8: 0.0084396, 0.0183907, 0.0084095, 0.0183907, -0.0066424, 0.0064973
9: 0.9026718, 0.9472603, 0.9025958, 0.9468501, -0.0252235, 0.0273652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152410, upper bound: 0.0155486
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152410, upper bound: 0.0159943
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038231, -0.0007478, -0.0038998, -0.0012571, -0.0025660, 0.0030865
1: 0.0189903, 0.0317775, 0.0187421, 0.0316534, -0.0066667, 0.0074491
2: 0.0210448, 0.0297834, 0.0207435, 0.0296598, -0.0050364, 0.0053419
3: 0.0074231, 0.0173521, 0.0072454, 0.0167472, -0.0066065, 0.0069108
4: -0.0183832, -0.0083964, -0.0175700, -0.0080340, -0.0081543, 0.0074038
5: 0.0142238, 0.0264376, 0.0138111, 0.0256026, -0.0088264, 0.0097216
6: 0.0060052, 0.0153723, 0.0057273, 0.0146507, -0.0068870, 0.0073982
7: -0.0228039, -0.0129404, -0.0222036, -0.0124999, -0.0087283, 0.0078158
8: 0.0090632, 0.0189046, 0.0089131, 0.0183907, -0.0063944, 0.0068572
9: 0.9004252, 0.9442251, 0.9026550, 0.9453450, -0.0284815, 0.0266434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154616, upper bound: 0.0152410
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160988, upper bound: 0.0160988
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036900, -0.0013957, -0.0038422, -0.0012721, -0.0024179, 0.0024465
1: 0.0190905, 0.0317720, 0.0188281, 0.0318816, -0.0064210, 0.0067834
2: 0.0206027, 0.0292982, 0.0207318, 0.0296723, -0.0051316, 0.0044788
3: 0.0072622, 0.0171812, 0.0069670, 0.0169411, -0.0059602, 0.0061455
4: -0.0178365, -0.0077987, -0.0177008, -0.0077708, -0.0068107, 0.0071778
5: 0.0139159, 0.0259555, 0.0135839, 0.0257688, -0.0081541, 0.0082610
6: 0.0056481, 0.0149728, 0.0054734, 0.0147890, -0.0064161, 0.0063073
7: -0.0224247, -0.0126615, -0.0223285, -0.0123504, -0.0072247, 0.0072572
8: 0.0091288, 0.0188012, 0.0086977, 0.0185713, -0.0056997, 0.0062204
9: 0.9011732, 0.9460847, 0.9019033, 0.9463536, -0.0250798, 0.0255268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0138747, upper bound: 0.0148278
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039904, -0.0012143, -0.0039156, -0.0012500, -0.0027404, 0.0027012
1: 0.0184033, 0.0316527, 0.0186361, 0.0318816, -0.0079480, 0.0070688
2: 0.0206251, 0.0297886, 0.0207185, 0.0297910, -0.0054522, 0.0053174
3: 0.0068877, 0.0167563, 0.0068227, 0.0169467, -0.0064145, 0.0067202
4: -0.0176056, -0.0077438, -0.0177222, -0.0077118, -0.0074866, 0.0070498
5: 0.0133243, 0.0256350, 0.0133824, 0.0257879, -0.0089086, 0.0090730
6: 0.0054019, 0.0146749, 0.0053620, 0.0148044, -0.0065458, 0.0069944
7: -0.0222413, -0.0120683, -0.0223516, -0.0121711, -0.0080587, 0.0079770
8: 0.0084913, 0.0183907, 0.0084916, 0.0185713, -0.0067221, 0.0066055
9: 0.9025991, 0.9464516, 0.9018689, 0.9466523, -0.0268472, 0.0260574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145360, upper bound: 0.0148031
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036900, -0.0013957, -0.0038040, -0.0007343, -0.0029557, 0.0024083
1: 0.0190905, 0.0317720, 0.0192970, 0.0319850, -0.0065873, 0.0065283
2: 0.0206027, 0.0292982, 0.0212485, 0.0298815, -0.0053752, 0.0043198
3: 0.0072622, 0.0171812, 0.0076188, 0.0174859, -0.0068770, 0.0059726
4: -0.0178365, -0.0077987, -0.0184878, -0.0085774, -0.0066125, 0.0085186
5: 0.0139159, 0.0259555, 0.0144358, 0.0265555, -0.0094855, 0.0080350
6: 0.0056481, 0.0149728, 0.0061723, 0.0154747, -0.0075658, 0.0061337
7: -0.0224247, -0.0126615, -0.0229168, -0.0131538, -0.0070032, 0.0084224
8: 0.0091288, 0.0188012, 0.0092345, 0.0190317, -0.0063754, 0.0060765
9: 0.9011732, 0.9460847, 0.8998581, 0.9433160, -0.0242874, 0.0293392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039904, -0.0012143, -0.0038706, -0.0007279, -0.0032625, 0.0026563
1: 0.0184033, 0.0316527, 0.0191336, 0.0319850, -0.0081159, 0.0067653
2: 0.0206251, 0.0297886, 0.0212321, 0.0299914, -0.0055984, 0.0051567
3: 0.0068877, 0.0167563, 0.0074849, 0.0174912, -0.0072968, 0.0065269
4: -0.0176056, -0.0077438, -0.0185083, -0.0085150, -0.0072834, 0.0084342
5: 0.0133243, 0.0256350, 0.0142566, 0.0265737, -0.0102425, 0.0088161
6: 0.0054019, 0.0146749, 0.0060727, 0.0154898, -0.0077212, 0.0068069
7: -0.0222413, -0.0120683, -0.0229384, -0.0129804, -0.0078058, 0.0091986
8: 0.0084913, 0.0183907, 0.0090481, 0.0190317, -0.0073569, 0.0064178
9: 0.9025991, 0.9464516, 0.8998262, 0.9436309, -0.0260191, 0.0296681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0036952, -0.0013653, -0.0037789, -0.0012907, -0.0024044, 0.0024136
1: 0.0189931, 0.0317726, 0.0188031, 0.0317185, -0.0063942, 0.0066218
2: 0.0205489, 0.0293141, 0.0205975, 0.0294501, -0.0049434, 0.0045681
3: 0.0072348, 0.0171822, 0.0070528, 0.0168217, -0.0058825, 0.0060276
4: -0.0178438, -0.0077787, -0.0175815, -0.0077594, -0.0068309, 0.0070966
5: 0.0138806, 0.0259592, 0.0137107, 0.0256381, -0.0080797, 0.0080934
6: 0.0056247, 0.0149805, 0.0055309, 0.0146831, -0.0063357, 0.0062404
7: -0.0224293, -0.0126310, -0.0221985, -0.0124302, -0.0071089, 0.0071904
8: 0.0090965, 0.0188012, 0.0088640, 0.0184785, -0.0056382, 0.0060093
9: 0.9011671, 0.9462265, 0.9024225, 0.9464810, -0.0251393, 0.0252156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152527, upper bound: 0.0156279
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151434, upper bound: 0.0156279
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039966, -0.0012092, -0.0039175, -0.0012568, -0.0027398, 0.0027083
1: 0.0182774, 0.0316534, 0.0184070, 0.0317185, -0.0079593, 0.0072167
2: 0.0205461, 0.0297974, 0.0205439, 0.0296770, -0.0054521, 0.0054585
3: 0.0067967, 0.0167568, 0.0067016, 0.0168319, -0.0064080, 0.0068883
4: -0.0176080, -0.0076590, -0.0176213, -0.0075809, -0.0076516, 0.0070453
5: 0.0132203, 0.0256370, 0.0132559, 0.0256738, -0.0089218, 0.0092355
6: 0.0053210, 0.0146766, 0.0052512, 0.0147106, -0.0065527, 0.0071460
7: -0.0222437, -0.0119716, -0.0222407, -0.0120308, -0.0082108, 0.0079812
8: 0.0084095, 0.0183907, 0.0084054, 0.0184785, -0.0067212, 0.0067317
9: 0.9025958, 0.9468501, 0.9023612, 0.9473296, -0.0276297, 0.0260123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161630, upper bound: 0.0160080
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159786, upper bound: 0.0160080
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036952, -0.0013653, -0.0037400, -0.0007480, -0.0029472, 0.0023747
1: 0.0189931, 0.0317726, 0.0192381, 0.0318254, -0.0065590, 0.0063930
2: 0.0205489, 0.0293141, 0.0210939, 0.0296638, -0.0052527, 0.0044167
3: 0.0072348, 0.0171822, 0.0076884, 0.0173663, -0.0067888, 0.0058717
4: -0.0178438, -0.0077787, -0.0183727, -0.0085472, -0.0066481, 0.0084224
5: 0.0138806, 0.0259592, 0.0145386, 0.0264273, -0.0093891, 0.0078897
6: 0.0056247, 0.0149805, 0.0062131, 0.0153623, -0.0074746, 0.0060843
7: -0.0224293, -0.0126310, -0.0227914, -0.0132093, -0.0069048, 0.0083441
8: 0.0090965, 0.0188012, 0.0093842, 0.0189295, -0.0063037, 0.0058830
9: 0.9011671, 0.9462265, 0.9003714, 0.9434901, -0.0243834, 0.0289587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151434, upper bound: 0.0156279
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152076, upper bound: 0.0156279
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039966, -0.0012092, -0.0038730, -0.0007367, -0.0032599, 0.0026638
1: 0.0182774, 0.0316534, 0.0188819, 0.0318254, -0.0081316, 0.0069259
2: 0.0205461, 0.0297974, 0.0210367, 0.0298818, -0.0056068, 0.0053053
3: 0.0067967, 0.0167568, 0.0073453, 0.0173764, -0.0072750, 0.0067103
4: -0.0176080, -0.0076590, -0.0184107, -0.0083567, -0.0074588, 0.0084025
5: 0.0132203, 0.0256370, 0.0141068, 0.0264618, -0.0102120, 0.0089931
6: 0.0053210, 0.0146766, 0.0059454, 0.0153905, -0.0076956, 0.0069695
7: -0.0222437, -0.0119716, -0.0228296, -0.0128086, -0.0079710, 0.0091655
8: 0.0084095, 0.0183907, 0.0089450, 0.0189295, -0.0073434, 0.0065599
9: 0.9025958, 0.9468501, 0.9003108, 0.9443833, -0.0268513, 0.0295633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159786, upper bound: 0.0160080
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A1_B2_B2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160300, upper bound: 0.0160080
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0038422, -0.0012721, -0.0036900, -0.0013957, -0.0024465, 0.0024179
1: 0.0188281, 0.0318816, 0.0190905, 0.0317720, -0.0067834, 0.0064210
2: 0.0207318, 0.0296723, 0.0206027, 0.0292982, -0.0044788, 0.0051316
3: 0.0069670, 0.0169411, 0.0072622, 0.0171812, -0.0061455, 0.0059602
4: -0.0177008, -0.0077708, -0.0178365, -0.0077987, -0.0071778, 0.0068107
5: 0.0135839, 0.0257688, 0.0139159, 0.0259555, -0.0082610, 0.0081541
6: 0.0054734, 0.0147890, 0.0056481, 0.0149728, -0.0063073, 0.0064161
7: -0.0223285, -0.0123504, -0.0224247, -0.0126615, -0.0072572, 0.0072247
8: 0.0086977, 0.0185713, 0.0091288, 0.0188012, -0.0062204, 0.0056997
9: 0.9019033, 0.9463536, 0.9011732, 0.9460847, -0.0255268, 0.0250798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148278, upper bound: 0.0138747
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039156, -0.0012500, -0.0039904, -0.0012143, -0.0027012, 0.0027404
1: 0.0186361, 0.0318816, 0.0184033, 0.0316527, -0.0070688, 0.0079480
2: 0.0207185, 0.0297910, 0.0206251, 0.0297886, -0.0053174, 0.0054522
3: 0.0068227, 0.0169467, 0.0068877, 0.0167563, -0.0067202, 0.0064145
4: -0.0177222, -0.0077118, -0.0176056, -0.0077438, -0.0070498, 0.0074866
5: 0.0133824, 0.0257879, 0.0133243, 0.0256350, -0.0090730, 0.0089086
6: 0.0053621, 0.0148044, 0.0054019, 0.0146749, -0.0069944, 0.0065458
7: -0.0223516, -0.0121711, -0.0222413, -0.0120683, -0.0079770, 0.0080587
8: 0.0084916, 0.0185713, 0.0084913, 0.0183907, -0.0066055, 0.0067221
9: 0.9018689, 0.9466523, 0.9025991, 0.9464516, -0.0260574, 0.0268472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148031, upper bound: 0.0145360
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0038040, -0.0007343, -0.0036900, -0.0013957, -0.0024083, 0.0029557
1: 0.0192970, 0.0319850, 0.0190905, 0.0317720, -0.0065283, 0.0065873
2: 0.0212485, 0.0298815, 0.0206027, 0.0292982, -0.0043198, 0.0053752
3: 0.0076188, 0.0174859, 0.0072622, 0.0171812, -0.0059726, 0.0068770
4: -0.0184878, -0.0085774, -0.0178365, -0.0077987, -0.0085186, 0.0066125
5: 0.0144358, 0.0265555, 0.0139159, 0.0259555, -0.0080350, 0.0094855
6: 0.0061723, 0.0154747, 0.0056481, 0.0149728, -0.0061337, 0.0075658
7: -0.0229168, -0.0131538, -0.0224247, -0.0126615, -0.0084224, 0.0070032
8: 0.0092345, 0.0190317, 0.0091288, 0.0188012, -0.0060765, 0.0063754
9: 0.8998581, 0.9433160, 0.9011732, 0.9460847, -0.0293392, 0.0242874

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
time: 0.72 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0038706, -0.0007279, -0.0039904, -0.0012143, -0.0026563, 0.0032625
1: 0.0191336, 0.0319850, 0.0184033, 0.0316527, -0.0067653, 0.0081159
2: 0.0212321, 0.0299914, 0.0206251, 0.0297886, -0.0051567, 0.0055984
3: 0.0074849, 0.0174912, 0.0068877, 0.0167563, -0.0065269, 0.0072968
4: -0.0185083, -0.0085151, -0.0176056, -0.0077438, -0.0084342, 0.0072834
5: 0.0142566, 0.0265737, 0.0133243, 0.0256350, -0.0088161, 0.0102425
6: 0.0060727, 0.0154898, 0.0054019, 0.0146749, -0.0068069, 0.0077212
7: -0.0229384, -0.0129804, -0.0222413, -0.0120683, -0.0091986, 0.0078058
8: 0.0090481, 0.0190317, 0.0084913, 0.0183907, -0.0064178, 0.0073569
9: 0.8998262, 0.9436309, 0.9025991, 0.9464516, -0.0296681, 0.0260191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0037789, -0.0012907, -0.0036952, -0.0013653, -0.0024136, 0.0024044
1: 0.0188031, 0.0317185, 0.0189931, 0.0317726, -0.0066218, 0.0063942
2: 0.0205975, 0.0294501, 0.0205489, 0.0293141, -0.0045681, 0.0049434
3: 0.0070528, 0.0168217, 0.0072348, 0.0171822, -0.0060276, 0.0058825
4: -0.0175815, -0.0077594, -0.0178438, -0.0077787, -0.0070966, 0.0068309
5: 0.0137107, 0.0256381, 0.0138806, 0.0259592, -0.0080934, 0.0080797
6: 0.0055309, 0.0146831, 0.0056247, 0.0149805, -0.0062404, 0.0063357
7: -0.0221985, -0.0124302, -0.0224293, -0.0126310, -0.0071904, 0.0071089
8: 0.0088640, 0.0184785, 0.0090965, 0.0188012, -0.0060093, 0.0056382
9: 0.9024225, 0.9464810, 0.9011671, 0.9462265, -0.0252156, 0.0251393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152527
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0151434
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039175, -0.0012568, -0.0039966, -0.0012092, -0.0027083, 0.0027398
1: 0.0184070, 0.0317185, 0.0182774, 0.0316534, -0.0072167, 0.0079593
2: 0.0205439, 0.0296770, 0.0205461, 0.0297974, -0.0054585, 0.0054521
3: 0.0067016, 0.0168319, 0.0067967, 0.0167568, -0.0068883, 0.0064080
4: -0.0176213, -0.0075809, -0.0176080, -0.0076590, -0.0070453, 0.0076516
5: 0.0132559, 0.0256738, 0.0132203, 0.0256370, -0.0092355, 0.0089218
6: 0.0052512, 0.0147106, 0.0053210, 0.0146766, -0.0071460, 0.0065527
7: -0.0222407, -0.0120308, -0.0222437, -0.0119716, -0.0079812, 0.0082108
8: 0.0084054, 0.0184785, 0.0084095, 0.0183907, -0.0067317, 0.0067212
9: 0.9023612, 0.9473296, 0.9025958, 0.9468501, -0.0260123, 0.0276297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0161630
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0159786
time: 0.76 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0037400, -0.0007480, -0.0036952, -0.0013653, -0.0023747, 0.0029472
1: 0.0192381, 0.0318254, 0.0189931, 0.0317726, -0.0063930, 0.0065590
2: 0.0210939, 0.0296638, 0.0205489, 0.0293141, -0.0044167, 0.0052527
3: 0.0076884, 0.0173663, 0.0072348, 0.0171822, -0.0058717, 0.0067888
4: -0.0183727, -0.0085472, -0.0178438, -0.0077787, -0.0084224, 0.0066481
5: 0.0145386, 0.0264273, 0.0138806, 0.0259592, -0.0078897, 0.0093891
6: 0.0062131, 0.0153623, 0.0056247, 0.0149805, -0.0060843, 0.0074746
7: -0.0227914, -0.0132093, -0.0224293, -0.0126310, -0.0083441, 0.0069048
8: 0.0093842, 0.0189295, 0.0090965, 0.0188012, -0.0058830, 0.0063037
9: 0.9003714, 0.9434901, 0.9011671, 0.9462265, -0.0289587, 0.0243834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0151434
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152076
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0038730, -0.0007367, -0.0039966, -0.0012092, -0.0026638, 0.0032599
1: 0.0188819, 0.0318254, 0.0182774, 0.0316534, -0.0069259, 0.0081316
2: 0.0210367, 0.0298818, 0.0205461, 0.0297974, -0.0053053, 0.0056068
3: 0.0073453, 0.0173764, 0.0067967, 0.0167568, -0.0067103, 0.0072750
4: -0.0184107, -0.0083567, -0.0176080, -0.0076590, -0.0084025, 0.0074588
5: 0.0141068, 0.0264618, 0.0132203, 0.0256370, -0.0089931, 0.0102121
6: 0.0059454, 0.0153905, 0.0053210, 0.0146766, -0.0069695, 0.0076956
7: -0.0228296, -0.0128086, -0.0222437, -0.0119716, -0.0091655, 0.0079710
8: 0.0089450, 0.0189295, 0.0084095, 0.0183907, -0.0065599, 0.0073434
9: 0.9003108, 0.9443833, 0.9025958, 0.9468501, -0.0295633, 0.0268513

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0159786
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0160300
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0038454, -0.0012780, -0.0037909, -0.0014370, -0.0024084, 0.0025129
1: 0.0186401, 0.0317182, 0.0189839, 0.0319714, -0.0070562, 0.0064202
2: 0.0205915, 0.0295626, 0.0206837, 0.0295638, -0.0049780, 0.0048870
3: 0.0068946, 0.0168266, 0.0070322, 0.0172801, -0.0063060, 0.0060612
4: -0.0176009, -0.0076848, -0.0179316, -0.0077001, -0.0071852, 0.0070096
5: 0.0135026, 0.0256554, 0.0136025, 0.0260690, -0.0084628, 0.0084087
6: 0.0054064, 0.0146964, 0.0054656, 0.0150563, -0.0064736, 0.0065154
7: -0.0222191, -0.0122505, -0.0225384, -0.0124276, -0.0074713, 0.0074548
8: 0.0086502, 0.0184785, 0.0087943, 0.0188888, -0.0063297, 0.0059012
9: 0.9023926, 0.9468184, 0.9006733, 0.9464105, -0.0252148, 0.0259777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141666, upper bound: 0.0146558
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141666, upper bound: 0.0146558
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0038941, -0.0012629, -0.0040285, -0.0011629, -0.0027312, 0.0027656
1: 0.0185373, 0.0317182, 0.0185034, 0.0318816, -0.0072676, 0.0076873
2: 0.0205967, 0.0296478, 0.0207730, 0.0299970, -0.0057258, 0.0050876
3: 0.0068399, 0.0168309, 0.0069182, 0.0169626, -0.0069099, 0.0061687
4: -0.0176175, -0.0076787, -0.0177833, -0.0078421, -0.0068663, 0.0077657
5: 0.0134081, 0.0256704, 0.0133256, 0.0258420, -0.0092982, 0.0087882
6: 0.0053643, 0.0147079, 0.0054436, 0.0148479, -0.0072153, 0.0063928
7: -0.0222366, -0.0121601, -0.0224170, -0.0120990, -0.0078961, 0.0083051
8: 0.0085516, 0.0184785, 0.0084433, 0.0185713, -0.0066910, 0.0065766
9: 0.9023670, 0.9468699, 0.9017717, 0.9460111, -0.0246653, 0.0278585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140765, upper bound: 0.0151040
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0038040, -0.0007433, -0.0037909, -0.0014370, -0.0023670, 0.0030475
1: 0.0191023, 0.0318250, 0.0189839, 0.0319714, -0.0068920, 0.0066516
2: 0.0210916, 0.0297702, 0.0206837, 0.0295638, -0.0048532, 0.0052192
3: 0.0075356, 0.0173712, 0.0070322, 0.0172801, -0.0061860, 0.0071201
4: -0.0183911, -0.0084722, -0.0179316, -0.0077001, -0.0086243, 0.0068239
5: 0.0143474, 0.0264439, 0.0136025, 0.0260690, -0.0082618, 0.0098642
6: 0.0060960, 0.0153758, 0.0054656, 0.0150563, -0.0063229, 0.0077731
7: -0.0228098, -0.0130350, -0.0225384, -0.0124276, -0.0087194, 0.0072353
8: 0.0091822, 0.0189295, 0.0087943, 0.0188888, -0.0062440, 0.0067233
9: 0.9003421, 0.9438405, 0.9006733, 0.9464105, -0.0295780, 0.0254098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140835, upper bound: 0.0144218
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0140835, upper bound: 0.0144218
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0038464, -0.0007391, -0.0040285, -0.0011629, -0.0026835, 0.0032838
1: 0.0190135, 0.0318250, 0.0185034, 0.0318816, -0.0070472, 0.0078989
2: 0.0210942, 0.0298446, 0.0207730, 0.0299970, -0.0056011, 0.0053017
3: 0.0074877, 0.0173754, 0.0069182, 0.0169626, -0.0067604, 0.0072227
4: -0.0184071, -0.0084540, -0.0177833, -0.0078421, -0.0083629, 0.0075691
5: 0.0142650, 0.0264585, 0.0133256, 0.0258420, -0.0090564, 0.0102849
6: 0.0060607, 0.0153878, 0.0054436, 0.0148479, -0.0070422, 0.0077037
7: -0.0228260, -0.0129446, -0.0224170, -0.0120990, -0.0092017, 0.0080520
8: 0.0090973, 0.0189295, 0.0084433, 0.0185713, -0.0065612, 0.0073564
9: 0.9003165, 0.9439163, 0.9017717, 0.9460111, -0.0289642, 0.0272476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0037327, -0.0013499, -0.0037761, -0.0012911, -0.0024416, 0.0024263
1: 0.0188639, 0.0318358, 0.0188109, 0.0317185, -0.0063917, 0.0065397
2: 0.0205187, 0.0294137, 0.0205986, 0.0294454, -0.0049126, 0.0046661
3: 0.0071059, 0.0172722, 0.0070598, 0.0168214, -0.0059686, 0.0059944
4: -0.0179279, -0.0076984, -0.0175807, -0.0077630, -0.0068632, 0.0072055
5: 0.0137106, 0.0260548, 0.0137198, 0.0256374, -0.0082733, 0.0080981
6: 0.0055191, 0.0150628, 0.0055364, 0.0146825, -0.0064569, 0.0062523
7: -0.0225028, -0.0124930, -0.0221977, -0.0124381, -0.0071350, 0.0073902
8: 0.0089340, 0.0188835, 0.0088731, 0.0184785, -0.0057271, 0.0059631
9: 0.9008107, 0.9465519, 0.9024237, 0.9464642, -0.0250348, 0.0253471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152357, upper bound: 0.0156219
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152357, upper bound: 0.0156406
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040352, -0.0011705, -0.0039175, -0.0012568, -0.0027784, 0.0027470
1: 0.0181509, 0.0317189, 0.0184070, 0.0317185, -0.0080049, 0.0071877
2: 0.0205160, 0.0298880, 0.0205439, 0.0296770, -0.0054052, 0.0055916
3: 0.0066728, 0.0168480, 0.0067016, 0.0168319, -0.0064112, 0.0068747
4: -0.0176841, -0.0075965, -0.0176213, -0.0075809, -0.0076878, 0.0071134
5: 0.0130656, 0.0257308, 0.0132559, 0.0256738, -0.0090383, 0.0092581
6: 0.0052279, 0.0147541, 0.0052512, 0.0147106, -0.0066111, 0.0071657
7: -0.0223074, -0.0118415, -0.0222407, -0.0120308, -0.0082485, 0.0081412
8: 0.0082448, 0.0184785, 0.0084054, 0.0184785, -0.0067633, 0.0067035
9: 0.9022639, 0.9471750, 0.9023612, 0.9473296, -0.0275692, 0.0258587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160932, upper bound: 0.0160120
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159532, upper bound: 0.0160120
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0037327, -0.0013499, -0.0037374, -0.0007481, -0.0029846, 0.0023875
1: 0.0188639, 0.0318358, 0.0192446, 0.0318254, -0.0066233, 0.0064206
2: 0.0205187, 0.0294137, 0.0210950, 0.0296593, -0.0053009, 0.0045489
3: 0.0071059, 0.0172722, 0.0076952, 0.0173661, -0.0070276, 0.0058873
4: -0.0179279, -0.0076984, -0.0183720, -0.0085508, -0.0066770, 0.0086453
5: 0.0137106, 0.0260548, 0.0145470, 0.0264266, -0.0097298, 0.0079153
6: 0.0055191, 0.0150628, 0.0062183, 0.0153617, -0.0077149, 0.0061087
7: -0.0225028, -0.0124930, -0.0227907, -0.0132172, -0.0069344, 0.0086389
8: 0.0089340, 0.0188835, 0.0093929, 0.0189295, -0.0065478, 0.0058996
9: 0.9008107, 0.9465519, 0.9003727, 0.9434724, -0.0244609, 0.0297131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151371, upper bound: 0.0156219
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151371, upper bound: 0.0156406
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040352, -0.0011705, -0.0038730, -0.0007367, -0.0032985, 0.0027025
1: 0.0181509, 0.0317189, 0.0188819, 0.0318254, -0.0082165, 0.0069777
2: 0.0205160, 0.0298880, 0.0210367, 0.0298818, -0.0056091, 0.0054718
3: 0.0066728, 0.0168480, 0.0073453, 0.0173764, -0.0074651, 0.0067321
4: -0.0176841, -0.0075965, -0.0184107, -0.0083567, -0.0074912, 0.0086096
5: 0.0130656, 0.0257308, 0.0141068, 0.0264618, -0.0105343, 0.0090280
6: 0.0052279, 0.0147541, 0.0059454, 0.0153905, -0.0079216, 0.0069974
7: -0.0223074, -0.0118415, -0.0228296, -0.0128086, -0.0080086, 0.0094465
8: 0.0082448, 0.0184785, 0.0089450, 0.0189295, -0.0075433, 0.0065862
9: 0.9022639, 0.9471750, 0.9003108, 0.9443833, -0.0269533, 0.0301564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159532, upper bound: 0.0159694
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159532, upper bound: 0.0160120
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040950, -0.0011180, -0.0036046, -0.0014132, -0.0026544, 0.0024866
1: 0.0180102, 0.0318097, 0.0186341, 0.0320361, -0.0087856, 0.0071657
2: 0.0205854, 0.0300432, 0.0201282, 0.0293411, -0.0050724, 0.0064826
3: 0.0062686, 0.0168785, 0.0068590, 0.0174577, -0.0077316, 0.0065390
4: -0.0177240, -0.0073628, -0.0181188, -0.0072491, -0.0078127, 0.0079439
5: 0.0126354, 0.0257630, 0.0134670, 0.0262796, -0.0102471, 0.0086809
6: 0.0048994, 0.0147838, 0.0052314, 0.0152272, -0.0076356, 0.0069301
7: -0.0223695, -0.0115517, -0.0226657, -0.0123038, -0.0075616, 0.0088830
8: 0.0077603, 0.0184851, 0.0088535, 0.0190545, -0.0079956, 0.0060931
9: 0.9020864, 0.9481564, 0.8999601, 0.9484153, -0.0290229, 0.0305130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151044, upper bound: 0.0143119
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0151044, upper bound: 0.0143119
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040116, -0.0012026, -0.0034915, -0.0008760, -0.0030427, 0.0022889
1: 0.0184520, 0.0318097, 0.0191843, 0.0321678, -0.0085221, 0.0069272
2: 0.0207677, 0.0299127, 0.0206492, 0.0294688, -0.0052831, 0.0060072
3: 0.0066718, 0.0168704, 0.0076742, 0.0180026, -0.0081674, 0.0062220
4: -0.0176926, -0.0077117, -0.0189000, -0.0081135, -0.0073600, 0.0088220
5: 0.0131813, 0.0257349, 0.0145090, 0.0270554, -0.0108698, 0.0080874
6: 0.0052757, 0.0147623, 0.0060399, 0.0159152, -0.0083041, 0.0065174
7: -0.0223366, -0.0120391, -0.0232546, -0.0132445, -0.0069633, 0.0094762
8: 0.0082108, 0.0184851, 0.0096094, 0.0195262, -0.0081814, 0.0058477
9: 0.9021362, 0.9467839, 0.8978013, 0.9451128, -0.0277995, 0.0326077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147146, upper bound: 0.0141498
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147146, upper bound: 0.0141498
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039196, -0.0012547, -0.0040582, -0.0011684, -0.0027511, 0.0028035
1: 0.0186156, 0.0318097, 0.0174437, 0.0319168, -0.0077328, 0.0090375
2: 0.0207098, 0.0297621, 0.0200498, 0.0300308, -0.0058559, 0.0061860
3: 0.0068008, 0.0168640, 0.0060326, 0.0170190, -0.0070578, 0.0072413
4: -0.0176678, -0.0076784, -0.0179337, -0.0068965, -0.0078356, 0.0078038
5: 0.0133408, 0.0257128, 0.0122186, 0.0259860, -0.0094854, 0.0098803
6: 0.0053318, 0.0147454, 0.0045791, 0.0149628, -0.0072782, 0.0072911
7: -0.0223108, -0.0121375, -0.0225280, -0.0111824, -0.0087302, 0.0083676
8: 0.0084705, 0.0184851, 0.0076451, 0.0186232, -0.0069571, 0.0075215
9: 0.9021754, 0.9467339, 0.9014003, 0.9500028, -0.0301087, 0.0286682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144327, upper bound: 0.0154124
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0144327, upper bound: 0.0149794
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038331, -0.0007389, -0.0039669, -0.0012338, -0.0025993, 0.0032087
1: 0.0192018, 0.0319310, 0.0178798, 0.0319168, -0.0072875, 0.0087879
2: 0.0212273, 0.0299071, 0.0202268, 0.0299120, -0.0054807, 0.0061492
3: 0.0075075, 0.0174673, 0.0064562, 0.0170107, -0.0068033, 0.0079006
4: -0.0184836, -0.0085166, -0.0179016, -0.0072414, -0.0089934, 0.0075280
5: 0.0143122, 0.0265500, 0.0127822, 0.0259595, -0.0090762, 0.0108036
6: 0.0060864, 0.0154719, 0.0049633, 0.0149409, -0.0070101, 0.0082209
7: -0.0229138, -0.0130641, -0.0224953, -0.0116686, -0.0095626, 0.0079600
8: 0.0091050, 0.0190066, 0.0081254, 0.0186232, -0.0066740, 0.0078396
9: 0.8999424, 0.9436350, 0.9014523, 0.9486332, -0.0331222, 0.0277587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145600, upper bound: 0.0141498
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147207, upper bound: 0.0148894
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0036094, -0.0013833, -0.0026918, 0.0024713
1: 0.0178103, 0.0316530, 0.0185364, 0.0320369, -0.0089073, 0.0071232
2: 0.0204141, 0.0299231, 0.0200750, 0.0293535, -0.0052299, 0.0064180
3: 0.0061951, 0.0167591, 0.0068343, 0.0174586, -0.0078949, 0.0064538
4: -0.0176166, -0.0072584, -0.0181263, -0.0072294, -0.0077308, 0.0081259
5: 0.0125631, 0.0256449, 0.0134352, 0.0262829, -0.0104088, 0.0085993
6: 0.0048292, 0.0146824, 0.0052095, 0.0152351, -0.0077902, 0.0068488
7: -0.0222527, -0.0114462, -0.0226698, -0.0122760, -0.0074986, 0.0090472
8: 0.0077319, 0.0183907, 0.0088255, 0.0190545, -0.0081073, 0.0060458
9: 0.9025822, 0.9486928, 0.8999548, 0.9485529, -0.0286837, 0.0313064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156764, upper bound: 0.0160950
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0153857, upper bound: 0.0160950
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040072, -0.0012135, -0.0034961, -0.0008486, -0.0030759, 0.0022825
1: 0.0182436, 0.0316530, 0.0190835, 0.0321686, -0.0086499, 0.0068853
2: 0.0205896, 0.0297941, 0.0205946, 0.0294788, -0.0054227, 0.0059357
3: 0.0065901, 0.0167508, 0.0076492, 0.0180033, -0.0083255, 0.0061339
4: -0.0175842, -0.0076046, -0.0189050, -0.0080939, -0.0072775, 0.0090020
5: 0.0130949, 0.0256154, 0.0144790, 0.0270576, -0.0110297, 0.0080025
6: 0.0051990, 0.0146603, 0.0060190, 0.0159210, -0.0084556, 0.0064321
7: -0.0222186, -0.0119252, -0.0232573, -0.0132112, -0.0068985, 0.0096358
8: 0.0081740, 0.0183907, 0.0095821, 0.0195262, -0.0082940, 0.0057967
9: 0.9026326, 0.9473403, 0.8977976, 0.9452673, -0.0274656, 0.0333616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155044, upper bound: 0.0160950
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0155044, upper bound: 0.0160950
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0039145, -0.0012636, -0.0040643, -0.0011620, -0.0027525, 0.0028007
1: 0.0184181, 0.0316530, 0.0173192, 0.0319175, -0.0078357, 0.0090391
2: 0.0205435, 0.0296403, 0.0199713, 0.0300403, -0.0059917, 0.0061534
3: 0.0067288, 0.0167444, 0.0059393, 0.0170196, -0.0072121, 0.0072159
4: -0.0175591, -0.0075858, -0.0179360, -0.0068119, -0.0078198, 0.0079626
5: 0.0132685, 0.0255931, 0.0121168, 0.0259878, -0.0096279, 0.0098658
6: 0.0052648, 0.0146432, 0.0044993, 0.0149644, -0.0074206, 0.0072776
7: -0.0221921, -0.0120437, -0.0225303, -0.0110853, -0.0087208, 0.0084972
8: 0.0084396, 0.0183907, 0.0075605, 0.0186232, -0.0070593, 0.0074978
9: 0.9026718, 0.9472603, 0.9013969, 0.9504046, -0.0300218, 0.0294102

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152805, upper bound: 0.0161356
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0171563
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0038231, -0.0007478, -0.0039739, -0.0012286, -0.0025945, 0.0031717
1: 0.0189903, 0.0317775, 0.0177505, 0.0319175, -0.0073860, 0.0088061
2: 0.0210448, 0.0297834, 0.0201459, 0.0299214, -0.0056243, 0.0061401
3: 0.0074231, 0.0173521, 0.0063604, 0.0170113, -0.0069590, 0.0078857
4: -0.0183832, -0.0083964, -0.0179042, -0.0071523, -0.0089661, 0.0076957
5: 0.0142238, 0.0264376, 0.0126730, 0.0259616, -0.0092198, 0.0107834
6: 0.0060052, 0.0153723, 0.0048786, 0.0149427, -0.0071560, 0.0081998
7: -0.0228039, -0.0129404, -0.0224979, -0.0115663, -0.0095354, 0.0080910
8: 0.0090632, 0.0189046, 0.0080348, 0.0186232, -0.0067733, 0.0078355
9: 0.9004252, 0.9442251, 0.9014485, 0.9490513, -0.0330234, 0.0285251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0152805, upper bound: 0.0160950
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0170393
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0037531, -0.0014536, -0.0039358, -0.0012584, -0.0024947, 0.0024822
1: 0.0191233, 0.0319055, 0.0176204, 0.0319700, -0.0070031, 0.0086461
2: 0.0207164, 0.0294683, 0.0200117, 0.0298508, -0.0054924, 0.0057491
3: 0.0071643, 0.0171908, 0.0059934, 0.0170978, -0.0063584, 0.0074689
4: -0.0178486, -0.0077817, -0.0179159, -0.0068370, -0.0079290, 0.0074101
5: 0.0137739, 0.0259752, 0.0123700, 0.0260117, -0.0086744, 0.0097372
6: 0.0055729, 0.0149748, 0.0045743, 0.0149724, -0.0067121, 0.0074087
7: -0.0224668, -0.0125670, -0.0225011, -0.0113214, -0.0084071, 0.0075838
8: 0.0089621, 0.0188062, 0.0077367, 0.0187207, -0.0061788, 0.0075397
9: 0.9010233, 0.9460711, 0.9011636, 0.9504850, -0.0311555, 0.0270795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145409, upper bound: 0.0148680
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143225, upper bound: 0.0147842
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0039902, -0.0012034, -0.0039789, -0.0012361, -0.0027541, 0.0027755
1: 0.0186371, 0.0318097, 0.0175189, 0.0319700, -0.0082256, 0.0087866
2: 0.0208021, 0.0299066, 0.0200180, 0.0299197, -0.0056980, 0.0064541
3: 0.0070426, 0.0168759, 0.0059493, 0.0171014, -0.0065872, 0.0079817
4: -0.0177137, -0.0079080, -0.0179301, -0.0068315, -0.0086287, 0.0072402
5: 0.0134865, 0.0257539, 0.0122830, 0.0260243, -0.0092042, 0.0104809
6: 0.0055432, 0.0147769, 0.0045405, 0.0149827, -0.0067388, 0.0080804
7: -0.0223589, -0.0122311, -0.0225160, -0.0112357, -0.0092006, 0.0081433
8: 0.0086056, 0.0184851, 0.0076492, 0.0187207, -0.0068811, 0.0078223
9: 0.9021025, 0.9456860, 0.9011419, 0.9504979, -0.0327664, 0.0269361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149400, upper bound: 0.0148081
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0037531, -0.0014536, -0.0038585, -0.0007485, -0.0030046, 0.0024049
1: 0.0191233, 0.0319055, 0.0181772, 0.0320908, -0.0071591, 0.0082797
2: 0.0207164, 0.0294683, 0.0205070, 0.0300396, -0.0056654, 0.0054937
3: 0.0071643, 0.0171908, 0.0067601, 0.0175948, -0.0070718, 0.0071325
4: -0.0178486, -0.0077817, -0.0186998, -0.0076531, -0.0075404, 0.0084876
5: 0.0137739, 0.0259752, 0.0133289, 0.0267784, -0.0096911, 0.0092216
6: 0.0055729, 0.0149748, 0.0053356, 0.0156460, -0.0076243, 0.0070486
7: -0.0224668, -0.0125670, -0.0231128, -0.0121880, -0.0079208, 0.0085377
8: 0.0089621, 0.0188062, 0.0084201, 0.0191380, -0.0067123, 0.0072200
9: 0.9010233, 0.9460711, 0.8990978, 0.9473280, -0.0298484, 0.0299322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143225, upper bound: 0.0147842
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143276, upper bound: 0.0147842
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039902, -0.0012034, -0.0039013, -0.0007438, -0.0032464, 0.0026979
1: 0.0186371, 0.0318097, 0.0180917, 0.0320908, -0.0083713, 0.0083779
2: 0.0208021, 0.0299066, 0.0205115, 0.0301087, -0.0058035, 0.0061998
3: 0.0070426, 0.0168759, 0.0067103, 0.0175987, -0.0072736, 0.0076412
4: -0.0177137, -0.0079080, -0.0187146, -0.0076465, -0.0082521, 0.0082696
5: 0.0134865, 0.0257539, 0.0132504, 0.0267909, -0.0101950, 0.0099639
6: 0.0055432, 0.0147769, 0.0053016, 0.0156561, -0.0076201, 0.0077272
7: -0.0223589, -0.0122311, -0.0231279, -0.0121156, -0.0087097, 0.0090500
8: 0.0086056, 0.0184851, 0.0083392, 0.0191380, -0.0073890, 0.0074784
9: 0.9021025, 0.9456860, 0.8990747, 0.9473776, -0.0314445, 0.0297427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0039890, -0.0012300, -0.0036118, -0.0013768, -0.0025709, 0.0023818
1: 0.0181271, 0.0316530, 0.0185292, 0.0320996, -0.0084665, 0.0071336
2: 0.0204558, 0.0297538, 0.0200869, 0.0293880, -0.0051909, 0.0061346
3: 0.0064658, 0.0167523, 0.0068174, 0.0175390, -0.0075107, 0.0064781
4: -0.0175900, -0.0073922, -0.0181889, -0.0072288, -0.0077381, 0.0079390
5: 0.0129010, 0.0256208, 0.0134237, 0.0263647, -0.0099398, 0.0086237
6: 0.0050345, 0.0146642, 0.0051965, 0.0153070, -0.0074980, 0.0068672
7: -0.0222248, -0.0117515, -0.0227206, -0.0122730, -0.0075145, 0.0086264
8: 0.0080838, 0.0183907, 0.0088088, 0.0191372, -0.0076377, 0.0060711
9: 0.9026234, 0.9480568, 0.8996102, 0.9485875, -0.0287529, 0.0303952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157059, upper bound: 0.0160780
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154503, upper bound: 0.0160780
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0040929, -0.0011381, -0.0038928, -0.0012624, -0.0028304, 0.0027547
1: 0.0178103, 0.0316530, 0.0178827, 0.0319708, -0.0089428, 0.0083308
2: 0.0204141, 0.0299231, 0.0201110, 0.0298284, -0.0058340, 0.0065596
3: 0.0061951, 0.0167591, 0.0064673, 0.0170990, -0.0080625, 0.0066773
4: -0.0176166, -0.0072584, -0.0179207, -0.0071651, -0.0074833, 0.0084632
5: 0.0125631, 0.0256449, 0.0128393, 0.0260159, -0.0107067, 0.0090833
6: 0.0048292, 0.0146824, 0.0049502, 0.0149759, -0.0080864, 0.0068231
7: -0.0222527, -0.0114462, -0.0225062, -0.0116802, -0.0080214, 0.0093556
8: 0.0077319, 0.0183907, 0.0082192, 0.0187207, -0.0081244, 0.0067913
9: 0.9025822, 0.9486928, 0.9011564, 0.9489450, -0.0286043, 0.0319414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159484, upper bound: 0.0169337
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157738, upper bound: 0.0169337
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0036914, -0.0013881, -0.0039042, -0.0007431, -0.0029484, 0.0025161
1: 0.0190581, 0.0317723, 0.0179584, 0.0320916, -0.0070035, 0.0084327
2: 0.0205837, 0.0293029, 0.0204216, 0.0301110, -0.0058009, 0.0053838
3: 0.0072540, 0.0171814, 0.0065958, 0.0175982, -0.0069595, 0.0073385
4: -0.0178370, -0.0077924, -0.0187128, -0.0075465, -0.0076904, 0.0084850
5: 0.0139053, 0.0259562, 0.0131320, 0.0267893, -0.0095186, 0.0094711
6: 0.0056409, 0.0149725, 0.0052054, 0.0156549, -0.0075520, 0.0072304
7: -0.0224255, -0.0126518, -0.0231261, -0.0120075, -0.0081409, 0.0083990
8: 0.0091194, 0.0188012, 0.0082352, 0.0191380, -0.0065225, 0.0074598
9: 0.9011719, 0.9461287, 0.8990777, 0.9478631, -0.0303690, 0.0299089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149809, upper bound: 0.0166586
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0150554, upper bound: 0.0166586
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0039920, -0.0012132, -0.0039171, -0.0007410, -0.0032510, 0.0027039
1: 0.0183576, 0.0316530, 0.0179485, 0.0320916, -0.0084913, 0.0083945
2: 0.0205961, 0.0297911, 0.0204326, 0.0301328, -0.0059317, 0.0061661
3: 0.0068610, 0.0167564, 0.0066088, 0.0175998, -0.0073814, 0.0076376
4: -0.0176061, -0.0077139, -0.0187190, -0.0075645, -0.0082213, 0.0084065
5: 0.0132893, 0.0256355, 0.0131328, 0.0267946, -0.0103122, 0.0099625
6: 0.0053748, 0.0146753, 0.0052167, 0.0156591, -0.0077276, 0.0077081
7: -0.0222418, -0.0120382, -0.0231324, -0.0120058, -0.0087047, 0.0091728
8: 0.0084693, 0.0183907, 0.0082373, 0.0191380, -0.0074770, 0.0075168
9: 0.9025983, 0.9465885, 0.8990680, 0.9477819, -0.0313811, 0.0302946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157738, upper bound: 0.0169337
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158301, upper bound: 0.0169337
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0038488, -0.0012703, -0.0037763, -0.0013821, -0.0024667, 0.0025060
1: 0.0188081, 0.0318816, 0.0180819, 0.0320361, -0.0075673, 0.0080674
2: 0.0207294, 0.0296838, 0.0200002, 0.0295901, -0.0051420, 0.0060665
3: 0.0069498, 0.0169416, 0.0063578, 0.0174726, -0.0067595, 0.0072037
4: -0.0177028, -0.0077625, -0.0181796, -0.0069277, -0.0082094, 0.0073685
5: 0.0135615, 0.0257705, 0.0127861, 0.0263298, -0.0089640, 0.0095229
6: 0.0054602, 0.0147904, 0.0048096, 0.0152701, -0.0068525, 0.0074547
7: -0.0223306, -0.0123308, -0.0227282, -0.0117322, -0.0082789, 0.0077700
8: 0.0086750, 0.0185713, 0.0082174, 0.0190545, -0.0067572, 0.0069442
9: 0.9019001, 0.9463937, 0.8998722, 0.9498303, -0.0310522, 0.0279388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147293, upper bound: 0.0146145
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039165, -0.0012497, -0.0040582, -0.0011684, -0.0027481, 0.0028085
1: 0.0186311, 0.0318816, 0.0174437, 0.0319168, -0.0078316, 0.0092761
2: 0.0207166, 0.0297921, 0.0200498, 0.0300308, -0.0058780, 0.0062884
3: 0.0068160, 0.0169467, 0.0060326, 0.0170190, -0.0071120, 0.0074228
4: -0.0177223, -0.0077072, -0.0179337, -0.0068965, -0.0079111, 0.0078112
5: 0.0133753, 0.0257880, 0.0122186, 0.0259860, -0.0095078, 0.0100017
6: 0.0053567, 0.0148045, 0.0045791, 0.0149628, -0.0072961, 0.0073914
7: -0.0223517, -0.0121652, -0.0225280, -0.0111824, -0.0087895, 0.0083667
8: 0.0084845, 0.0185713, 0.0076451, 0.0186232, -0.0070259, 0.0077181
9: 0.9018687, 0.9466732, 0.9014003, 0.9500028, -0.0308439, 0.0288942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146669, upper bound: 0.0154260
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0038106, -0.0007337, -0.0037763, -0.0013821, -0.0024286, 0.0030426
1: 0.0192792, 0.0319850, 0.0180819, 0.0320361, -0.0072795, 0.0082337
2: 0.0212459, 0.0298926, 0.0200002, 0.0295901, -0.0049634, 0.0063040
3: 0.0076024, 0.0174863, 0.0063578, 0.0174726, -0.0065435, 0.0081204
4: -0.0184897, -0.0085693, -0.0181796, -0.0069277, -0.0095501, 0.0071370
5: 0.0144151, 0.0265572, 0.0127861, 0.0263298, -0.0086955, 0.0108542
6: 0.0061598, 0.0154761, 0.0048096, 0.0152701, -0.0066441, 0.0086044
7: -0.0229188, -0.0131345, -0.0227282, -0.0117322, -0.0094442, 0.0075204
8: 0.0092130, 0.0190317, 0.0082174, 0.0190545, -0.0065708, 0.0076200
9: 0.8998553, 0.9433578, 0.8998722, 0.9498303, -0.0348641, 0.0269788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0038718, -0.0007278, -0.0040582, -0.0011684, -0.0027034, 0.0033304
1: 0.0191283, 0.0319850, 0.0174437, 0.0319168, -0.0075006, 0.0094440
2: 0.0212304, 0.0299932, 0.0200498, 0.0300308, -0.0056991, 0.0064349
3: 0.0074779, 0.0174912, 0.0060326, 0.0170190, -0.0068837, 0.0083051
4: -0.0185085, -0.0085106, -0.0179337, -0.0068965, -0.0092955, 0.0075817
5: 0.0142488, 0.0265738, 0.0122186, 0.0259860, -0.0092165, 0.0113357
6: 0.0060670, 0.0154899, 0.0045791, 0.0149628, -0.0070807, 0.0085667
7: -0.0229385, -0.0129743, -0.0225280, -0.0111824, -0.0100111, 0.0080941
8: 0.0090405, 0.0190317, 0.0076451, 0.0186232, -0.0068001, 0.0083530
9: 0.8998262, 0.9436519, 0.9014003, 0.9500028, -0.0344545, 0.0279130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0038165, -0.0012844, -0.0037817, -0.0013522, -0.0024643, 0.0024973
1: 0.0186953, 0.0317185, 0.0179879, 0.0320369, -0.0075346, 0.0080530
2: 0.0205831, 0.0295145, 0.0199461, 0.0296071, -0.0052569, 0.0059441
3: 0.0069572, 0.0168245, 0.0063291, 0.0174736, -0.0067795, 0.0071322
4: -0.0175927, -0.0077104, -0.0181868, -0.0069071, -0.0081317, 0.0074661
5: 0.0135859, 0.0256481, 0.0127493, 0.0263337, -0.0089594, 0.0094529
6: 0.0054552, 0.0146908, 0.0047853, 0.0152778, -0.0068946, 0.0073775
7: -0.0222104, -0.0123220, -0.0227331, -0.0117005, -0.0082159, 0.0077938
8: 0.0087384, 0.0184785, 0.0081835, 0.0190545, -0.0067048, 0.0068934
9: 0.9024051, 0.9467139, 0.8998659, 0.9499702, -0.0307540, 0.0283416

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0161309
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160217
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039175, -0.0012568, -0.0040643, -0.0011620, -0.0027555, 0.0028075
1: 0.0184070, 0.0317185, 0.0173192, 0.0319175, -0.0079635, 0.0092886
2: 0.0205439, 0.0296770, 0.0199713, 0.0300403, -0.0060144, 0.0062823
3: 0.0067016, 0.0168319, 0.0059393, 0.0170196, -0.0072776, 0.0074184
4: -0.0176213, -0.0075809, -0.0179360, -0.0068119, -0.0079072, 0.0079740
5: 0.0132559, 0.0256738, 0.0121168, 0.0259878, -0.0096676, 0.0100191
6: 0.0052512, 0.0147106, 0.0044993, 0.0149644, -0.0074462, 0.0074001
7: -0.0222407, -0.0120308, -0.0225303, -0.0110853, -0.0087957, 0.0085120
8: 0.0084054, 0.0184785, 0.0075605, 0.0186232, -0.0071485, 0.0077217
9: 0.9023612, 0.9473296, 0.9013969, 0.9504046, -0.0308107, 0.0296747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0171183
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169345
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0037763, -0.0007455, -0.0037817, -0.0013522, -0.0024241, 0.0030362
1: 0.0191441, 0.0318254, 0.0179879, 0.0320369, -0.0072576, 0.0082179
2: 0.0210781, 0.0297247, 0.0199461, 0.0296071, -0.0050859, 0.0062217
3: 0.0075953, 0.0173691, 0.0063291, 0.0174736, -0.0065745, 0.0080382
4: -0.0183833, -0.0084968, -0.0181868, -0.0069071, -0.0094568, 0.0072478
5: 0.0144235, 0.0264368, 0.0127493, 0.0263337, -0.0087036, 0.0107614
6: 0.0061412, 0.0153701, 0.0047853, 0.0152778, -0.0066989, 0.0085159
7: -0.0228020, -0.0131008, -0.0227331, -0.0117005, -0.0093690, 0.0075525
8: 0.0092652, 0.0189295, 0.0081835, 0.0190545, -0.0065264, 0.0075587
9: 0.9003546, 0.9437329, 0.8998659, 0.9499702, -0.0344947, 0.0274165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160217
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160741
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0038730, -0.0007367, -0.0040643, -0.0011620, -0.0027110, 0.0033276
1: 0.0188819, 0.0318254, 0.0173192, 0.0319175, -0.0076432, 0.0094610
2: 0.0210367, 0.0298818, 0.0199713, 0.0300403, -0.0058435, 0.0064370
3: 0.0073453, 0.0173764, 0.0059393, 0.0170196, -0.0070640, 0.0082854
4: -0.0184107, -0.0083567, -0.0179360, -0.0068119, -0.0092643, 0.0077545
5: 0.0141068, 0.0264618, 0.0121168, 0.0259878, -0.0093899, 0.0113094
6: 0.0059454, 0.0153905, 0.0044993, 0.0149644, -0.0072413, 0.0085431
7: -0.0228296, -0.0128086, -0.0225303, -0.0110853, -0.0099800, 0.0082511
8: 0.0089450, 0.0189295, 0.0075605, 0.0186232, -0.0069374, 0.0083439
9: 0.9003108, 0.9443833, 0.9013969, 0.9504046, -0.0343617, 0.0287408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169345
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169503
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0037909, -0.0014370, -0.0039356, -0.0012585, -0.0025323, 0.0024986
1: 0.0189839, 0.0319714, 0.0176209, 0.0319700, -0.0071530, 0.0086779
2: 0.0206837, 0.0295638, 0.0200117, 0.0298505, -0.0055173, 0.0058644
3: 0.0070322, 0.0172801, 0.0059938, 0.0170978, -0.0064493, 0.0074657
4: -0.0179316, -0.0077001, -0.0179159, -0.0068372, -0.0079595, 0.0075010
5: 0.0136025, 0.0260690, 0.0123706, 0.0260117, -0.0088356, 0.0097644
6: 0.0054656, 0.0150563, 0.0045747, 0.0149724, -0.0068101, 0.0074299
7: -0.0225384, -0.0124276, -0.0225011, -0.0113219, -0.0084439, 0.0077667
8: 0.0087943, 0.0188888, 0.0077373, 0.0187207, -0.0063143, 0.0075356
9: 0.9006733, 0.9464105, 0.9011637, 0.9504839, -0.0311674, 0.0272491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145373, upper bound: 0.0148672
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0145373, upper bound: 0.0148672
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0040285, -0.0011629, -0.0039787, -0.0012362, -0.0027923, 0.0028158
1: 0.0185034, 0.0318816, 0.0175199, 0.0319700, -0.0083760, 0.0088071
2: 0.0207730, 0.0299970, 0.0200183, 0.0299195, -0.0057110, 0.0065878
3: 0.0069182, 0.0169626, 0.0059505, 0.0171014, -0.0066858, 0.0079788
4: -0.0177833, -0.0078421, -0.0179301, -0.0068323, -0.0086659, 0.0072980
5: 0.0133256, 0.0258420, 0.0122844, 0.0260243, -0.0093644, 0.0105149
6: 0.0054436, 0.0148479, 0.0045414, 0.0149827, -0.0068305, 0.0081109
7: -0.0224170, -0.0120990, -0.0225159, -0.0112368, -0.0092340, 0.0082996
8: 0.0084433, 0.0185713, 0.0076505, 0.0187207, -0.0070238, 0.0078039
9: 0.9017717, 0.9460111, 0.9011419, 0.9504939, -0.0327574, 0.0271433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149293, upper bound: 0.0147974
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0037909, -0.0014370, -0.0038583, -0.0007486, -0.0030423, 0.0024213
1: 0.0189839, 0.0319714, 0.0181777, 0.0320908, -0.0073275, 0.0083552
2: 0.0206837, 0.0295638, 0.0205071, 0.0300394, -0.0057271, 0.0056269
3: 0.0070322, 0.0172801, 0.0067606, 0.0175948, -0.0072721, 0.0071496
4: -0.0179316, -0.0077001, -0.0186997, -0.0076533, -0.0075668, 0.0086830
5: 0.0136025, 0.0260690, 0.0133294, 0.0267784, -0.0099862, 0.0092463
6: 0.0054656, 0.0150563, 0.0053360, 0.0156460, -0.0078295, 0.0070698
7: -0.0225384, -0.0124276, -0.0231128, -0.0121885, -0.0079539, 0.0087998
8: 0.0087943, 0.0188888, 0.0084207, 0.0191380, -0.0069430, 0.0072515
9: 0.9006733, 0.9464105, 0.8990980, 0.9473268, -0.0299489, 0.0305976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0147827
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0147827
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040285, -0.0011629, -0.0039011, -0.0007438, -0.0032847, 0.0027382
1: 0.0185034, 0.0318816, 0.0180926, 0.0320908, -0.0085360, 0.0084475
2: 0.0207730, 0.0299970, 0.0205118, 0.0301084, -0.0058498, 0.0063515
3: 0.0069182, 0.0169626, 0.0067115, 0.0175987, -0.0074926, 0.0076531
4: -0.0177833, -0.0078421, -0.0187146, -0.0076473, -0.0082746, 0.0084607
5: 0.0133256, 0.0258420, 0.0132517, 0.0267909, -0.0105010, 0.0099856
6: 0.0054436, 0.0148479, 0.0053026, 0.0156561, -0.0078373, 0.0077466
7: -0.0224170, -0.0120990, -0.0231279, -0.0121167, -0.0087389, 0.0093238
8: 0.0084433, 0.0185713, 0.0083405, 0.0191380, -0.0076130, 0.0074963
9: 0.9017717, 0.9460111, 0.8990747, 0.9473738, -0.0315146, 0.0303625

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
time: 0.75 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0040255, -0.0011961, -0.0036118, -0.0013768, -0.0026274, 0.0024157
1: 0.0179871, 0.0317185, 0.0185292, 0.0320996, -0.0086156, 0.0071891
2: 0.0204256, 0.0298431, 0.0200869, 0.0293880, -0.0052193, 0.0062670
3: 0.0063315, 0.0168435, 0.0068174, 0.0175390, -0.0076274, 0.0064997
4: -0.0176663, -0.0073214, -0.0181889, -0.0072288, -0.0077646, 0.0080291
5: 0.0127413, 0.0257148, 0.0134237, 0.0263647, -0.0101272, 0.0086457
6: 0.0049375, 0.0147419, 0.0051965, 0.0153070, -0.0076080, 0.0068833
7: -0.0222886, -0.0116189, -0.0227206, -0.0122730, -0.0075355, 0.0087997
8: 0.0079106, 0.0184785, 0.0088088, 0.0191372, -0.0077900, 0.0060954
9: 0.9022910, 0.9483978, 0.8996102, 0.9485875, -0.0288554, 0.0306945

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156478, upper bound: 0.0160217
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0156478, upper bound: 0.0160741
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0041301, -0.0010666, -0.0038928, -0.0012624, -0.0028676, 0.0028261
1: 0.0176665, 0.0317185, 0.0178827, 0.0319708, -0.0091061, 0.0083605
2: 0.0203836, 0.0300238, 0.0201110, 0.0298284, -0.0058533, 0.0067040
3: 0.0060571, 0.0168502, 0.0064673, 0.0170990, -0.0081546, 0.0066686
4: -0.0176928, -0.0071885, -0.0179207, -0.0071651, -0.0074999, 0.0085464
5: 0.0124001, 0.0257386, 0.0128393, 0.0260159, -0.0108647, 0.0090947
6: 0.0047325, 0.0147601, 0.0049502, 0.0149759, -0.0081751, 0.0068290
7: -0.0223166, -0.0113103, -0.0225062, -0.0116802, -0.0080462, 0.0095443
8: 0.0075582, 0.0184785, 0.0082192, 0.0187207, -0.0082671, 0.0067865
9: 0.9022505, 0.9490474, 0.9011564, 0.9489450, -0.0286366, 0.0321126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159204, upper bound: 0.0169136
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0159204, upper bound: 0.0169334
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0037290, -0.0013722, -0.0039041, -0.0007431, -0.0029860, 0.0025319
1: 0.0189294, 0.0318354, 0.0179587, 0.0320916, -0.0071611, 0.0085106
2: 0.0205538, 0.0294019, 0.0204217, 0.0301108, -0.0058651, 0.0055019
3: 0.0071253, 0.0172714, 0.0065961, 0.0175982, -0.0071622, 0.0073555
4: -0.0179212, -0.0077120, -0.0187128, -0.0075467, -0.0077175, 0.0086833
5: 0.0137357, 0.0260516, 0.0131323, 0.0267893, -0.0098212, 0.0094969
6: 0.0055354, 0.0150549, 0.0052056, 0.0156549, -0.0077586, 0.0072531
7: -0.0224988, -0.0125142, -0.0231261, -0.0120078, -0.0081730, 0.0086704
8: 0.0089574, 0.0188835, 0.0082355, 0.0191380, -0.0067498, 0.0074897
9: 0.9008157, 0.9464533, 0.8990778, 0.9478626, -0.0304578, 0.0305845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149809, upper bound: 0.0166620
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0149809, upper bound: 0.0166659
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0040304, -0.0011750, -0.0039169, -0.0007410, -0.0032894, 0.0027419
1: 0.0182315, 0.0317185, 0.0179498, 0.0320916, -0.0086687, 0.0084706
2: 0.0205658, 0.0298810, 0.0204330, 0.0301324, -0.0059865, 0.0063174
3: 0.0067359, 0.0168476, 0.0066104, 0.0175998, -0.0076092, 0.0076507
4: -0.0176822, -0.0076524, -0.0187189, -0.0075655, -0.0082428, 0.0086065
5: 0.0131355, 0.0257292, 0.0131344, 0.0267945, -0.0106263, 0.0099849
6: 0.0052822, 0.0147528, 0.0052179, 0.0156590, -0.0079526, 0.0077274
7: -0.0223055, -0.0119086, -0.0231323, -0.0120073, -0.0087331, 0.0094489
8: 0.0083040, 0.0184785, 0.0082390, 0.0191380, -0.0076996, 0.0075356
9: 0.9022665, 0.9469124, 0.8990681, 0.9477773, -0.0314564, 0.0309728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157619, upper bound: 0.0169136
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0157619, upper bound: 0.0169334
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0036046, -0.0014132, -0.0040950, -0.0011180, -0.0024866, 0.0026544
1: 0.0186341, 0.0320361, 0.0180102, 0.0318097, -0.0071657, 0.0087856
2: 0.0201282, 0.0293411, 0.0205854, 0.0300432, -0.0064826, 0.0050724
3: 0.0068590, 0.0174577, 0.0062686, 0.0168785, -0.0065390, 0.0077316
4: -0.0181188, -0.0072491, -0.0177240, -0.0073628, -0.0079439, 0.0078127
5: 0.0134670, 0.0262796, 0.0126354, 0.0257630, -0.0086809, 0.0102471
6: 0.0052314, 0.0152272, 0.0048994, 0.0147838, -0.0069301, 0.0076356
7: -0.0226657, -0.0123038, -0.0223695, -0.0115517, -0.0088830, 0.0075616
8: 0.0088535, 0.0190545, 0.0077603, 0.0184851, -0.0060931, 0.0079956
9: 0.8999601, 0.9484153, 0.9020864, 0.9481564, -0.0305130, 0.0290229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143119, upper bound: 0.0151044
time: 0.69 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0143119, upper bound: 0.0151044
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034915, -0.0008760, -0.0040116, -0.0012026, -0.0022889, 0.0030427
1: 0.0191843, 0.0321678, 0.0184520, 0.0318097, -0.0069272, 0.0085221
2: 0.0206492, 0.0294688, 0.0207677, 0.0299127, -0.0060072, 0.0052831
3: 0.0076742, 0.0180026, 0.0066718, 0.0168704, -0.0062220, 0.0081674
4: -0.0189000, -0.0081135, -0.0176926, -0.0077117, -0.0088220, 0.0073600
5: 0.0145090, 0.0270554, 0.0131813, 0.0257349, -0.0080874, 0.0108698
6: 0.0060399, 0.0159152, 0.0052757, 0.0147623, -0.0065174, 0.0083041
7: -0.0232546, -0.0132445, -0.0223366, -0.0120391, -0.0094762, 0.0069633
8: 0.0096094, 0.0195262, 0.0082108, 0.0184851, -0.0058477, 0.0081814
9: 0.8978013, 0.9451128, 0.9021362, 0.9467840, -0.0326077, 0.0277995

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0147146
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0147146
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040582, -0.0011684, -0.0039196, -0.0012547, -0.0028035, 0.0027511
1: 0.0174437, 0.0319168, 0.0186156, 0.0318097, -0.0090375, 0.0077328
2: 0.0200498, 0.0300308, 0.0207098, 0.0297621, -0.0061860, 0.0058559
3: 0.0060326, 0.0170190, 0.0068008, 0.0168640, -0.0072413, 0.0070578
4: -0.0179337, -0.0068965, -0.0176678, -0.0076784, -0.0078038, 0.0078356
5: 0.0122186, 0.0259860, 0.0133408, 0.0257128, -0.0098803, 0.0094854
6: 0.0045791, 0.0149628, 0.0053318, 0.0147454, -0.0072911, 0.0072782
7: -0.0225280, -0.0111824, -0.0223108, -0.0121375, -0.0083676, 0.0087302
8: 0.0076451, 0.0186232, 0.0084705, 0.0184851, -0.0075215, 0.0069571
9: 0.9014003, 0.9500028, 0.9021754, 0.9467339, -0.0286682, 0.0301087

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154124, upper bound: 0.0144327
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0154124, upper bound: 0.0146841
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039669, -0.0012338, -0.0038331, -0.0007389, -0.0032086, 0.0025993
1: 0.0178798, 0.0319168, 0.0192018, 0.0319310, -0.0087879, 0.0072875
2: 0.0202268, 0.0299120, 0.0212273, 0.0299071, -0.0061492, 0.0054807
3: 0.0064562, 0.0170107, 0.0075075, 0.0174673, -0.0079006, 0.0068033
4: -0.0179016, -0.0072414, -0.0184836, -0.0085166, -0.0075280, 0.0089934
5: 0.0127822, 0.0259595, 0.0143122, 0.0265500, -0.0108036, 0.0090762
6: 0.0049633, 0.0149409, 0.0060864, 0.0154719, -0.0082209, 0.0070102
7: -0.0224953, -0.0116686, -0.0229138, -0.0130641, -0.0079600, 0.0095627
8: 0.0081254, 0.0186232, 0.0091050, 0.0190066, -0.0078396, 0.0066740
9: 0.9014523, 0.9486332, 0.8999424, 0.9436350, -0.0277587, 0.0331222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0145600
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148894, upper bound: 0.0147207
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0036094, -0.0013833, -0.0040929, -0.0011381, -0.0024713, 0.0026918
1: 0.0185364, 0.0320369, 0.0178103, 0.0316530, -0.0071232, 0.0089073
2: 0.0200750, 0.0293535, 0.0204141, 0.0299231, -0.0064180, 0.0052299
3: 0.0068343, 0.0174586, 0.0061951, 0.0167591, -0.0064538, 0.0078949
4: -0.0181263, -0.0072294, -0.0176166, -0.0072584, -0.0081259, 0.0077308
5: 0.0134352, 0.0262829, 0.0125631, 0.0256449, -0.0085993, 0.0104088
6: 0.0052095, 0.0152351, 0.0048292, 0.0146824, -0.0068488, 0.0077902
7: -0.0226698, -0.0122760, -0.0222527, -0.0114462, -0.0090472, 0.0074986
8: 0.0088255, 0.0190545, 0.0077319, 0.0183907, -0.0060458, 0.0081073
9: 0.8999548, 0.9485529, 0.9025822, 0.9486928, -0.0313064, 0.0286837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0156764
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0153857
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0034961, -0.0008486, -0.0040072, -0.0012135, -0.0022825, 0.0030759
1: 0.0190835, 0.0321686, 0.0182436, 0.0316530, -0.0068853, 0.0086499
2: 0.0205946, 0.0294788, 0.0205896, 0.0297941, -0.0059357, 0.0054227
3: 0.0076492, 0.0180033, 0.0065901, 0.0167508, -0.0061339, 0.0083255
4: -0.0189050, -0.0080939, -0.0175842, -0.0076046, -0.0090020, 0.0072775
5: 0.0144790, 0.0270576, 0.0130949, 0.0256154, -0.0080025, 0.0110297
6: 0.0060190, 0.0159210, 0.0051990, 0.0146603, -0.0064321, 0.0084556
7: -0.0232573, -0.0132112, -0.0222186, -0.0119252, -0.0096358, 0.0068985
8: 0.0095821, 0.0195262, 0.0081740, 0.0183907, -0.0057967, 0.0082940
9: 0.8977976, 0.9452673, 0.9026326, 0.9473403, -0.0333616, 0.0274656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0155044
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0155044
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0040643, -0.0011620, -0.0039145, -0.0012636, -0.0028007, 0.0027525
1: 0.0173192, 0.0319175, 0.0184181, 0.0316530, -0.0090391, 0.0078357
2: 0.0199713, 0.0300403, 0.0205435, 0.0296403, -0.0061534, 0.0059917
3: 0.0059393, 0.0170196, 0.0067288, 0.0167444, -0.0072159, 0.0072121
4: -0.0179360, -0.0068119, -0.0175591, -0.0075858, -0.0079626, 0.0078198
5: 0.0121168, 0.0259878, 0.0132685, 0.0255931, -0.0098658, 0.0096279
6: 0.0044993, 0.0149644, 0.0052648, 0.0146432, -0.0072776, 0.0074206
7: -0.0225303, -0.0110853, -0.0221921, -0.0120437, -0.0084972, 0.0087208
8: 0.0075605, 0.0186232, 0.0084396, 0.0183907, -0.0074978, 0.0070593
9: 0.9013969, 0.9504046, 0.9026718, 0.9472603, -0.0294102, 0.0300219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 96
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0161356, upper bound: 0.0152805
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0171563, upper bound: 0.0159041
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039739, -0.0012286, -0.0038231, -0.0007478, -0.0031717, 0.0025945
1: 0.0177505, 0.0319175, 0.0189903, 0.0317775, -0.0088061, 0.0073860
2: 0.0201459, 0.0299214, 0.0210448, 0.0297834, -0.0061401, 0.0056243
3: 0.0063604, 0.0170113, 0.0074231, 0.0173521, -0.0078857, 0.0069590
4: -0.0179042, -0.0071523, -0.0183832, -0.0083964, -0.0076957, 0.0089661
5: 0.0126730, 0.0259616, 0.0142238, 0.0264376, -0.0107834, 0.0092198
6: 0.0048786, 0.0149427, 0.0060052, 0.0153723, -0.0081998, 0.0071560
7: -0.0224979, -0.0115663, -0.0228039, -0.0129404, -0.0080910, 0.0095354
8: 0.0080348, 0.0186232, 0.0090632, 0.0189046, -0.0078355, 0.0067733
9: 0.9014485, 0.9490513, 0.9004252, 0.9442251, -0.0285251, 0.0330234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 120
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 120

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0152805
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159041
time: 0.72 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0039358, -0.0012584, -0.0037531, -0.0014536, -0.0024822, 0.0024947
1: 0.0176204, 0.0319700, 0.0191233, 0.0319055, -0.0086461, 0.0070031
2: 0.0200117, 0.0298508, 0.0207164, 0.0294683, -0.0057491, 0.0054924
3: 0.0059934, 0.0170978, 0.0071643, 0.0171908, -0.0074689, 0.0063583
4: -0.0179159, -0.0068370, -0.0178486, -0.0077817, -0.0074101, 0.0079290
5: 0.0123700, 0.0260117, 0.0137739, 0.0259752, -0.0097372, 0.0086744
6: 0.0045743, 0.0149724, 0.0055729, 0.0149748, -0.0074087, 0.0067121
7: -0.0225011, -0.0113214, -0.0224668, -0.0125670, -0.0075838, 0.0084071
8: 0.0077367, 0.0187207, 0.0089621, 0.0188062, -0.0075397, 0.0061788
9: 0.9011636, 0.9504850, 0.9010233, 0.9460711, -0.0270795, 0.0311555

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148680, upper bound: 0.0145409
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143225
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0039789, -0.0012361, -0.0039902, -0.0012034, -0.0027755, 0.0027541
1: 0.0175189, 0.0319700, 0.0186371, 0.0318097, -0.0087866, 0.0082256
2: 0.0200180, 0.0299197, 0.0208021, 0.0299066, -0.0064541, 0.0056980
3: 0.0059493, 0.0171014, 0.0070426, 0.0168759, -0.0079817, 0.0065872
4: -0.0179301, -0.0068315, -0.0177137, -0.0079080, -0.0072402, 0.0086287
5: 0.0122830, 0.0260243, 0.0134865, 0.0257539, -0.0104809, 0.0092042
6: 0.0045405, 0.0149827, 0.0055432, 0.0147769, -0.0080804, 0.0067388
7: -0.0225160, -0.0112357, -0.0223589, -0.0122311, -0.0081433, 0.0092006
8: 0.0076492, 0.0187207, 0.0086056, 0.0184851, -0.0078223, 0.0068811
9: 0.9011419, 0.9504979, 0.9021025, 0.9456860, -0.0269361, 0.0327664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: A, layer: 3, pos: 241
type: B, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0148081, upper bound: 0.0149400
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0038585, -0.0007485, -0.0037531, -0.0014536, -0.0024049, 0.0030046
1: 0.0181772, 0.0320908, 0.0191233, 0.0319055, -0.0082797, 0.0071591
2: 0.0205070, 0.0300396, 0.0207164, 0.0294683, -0.0054937, 0.0056654
3: 0.0067601, 0.0175948, 0.0071643, 0.0171908, -0.0071325, 0.0070718
4: -0.0186998, -0.0076531, -0.0178486, -0.0077817, -0.0084876, 0.0075404
5: 0.0133288, 0.0267784, 0.0137739, 0.0259752, -0.0092216, 0.0096911
6: 0.0053356, 0.0156460, 0.0055729, 0.0149748, -0.0070486, 0.0076243
7: -0.0231128, -0.0121880, -0.0224668, -0.0125670, -0.0085377, 0.0079208
8: 0.0084201, 0.0191380, 0.0089621, 0.0188062, -0.0072200, 0.0067123
9: 0.8990978, 0.9473280, 0.9010233, 0.9460711, -0.0299322, 0.0298484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143225
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143276
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039013, -0.0007438, -0.0039902, -0.0012034, -0.0026979, 0.0032464
1: 0.0180917, 0.0320908, 0.0186371, 0.0318097, -0.0083779, 0.0083713
2: 0.0205115, 0.0301087, 0.0208021, 0.0299066, -0.0061998, 0.0058035
3: 0.0067103, 0.0175987, 0.0070426, 0.0168759, -0.0076412, 0.0072736
4: -0.0187146, -0.0076465, -0.0177137, -0.0079080, -0.0082696, 0.0082521
5: 0.0132504, 0.0267909, 0.0134865, 0.0257539, -0.0099639, 0.0101950
6: 0.0053016, 0.0156561, 0.0055432, 0.0147769, -0.0077272, 0.0076201
7: -0.0231279, -0.0121156, -0.0223589, -0.0122311, -0.0090500, 0.0087097
8: 0.0083392, 0.0191380, 0.0086056, 0.0184851, -0.0074784, 0.0073890
9: 0.8990747, 0.9473776, 0.9021025, 0.9456860, -0.0297427, 0.0314445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
time: 0.70 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0036118, -0.0013768, -0.0039890, -0.0012300, -0.0023818, 0.0025709
1: 0.0185292, 0.0320996, 0.0181271, 0.0316530, -0.0071336, 0.0084665
2: 0.0200869, 0.0293880, 0.0204558, 0.0297538, -0.0061346, 0.0051909
3: 0.0068174, 0.0175390, 0.0064658, 0.0167523, -0.0064781, 0.0075107
4: -0.0181889, -0.0072288, -0.0175900, -0.0073922, -0.0079390, 0.0077381
5: 0.0134237, 0.0263647, 0.0129010, 0.0256208, -0.0086237, 0.0099398
6: 0.0051965, 0.0153070, 0.0050345, 0.0146642, -0.0068672, 0.0074980
7: -0.0227206, -0.0122730, -0.0222248, -0.0117515, -0.0086264, 0.0075145
8: 0.0088088, 0.0191372, 0.0080838, 0.0183907, -0.0060711, 0.0076377
9: 0.8996102, 0.9485875, 0.9026234, 0.9480568, -0.0303952, 0.0287529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: A, layer: 3, pos: 180
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160780, upper bound: 0.0157059
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0160780, upper bound: 0.0154503
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0038928, -0.0012624, -0.0040929, -0.0011381, -0.0027547, 0.0028304
1: 0.0178827, 0.0319708, 0.0178103, 0.0316530, -0.0083308, 0.0089428
2: 0.0201110, 0.0298284, 0.0204141, 0.0299231, -0.0065596, 0.0058340
3: 0.0064673, 0.0170990, 0.0061951, 0.0167591, -0.0066773, 0.0080625
4: -0.0179207, -0.0071651, -0.0176166, -0.0072584, -0.0084632, 0.0074833
5: 0.0128393, 0.0260159, 0.0125631, 0.0256449, -0.0090833, 0.0107067
6: 0.0049502, 0.0149759, 0.0048292, 0.0146824, -0.0068231, 0.0080864
7: -0.0225062, -0.0116802, -0.0222527, -0.0114462, -0.0093556, 0.0080214
8: 0.0082192, 0.0187207, 0.0077319, 0.0183907, -0.0067913, 0.0081244
9: 0.9011564, 0.9489450, 0.9025822, 0.9486928, -0.0319414, 0.0286043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 120
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0159484
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0157738
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0039042, -0.0007431, -0.0036914, -0.0013881, -0.0025161, 0.0029484
1: 0.0179584, 0.0320916, 0.0190581, 0.0317723, -0.0084327, 0.0070035
2: 0.0204216, 0.0301110, 0.0205837, 0.0293029, -0.0053838, 0.0058009
3: 0.0065958, 0.0175982, 0.0072540, 0.0171814, -0.0073385, 0.0069595
4: -0.0187128, -0.0075465, -0.0178370, -0.0077924, -0.0084850, 0.0076904
5: 0.0131320, 0.0267893, 0.0139053, 0.0259562, -0.0094711, 0.0095186
6: 0.0052054, 0.0156549, 0.0056409, 0.0149725, -0.0072304, 0.0075520
7: -0.0231261, -0.0120075, -0.0224255, -0.0126518, -0.0083990, 0.0081409
8: 0.0082352, 0.0191380, 0.0091194, 0.0188012, -0.0074598, 0.0065225
9: 0.8990777, 0.9478631, 0.9011719, 0.9461287, -0.0299089, 0.0303690

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0149809
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0039171, -0.0007410, -0.0039920, -0.0012132, -0.0027039, 0.0032510
1: 0.0179485, 0.0320916, 0.0183576, 0.0316530, -0.0083945, 0.0084913
2: 0.0204326, 0.0301328, 0.0205961, 0.0297911, -0.0061661, 0.0059317
3: 0.0066088, 0.0175998, 0.0068610, 0.0167564, -0.0076376, 0.0073814
4: -0.0187190, -0.0075645, -0.0176061, -0.0077139, -0.0084065, 0.0082213
5: 0.0131328, 0.0267946, 0.0132893, 0.0256355, -0.0099625, 0.0103122
6: 0.0052167, 0.0156591, 0.0053748, 0.0146753, -0.0077081, 0.0077276
7: -0.0231324, -0.0120058, -0.0222418, -0.0120382, -0.0091728, 0.0087047
8: 0.0082373, 0.0191380, 0.0084693, 0.0183907, -0.0075168, 0.0074770
9: 0.8990680, 0.9477819, 0.9025983, 0.9465885, -0.0302946, 0.0313811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 96
type: B, layer: 3, pos: 96
type: B, layer: 3, pos: 180
type: A, layer: 3, pos: 180
type: A, layer: 3, pos: 120
type: A, layer: 3, pos: 43
type: B, layer: 3, pos: 43
type: A, layer: 3, pos: 133
type: B, layer: 3, pos: 133
type: B, layer: 3, pos: 241
type: A, layer: 3, pos: 241

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 3, pos: 96

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0157738
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0158301
time: 0.80 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.10 seconds
IS_A1_B1_A1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0145575, upper bound: 0.0148610
IS_A1_B1_A1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0145575, upper bound: 0.0148610
IS_A1_B1_A1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0141949
IS_A1_B1_A1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0141975
IS_A1_B1_A1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0149742, upper bound: 0.0137180
IS_A1_B1_A1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0149742, upper bound: 0.0139274
IS_A1_B1_A1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0144884, upper bound: 0.0139860
IS_A1_B1_A1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0148435, upper bound: 0.0140992
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152647
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154616, upper bound: 0.0152647
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156892, upper bound: 0.0152410
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154616, upper bound: 0.0152410
IS_A1_B1_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152410, upper bound: 0.0155486
IS_A1_B1_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152410, upper bound: 0.0159943
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154616, upper bound: 0.0152410
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160988, upper bound: 0.0160988
IS_A1_B1_A1_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0138747, upper bound: 0.0148278
IS_A1_B1_A1_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
IS_A1_B1_A1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0145360, upper bound: 0.0148031
IS_A1_B1_A1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
IS_A1_B1_A1_B2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
IS_A1_B1_A1_B2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0134268, upper bound: 0.0147434
IS_A1_B1_A1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
IS_A1_B1_A1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0140006, upper bound: 0.0147370
IS_A1_B1_A1_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152527, upper bound: 0.0156279
IS_A1_B1_A1_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0151434, upper bound: 0.0156279
IS_A1_B1_A1_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0161630, upper bound: 0.0160080
IS_A1_B1_A1_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159786, upper bound: 0.0160080
IS_A1_B1_A1_B2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0151434, upper bound: 0.0156279
IS_A1_B1_A1_B2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152076, upper bound: 0.0156279
IS_A1_B1_A1_B2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159786, upper bound: 0.0160080
IS_A1_B1_A1_B2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160300, upper bound: 0.0160080
IS_A1_B1_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0148278, upper bound: 0.0138747
IS_A1_B1_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
IS_A1_B1_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0148031, upper bound: 0.0145360
IS_A1_B1_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
IS_A1_B1_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
IS_A1_B1_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147434, upper bound: 0.0134268
IS_A1_B1_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
IS_A1_B1_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147370, upper bound: 0.0140006
IS_A1_B1_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152527
IS_A1_B1_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0151434
IS_A1_B1_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0161630
IS_A1_B1_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0159786
IS_A1_B1_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0151434
IS_A1_B1_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156279, upper bound: 0.0152076
IS_A1_B1_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0159786
IS_A1_B1_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160080, upper bound: 0.0160300
IS_A1_B1_A2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0141666, upper bound: 0.0146558
IS_A1_B1_A2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0141666, upper bound: 0.0146558
IS_A1_B1_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0140765, upper bound: 0.0151040
IS_A1_B1_A2_B2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
IS_A1_B1_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0140835, upper bound: 0.0144218
IS_A1_B1_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0140835, upper bound: 0.0144218
IS_A1_B1_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
IS_A1_B1_A2_B2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0139894, upper bound: 0.0147366
IS_A1_B1_A2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152357, upper bound: 0.0156219
IS_A1_B1_A2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152357, upper bound: 0.0156406
IS_A1_B1_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160932, upper bound: 0.0160120
IS_A1_B1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159532, upper bound: 0.0160120
IS_A1_B1_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0151371, upper bound: 0.0156219
IS_A1_B1_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0151371, upper bound: 0.0156406
IS_A1_B1_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159532, upper bound: 0.0159694
IS_A1_B1_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159532, upper bound: 0.0160120
IS_A1_B2_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0151044, upper bound: 0.0143119
IS_A1_B2_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0151044, upper bound: 0.0143119
IS_A1_B2_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147146, upper bound: 0.0141498
IS_A1_B2_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147146, upper bound: 0.0141498
IS_A1_B2_A1_B1_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0144327, upper bound: 0.0154124
IS_A1_B2_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0144327, upper bound: 0.0149794
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0145600, upper bound: 0.0141498
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147207, upper bound: 0.0148894
IS_A1_B2_A1_B1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156764, upper bound: 0.0160950
IS_A1_B2_A1_B1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0153857, upper bound: 0.0160950
IS_A1_B2_A1_B1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0155044, upper bound: 0.0160950
IS_A1_B2_A1_B1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0155044, upper bound: 0.0160950
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152805, upper bound: 0.0161356
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0171563
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0152805, upper bound: 0.0160950
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159041, upper bound: 0.0170393
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0145409, upper bound: 0.0148680
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0143225, upper bound: 0.0147842
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0149400, upper bound: 0.0148081
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0143225, upper bound: 0.0147842
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0143276, upper bound: 0.0147842
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146391, upper bound: 0.0147399
IS_A1_B2_A1_B2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0157059, upper bound: 0.0160780
IS_A1_B2_A1_B2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154503, upper bound: 0.0160780
IS_A1_B2_A1_B2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159484, upper bound: 0.0169337
IS_A1_B2_A1_B2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0157738, upper bound: 0.0169337
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0149809, upper bound: 0.0166586
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0150554, upper bound: 0.0166586
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0157738, upper bound: 0.0169337
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0158301, upper bound: 0.0169337
IS_A1_B2_A2_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147293, upper bound: 0.0146145
IS_A1_B2_A2_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
IS_A1_B2_A2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146669, upper bound: 0.0154260
IS_A1_B2_A2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
IS_A1_B2_A2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
IS_A1_B2_A2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146479, upper bound: 0.0140702
IS_A1_B2_A2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
IS_A1_B2_A2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147606
IS_A1_B2_A2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0161309
IS_A1_B2_A2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160217
IS_A1_B2_A2_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0171183
IS_A1_B2_A2_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169345
IS_A1_B2_A2_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160217
IS_A1_B2_A2_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154774, upper bound: 0.0160741
IS_A1_B2_A2_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169345
IS_A1_B2_A2_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0158176, upper bound: 0.0169503
IS_A1_B2_A2_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0145373, upper bound: 0.0148672
IS_A1_B2_A2_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0145373, upper bound: 0.0148672
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0149293, upper bound: 0.0147974
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
IS_A1_B2_A2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0147827
IS_A1_B2_A2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0143197, upper bound: 0.0147827
IS_A1_B2_A2_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
IS_A1_B2_A2_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0146180, upper bound: 0.0147325
IS_A1_B2_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156478, upper bound: 0.0160217
IS_A1_B2_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0156478, upper bound: 0.0160741
IS_A1_B2_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159204, upper bound: 0.0169136
IS_A1_B2_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0159204, upper bound: 0.0169334
IS_A1_B2_A2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0149809, upper bound: 0.0166620
IS_A1_B2_A2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0149809, upper bound: 0.0166659
IS_A1_B2_A2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0157619, upper bound: 0.0169136
IS_A1_B2_A2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0157619, upper bound: 0.0169334
IS_A2_B1_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0143119, upper bound: 0.0151044
IS_A2_B1_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0143119, upper bound: 0.0151044
IS_A2_B1_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0147146
IS_A2_B1_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0147146
IS_A2_B1_B1_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154124, upper bound: 0.0144327
IS_A2_B1_B1_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0154124, upper bound: 0.0146841
IS_A2_B1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0141498, upper bound: 0.0145600
IS_A2_B1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0148894, upper bound: 0.0147207
IS_A2_B1_B1_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0156764
IS_A2_B1_B1_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0153857
IS_A2_B1_B1_A1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0155044
IS_A2_B1_B1_A1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0155044
IS_A2_B1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0161356, upper bound: 0.0152805
IS_A2_B1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0171563, upper bound: 0.0159041
IS_A2_B1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160950, upper bound: 0.0152805
IS_A2_B1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159041
IS_A2_B1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0148680, upper bound: 0.0145409
IS_A2_B1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143225
IS_A2_B1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0148081, upper bound: 0.0149400
IS_A2_B1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
IS_A2_B1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143225
IS_A2_B1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147842, upper bound: 0.0143276
IS_A2_B1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
IS_A2_B1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0147399, upper bound: 0.0146391
IS_A2_B1_B1_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160780, upper bound: 0.0157059
IS_A2_B1_B1_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0160780, upper bound: 0.0154503
IS_A2_B1_B1_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0159484
IS_A2_B1_B1_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0157738
IS_A2_B1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0149809
IS_A2_B1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0157738
IS_A2_B1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.10
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0158301
IS_A2_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0146145, upper bound: 0.0147293
IS_A2_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0154260, upper bound: 0.0146669
IS_A2_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0140702, upper bound: 0.0146479
IS_A2_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0147606, upper bound: 0.0146180
IS_A2_B1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0161309, upper bound: 0.0154774
IS_A2_B1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0171183, upper bound: 0.0158176
IS_A2_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0154774
IS_A2_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169503, upper bound: 0.0158176
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0148672, upper bound: 0.0145409
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0147974, upper bound: 0.0149400
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0147827, upper bound: 0.0143224
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0147325, upper bound: 0.0146182
IS_A2_B1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0157059
IS_A2_B1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0159484
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0158177
IS_A2_B2_B1_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0154310, upper bound: 0.0148146
IS_A2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0153595, upper bound: 0.0141754
IS_A2_B2_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0161620, upper bound: 0.0141554
IS_A2_B2_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0158468, upper bound: 0.0140758
IS_A2_B2_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0166864, upper bound: 0.0151155
IS_A2_B2_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0166864, upper bound: 0.0150795
IS_A2_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159891
IS_A2_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0170393, upper bound: 0.0159041
IS_A2_B2_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0157601, upper bound: 0.0138293
IS_A2_B2_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0157875, upper bound: 0.0144901
IS_A2_B2_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0156761, upper bound: 0.0134165
IS_A2_B2_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0157285, upper bound: 0.0139730
IS_A2_B2_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150996
IS_A2_B2_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0159484
IS_A2_B2_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B2_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169337, upper bound: 0.0158301
IS_A2_B2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0146145, upper bound: 0.0147293
IS_A2_B2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0154260, upper bound: 0.0146669
IS_A2_B2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0140702, upper bound: 0.0146479
IS_A2_B2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0147606, upper bound: 0.0146180
IS_A2_B2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0161309, upper bound: 0.0154774
IS_A2_B2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0171183, upper bound: 0.0158176
IS_A2_B2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0160741, upper bound: 0.0154774
IS_A2_B2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169503, upper bound: 0.0158176
IS_A2_B2_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0155563, upper bound: 0.0141608
IS_A2_B2_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0160926, upper bound: 0.0140402
IS_A2_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0152806, upper bound: 0.0140804
IS_A2_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0157253, upper bound: 0.0139595
IS_A2_B2_B2_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150996
IS_A2_B2_B2_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0159484
IS_A2_B2_B2_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0166586, upper bound: 0.0150554
IS_A2_B2_B2_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.10
Output dim: 9, lower bound: -0.0169334, upper bound: 0.0158177

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.89 + 597.51 = 600.40 seconds
