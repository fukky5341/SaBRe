## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00365364


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0023443, 0.0023443)
1: (-0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0059489, 0.0059489)
2: (0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0036907, 0.0036907)
3: (0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0068916, 0.0068916)
4: (-0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0060511, 0.0060511)
5: (0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0022920, 0.0022920)
6: (0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0087463, 0.0087463)
7: (0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0061202, 0.0061202)
8: (-0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0065618, 0.0065618)
9: (-0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0043345, 0.0043345)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.57 + 2.40 = 3.97 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0039769, upper bound: 0.0039770

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038144, upper bound: 0.0038162
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038162, upper bound: 0.0038144
time: 1.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 2.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 2.58
Output dim: 7, lower bound: -0.0038144, upper bound: 0.0038162
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 2.58
Output dim: 7, lower bound: -0.0038162, upper bound: 0.0038144

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021912, 0.0022078
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0055604, 0.0056025
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034497, 0.0034758
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0064903, 0.0064415
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0056559, 0.0056987
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021423, 0.0021585
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0082370, 0.0081751
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0057638, 0.0057205
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0061797, 0.0061333
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040514, 0.0040821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0033196, upper bound: 0.0033243
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0033196, upper bound: 0.0033243
time: 1.07 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0022078, 0.0021912
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0056025, 0.0055604
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034758, 0.0034497
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0064415, 0.0064903
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0056987, 0.0056559
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021585, 0.0021423
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0081751, 0.0082370
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0057205, 0.0057638
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0061333, 0.0061797
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040821, 0.0040514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0038106, upper bound: 0.0037753
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037768, upper bound: 0.0038089
time: 1.30 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 4.17 seconds
RS_RSZ1_RSZ1, status: Status.VERIFIED, split count: 2, time: 4.17
Output dim: 7, lower bound: -0.0033196, upper bound: 0.0033243
RS_RSZ1_RSZ2, status: Status.VERIFIED, split count: 2, time: 4.17
Output dim: 7, lower bound: -0.0033196, upper bound: 0.0033243
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 4.17
Output dim: 7, lower bound: -0.0038106, upper bound: 0.0037753
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 4.17
Output dim: 7, lower bound: -0.0037768, upper bound: 0.0038089

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021950, 0.0021818
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0055701, 0.0055366
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034557, 0.0034349
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0064139, 0.0064527
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0056657, 0.0056316
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021460, 0.0021331
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0081400, 0.0081893
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0056960, 0.0057305
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0061070, 0.0061440
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040584, 0.0040340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037882, upper bound: 0.0035746
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036012, upper bound: 0.0037529
time: 1.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021996, 0.0021784
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0055817, 0.0055280
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034629, 0.0034296
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0064039, 0.0064661
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0056775, 0.0056229
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021505, 0.0021298
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0081274, 0.0082063
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0056872, 0.0057424
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0060975, 0.0061568
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040669, 0.0040278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037554, upper bound: 0.0036008
time: 1.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0035842, upper bound: 0.0037867
time: 1.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 4.10 seconds
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 7, lower bound: -0.0037882, upper bound: 0.0035746
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 7, lower bound: -0.0036012, upper bound: 0.0037529
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 7, lower bound: -0.0037554, upper bound: 0.0036008
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 4.10
Output dim: 7, lower bound: -0.0035842, upper bound: 0.0037867

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021677, 0.0021606
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0055008, 0.0054828
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034127, 0.0034016
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0063516, 0.0063725
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0055953, 0.0055770
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021193, 0.0021124
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0080610, 0.0080875
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0056407, 0.0056592
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0060477, 0.0060676
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040080, 0.0039949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037602, upper bound: 0.0035059
time: 0.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036736, upper bound: 0.0035471
time: 1.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021739, 0.0021545
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0055167, 0.0054673
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034226, 0.0033919
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0063336, 0.0063908
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0056114, 0.0055612
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021254, 0.0021064
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0080382, 0.0081108
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0056247, 0.0056755
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0060306, 0.0060851
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040195, 0.0039836

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0034992, upper bound: 0.0036739
time: 1.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0035257, upper bound: 0.0036558
time: 1.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021723, 0.0021573
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0055124, 0.0054745
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034199, 0.0033964
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0063419, 0.0063859
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0056071, 0.0055684
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021238, 0.0021092
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0080487, 0.0081045
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0056321, 0.0056712
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0060385, 0.0060804
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040164, 0.0039888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036023, upper bound: 0.0034859
time: 1.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036486, upper bound: 0.0034497
time: 0.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021785, 0.0021511
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0055282, 0.0054587
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0034297, 0.0033866
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0063237, 0.0064042
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0056231, 0.0055525
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0021299, 0.0021031
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0080256, 0.0081277
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0056159, 0.0056874
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0060212, 0.0060978
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0040279, 0.0039773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0031449, upper bound: 0.0032905
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0031449, upper bound: 0.0032905
time: 1.12 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 3.73 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0037602, upper bound: 0.0035059
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0036736, upper bound: 0.0035471
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0034992, upper bound: 0.0036739
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0035257, upper bound: 0.0036558
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0036023, upper bound: 0.0034859
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0036486, upper bound: 0.0034497
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0031449, upper bound: 0.0032905
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 3.73
Output dim: 7, lower bound: -0.0031449, upper bound: 0.0032905

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021313, 0.0021402
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0054085, 0.0054310
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033555, 0.0033694
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0062916, 0.0062655
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0055014, 0.0055242
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020838, 0.0020924
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0079848, 0.0079517
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0055874, 0.0055642
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0059906, 0.0059657
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0039407, 0.0039571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037273, upper bound: 0.0034721
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037123, upper bound: 0.0034722
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021475, 0.0021242
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0054495, 0.0053905
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033809, 0.0033443
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0062447, 0.0063130
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0055431, 0.0054831
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020996, 0.0020768
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0079253, 0.0080120
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0055457, 0.0056064
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0059459, 0.0060110
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0039706, 0.0039276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035996, upper bound: 0.0034536
time: 1.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035847, upper bound: 0.0034660
time: 1.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021226, 0.0020775
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0053864, 0.0052721
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033418, 0.0032708
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061074, 0.0062399
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0054789, 0.0053626
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020753, 0.0020312
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077511, 0.0079193
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054238, 0.0055415
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058152, 0.0059414
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0039246, 0.0038413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034982, upper bound: 0.0036089
time: 1.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0034740, upper bound: 0.0036728
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020970, 0.0021010
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0053214, 0.0053316
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033014, 0.0033077
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061764, 0.0061646
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0054128, 0.0054231
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020502, 0.0020541
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0078387, 0.0078237
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054851, 0.0054746
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058809, 0.0058697
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038772, 0.0038847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0033747, upper bound: 0.0035460
time: 1.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034117, upper bound: 0.0034977
time: 1.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 3.67 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0037273, upper bound: 0.0034721
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0037123, upper bound: 0.0034722
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0035996, upper bound: 0.0034536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0035847, upper bound: 0.0034660
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0034982, upper bound: 0.0036089
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0034740, upper bound: 0.0036728
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0033747, upper bound: 0.0035460
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 3.67
Output dim: 7, lower bound: -0.0034117, upper bound: 0.0034977

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021202, 0.0021316
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0053803, 0.0054092
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033380, 0.0033559
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0062663, 0.0062329
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0054727, 0.0055021
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020729, 0.0020840
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0079527, 0.0079103
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0055649, 0.0055352
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0059665, 0.0059347
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0039202, 0.0039412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037268, upper bound: 0.0034589
time: 1.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036273, upper bound: 0.0034701
time: 1.42 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021230, 0.0021291
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0053875, 0.0054028
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033424, 0.0033519
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0062589, 0.0062411
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0054800, 0.0054956
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020757, 0.0020816
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0079434, 0.0079208
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0055584, 0.0055426
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0059595, 0.0059425
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0039254, 0.0039366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037013, upper bound: 0.0034615
time: 1.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036994, upper bound: 0.0034609
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020951, 0.0020404
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0053167, 0.0051778
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032985, 0.0032123
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0059982, 0.0061591
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0054080, 0.0052667
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020484, 0.0019949
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0076125, 0.0078167
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0053268, 0.0054698
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0057112, 0.0058644
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038738, 0.0037726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034389, upper bound: 0.0036151
time: 1.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034376, upper bound: 0.0036400
time: 1.29 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 4.11 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 7, lower bound: -0.0037268, upper bound: 0.0034589
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 7, lower bound: -0.0036273, upper bound: 0.0034701
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 7, lower bound: -0.0037013, upper bound: 0.0034615
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 4.11
Output dim: 7, lower bound: -0.0036994, upper bound: 0.0034609
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 7, lower bound: -0.0034389, upper bound: 0.0036151
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 4.11
Output dim: 7, lower bound: -0.0034376, upper bound: 0.0036400

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020820, 0.0021035
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0052833, 0.0053380
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032778, 0.0033117
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061838, 0.0061204
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0053740, 0.0054296
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020355, 0.0020566
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0078480, 0.0077676
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054917, 0.0054354
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058879, 0.0058276
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038495, 0.0038893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037068, upper bound: 0.0034410
time: 1.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0037063, upper bound: 0.0034372
time: 1.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021184, 0.0021268
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0053757, 0.0053971
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033351, 0.0033484
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0062523, 0.0062275
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0054680, 0.0054898
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020711, 0.0020794
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0079350, 0.0079035
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0055525, 0.0055305
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0059532, 0.0059296
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0039168, 0.0039324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036786, upper bound: 0.0033732
time: 1.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035310, upper bound: 0.0034385
time: 1.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0021208, 0.0021243
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0053818, 0.0053907
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0033389, 0.0033444
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0062449, 0.0062345
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0054742, 0.0054833
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020735, 0.0020769
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0079256, 0.0079124
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0055459, 0.0055367
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0059461, 0.0059363
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0039212, 0.0039277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034096, upper bound: 0.0032107
time: 1.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0034096, upper bound: 0.0032107
time: 1.31 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 4.16 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.0037068, upper bound: 0.0034410
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.0037063, upper bound: 0.0034372
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.0036786, upper bound: 0.0033732
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.0035310, upper bound: 0.0034385
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.0034096, upper bound: 0.0032107
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 4.16
Output dim: 7, lower bound: -0.0034096, upper bound: 0.0032107

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020565, 0.0020797
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0052187, 0.0052774
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032377, 0.0032741
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061137, 0.0060457
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0053083, 0.0053680
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020106, 0.0020333
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077590, 0.0076727
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054294, 0.0053690
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058212, 0.0057564
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038024, 0.0038452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036958, upper bound: 0.0034301
time: 1.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036904, upper bound: 0.0034302
time: 1.51 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020581, 0.0020771
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0052228, 0.0052710
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032402, 0.0032702
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061062, 0.0060503
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0053124, 0.0053615
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020122, 0.0020308
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077496, 0.0076786
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054228, 0.0053731
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058141, 0.0057609
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038054, 0.0038405

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036954, upper bound: 0.0034261
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036903, upper bound: 0.0034263
time: 1.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0019886, 0.0020355
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0050464, 0.0051654
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0031308, 0.0032046
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0059838, 0.0058461
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0051331, 0.0052541
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0019443, 0.0019901
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0075943, 0.0074194
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0053141, 0.0051917
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0056976, 0.0055664
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0036769, 0.0037636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=249
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0033954, upper bound: 0.0031324
time: 1.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0033954, upper bound: 0.0031324
time: 1.43 seconds

## Summary of splitting (split count: 7)
- Time for RS candidates: 4.49 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.49
Output dim: 7, lower bound: -0.0036958, upper bound: 0.0034301
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.49
Output dim: 7, lower bound: -0.0036904, upper bound: 0.0034302
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 8, time: 4.49
Output dim: 7, lower bound: -0.0036954, upper bound: 0.0034261
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 8, time: 4.49
Output dim: 7, lower bound: -0.0036903, upper bound: 0.0034263
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 8, time: 4.49
Output dim: 7, lower bound: -0.0033954, upper bound: 0.0031324
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 8, time: 4.49
Output dim: 7, lower bound: -0.0033954, upper bound: 0.0031324

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020617, 0.0020868
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0052320, 0.0052956
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032459, 0.0032854
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061347, 0.0060610
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0053218, 0.0053865
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020157, 0.0020402
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077857, 0.0076922
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054480, 0.0053826
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058412, 0.0057710
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038121, 0.0038584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035871, upper bound: 0.0033524
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036193, upper bound: 0.0033298
time: 1.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020637, 0.0020838
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0052368, 0.0052880
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032490, 0.0032807
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061258, 0.0060666
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0053268, 0.0053787
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020176, 0.0020373
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077745, 0.0076993
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054402, 0.0053876
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058328, 0.0057764
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038156, 0.0038529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0036684, upper bound: 0.0033848
time: 1.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036193, upper bound: 0.0034059
time: 1.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020629, 0.0020843
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0052348, 0.0052891
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032477, 0.0032814
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061272, 0.0060643
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0053247, 0.0053799
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020168, 0.0020378
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077762, 0.0076964
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054414, 0.0053856
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058341, 0.0057742
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038142, 0.0038537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035402, upper bound: 0.0033071
time: 1.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035853, upper bound: 0.0032849
time: 1.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020653, 0.0020820
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0052409, 0.0052833
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032515, 0.0032778
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0061204, 0.0060713
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0053309, 0.0053740
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0020192, 0.0020355
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077676, 0.0077053
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0054354, 0.0053918
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0058276, 0.0057808
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0038186, 0.0038495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=248
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036081, upper bound: 0.0033325
time: 1.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0036008, upper bound: 0.0033483
time: 1.05 seconds

## Summary of splitting (split count: 8)
- Time for RS candidates: 3.63 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0035871, upper bound: 0.0033524
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0036193, upper bound: 0.0033298
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0036684, upper bound: 0.0033848
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0036193, upper bound: 0.0034059
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0035402, upper bound: 0.0033071
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0035853, upper bound: 0.0032849
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0036081, upper bound: 0.0033325
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 9, time: 3.63
Output dim: 7, lower bound: -0.0036008, upper bound: 0.0033483

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -0.0032831, -0.0003405, -0.0032831, -0.0003405, -0.0020336, 0.0020651
1: -0.0126420, -0.0051746, -0.0126420, -0.0051746, -0.0051606, 0.0052406
2: 0.0271869, 0.0318197, 0.0271869, 0.0318197, -0.0032017, 0.0032513
3: 0.0001472, 0.0087977, 0.0001472, 0.0087977, -0.0060709, 0.0059783
4: -0.0117521, -0.0041565, -0.0117521, -0.0041565, -0.0052492, 0.0053305
5: 0.0092868, 0.0121638, 0.0092868, 0.0121638, -0.0019883, 0.0020191
6: 0.0005749, 0.0115536, 0.0005749, 0.0115536, -0.0077048, 0.0075873
7: 0.9784616, 0.9861439, 0.9784616, 0.9861439, -0.0053915, 0.0053092
8: -0.0096568, -0.0014201, -0.0096568, -0.0014201, -0.0057805, 0.0056923
9: -0.0040615, 0.0013793, -0.0040615, 0.0013793, -0.0037601, 0.0038183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=247
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035626, upper bound: 0.0033093
time: 1.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.0035894, upper bound: 0.0032854
time: 1.16 seconds

## Summary of splitting (split count: 9)
- Time for RS candidates: 4.20 seconds
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 10, time: 4.20
Output dim: 7, lower bound: -0.0035626, upper bound: 0.0033093
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 10, time: 4.20
Output dim: 7, lower bound: -0.0035894, upper bound: 0.0032854

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 3.97 + 108.64 = 112.61 seconds
