## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0016185600000000002


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0021188, -0.0003786, -0.0021188, -0.0003786, -0.0013854, 0.0013854)
1: (-0.0096874, -0.0052713, -0.0096874, -0.0052713, -0.0035157, 0.0035157)
2: (0.0290199, 0.0317597, 0.0290199, 0.0317597, -0.0021812, 0.0021812)
3: (0.0002591, 0.0053750, 0.0002591, 0.0053750, -0.0040728, 0.0040728)
4: (-0.0087468, -0.0042548, -0.0087468, -0.0042548, -0.0035761, 0.0035761)
5: (0.0104251, 0.0121266, 0.0104251, 0.0121266, -0.0013545, 0.0013545)
6: (0.0007170, 0.0072097, 0.0007170, 0.0072097, -0.0051689, 0.0051689)
7: (0.9785610, 0.9831042, 0.9785610, 0.9831042, -0.0036170, 0.0036170)
8: (-0.0095502, -0.0046791, -0.0095502, -0.0046791, -0.0038780, 0.0038780)
9: (-0.0019088, 0.0013089, -0.0019088, 0.0013089, -0.0025616, 0.0025616)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.22 + 2.82 = 4.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0020232, upper bound: 0.0020232

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019767, upper bound: 0.0019441
time: 1.87 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019767, upper bound: 0.0019767
time: 1.71 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.70 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.70
Output dim: 7, lower bound: -0.0019767, upper bound: 0.0019441
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.70
Output dim: 7, lower bound: -0.0019767, upper bound: 0.0019767

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0020448, -0.0004118, -0.0020890, -0.0003841, -0.0013125, 0.0013257
1: -0.0094996, -0.0053556, -0.0096117, -0.0052852, -0.0033307, 0.0033640
2: 0.0291364, 0.0317074, 0.0290669, 0.0317510, -0.0020664, 0.0020871
3: 0.0003568, 0.0051574, 0.0002753, 0.0052873, -0.0038971, 0.0038585
4: -0.0085557, -0.0043406, -0.0086697, -0.0042690, -0.0033879, 0.0034218
5: 0.0104975, 0.0120941, 0.0104543, 0.0121212, -0.0012832, 0.0012961
6: 0.0008410, 0.0069335, 0.0007375, 0.0070984, -0.0049459, 0.0048969
7: 0.9786477, 0.9829110, 0.9785753, 0.9830264, -0.0034609, 0.0034266
8: -0.0094572, -0.0048863, -0.0095349, -0.0047626, -0.0037106, 0.0036739
9: -0.0017719, 0.0012474, -0.0018536, 0.0012987, -0.0024268, 0.0024511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0018055
time: 1.70 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0018698
time: 1.29 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0020817, -0.0003873, -0.0021044, -0.0003819, -0.0013124, 0.0013604
1: -0.0095932, -0.0052935, -0.0096508, -0.0052797, -0.0033304, 0.0034523
2: 0.0290784, 0.0317459, 0.0290426, 0.0317544, -0.0020662, 0.0021418
3: 0.0002848, 0.0052658, 0.0002689, 0.0053326, -0.0039994, 0.0038582
4: -0.0086509, -0.0042774, -0.0087095, -0.0042634, -0.0033876, 0.0035116
5: 0.0104615, 0.0121180, 0.0104393, 0.0121233, -0.0012831, 0.0013301
6: 0.0007496, 0.0070711, 0.0007294, 0.0071558, -0.0050757, 0.0048965
7: 0.9785838, 0.9830074, 0.9785697, 0.9830665, -0.0035517, 0.0034263
8: -0.0095258, -0.0047831, -0.0095409, -0.0047195, -0.0038080, 0.0036736
9: -0.0018401, 0.0012927, -0.0018821, 0.0013027, -0.0024266, 0.0025154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0018427
time: 1.81 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0019003
time: 1.67 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.72 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0018055
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0018698
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0018427
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 4.72
Output dim: 7, lower bound: -0.0019003, upper bound: 0.0019003

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0020057, -0.0004163, -0.0020890, -0.0003841, -0.0012747, 0.0013216
1: -0.0094002, -0.0053670, -0.0096117, -0.0052852, -0.0032348, 0.0033537
2: 0.0291981, 0.0317003, 0.0290669, 0.0317510, -0.0020069, 0.0020806
3: 0.0003700, 0.0050423, 0.0002753, 0.0052873, -0.0038851, 0.0037473
4: -0.0084546, -0.0043522, -0.0086697, -0.0042690, -0.0032903, 0.0034113
5: 0.0105358, 0.0120897, 0.0104543, 0.0121212, -0.0012463, 0.0012921
6: 0.0008577, 0.0067874, 0.0007375, 0.0070984, -0.0049307, 0.0047558
7: 0.9786595, 0.9828088, 0.9785753, 0.9830264, -0.0034502, 0.0033279
8: -0.0094447, -0.0049959, -0.0095349, -0.0047626, -0.0036992, 0.0035680
9: -0.0016995, 0.0012391, -0.0018536, 0.0012987, -0.0023569, 0.0024435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017550
time: 1.24 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017656
time: 1.32 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0019845, -0.0002665, -0.0020629, -0.0003890, -0.0012929, 0.0014495
1: -0.0093464, -0.0049870, -0.0095454, -0.0052978, -0.0032809, 0.0036782
2: 0.0292315, 0.0319361, 0.0291080, 0.0317433, -0.0020355, 0.0022820
3: -0.0000703, 0.0049800, 0.0002898, 0.0052105, -0.0042610, 0.0038007
4: -0.0083999, -0.0039656, -0.0086023, -0.0042817, -0.0033372, 0.0037413
5: 0.0105565, 0.0122361, 0.0104798, 0.0121164, -0.0012640, 0.0014171
6: 0.0002990, 0.0067084, 0.0007559, 0.0070009, -0.0054078, 0.0048236
7: 0.9782685, 0.9827535, 0.9785882, 0.9829582, -0.0037841, 0.0033753
8: -0.0098639, -0.0050552, -0.0095210, -0.0048357, -0.0040572, 0.0036189
9: -0.0016603, 0.0015160, -0.0018053, 0.0012896, -0.0023905, 0.0026800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0018080
time: 1.75 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0018282
time: 1.33 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0020436, -0.0003917, -0.0021044, -0.0003819, -0.0012748, 0.0013566
1: -0.0094966, -0.0053046, -0.0096508, -0.0052797, -0.0032349, 0.0034427
2: 0.0291383, 0.0317390, 0.0290426, 0.0317544, -0.0020070, 0.0021358
3: 0.0002977, 0.0051539, 0.0002689, 0.0053326, -0.0039882, 0.0037475
4: -0.0085527, -0.0042887, -0.0087095, -0.0042634, -0.0032905, 0.0035018
5: 0.0104987, 0.0121137, 0.0104393, 0.0121233, -0.0012463, 0.0013264
6: 0.0007660, 0.0069291, 0.0007294, 0.0071558, -0.0050615, 0.0047561
7: 0.9785953, 0.9829079, 0.9785697, 0.9830665, -0.0035418, 0.0033281
8: -0.0095135, -0.0048896, -0.0095409, -0.0047195, -0.0037973, 0.0035682
9: -0.0017697, 0.0012846, -0.0018821, 0.0013027, -0.0023570, 0.0025084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017845
time: 1.24 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017981
time: 1.70 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0020186, -0.0002424, -0.0020781, -0.0003869, -0.0012910, 0.0014868
1: -0.0094330, -0.0049258, -0.0095841, -0.0052923, -0.0032760, 0.0037731
2: 0.0291777, 0.0319741, 0.0290840, 0.0317466, -0.0020325, 0.0023408
3: -0.0001411, 0.0050803, 0.0002835, 0.0052553, -0.0043709, 0.0037951
4: -0.0084880, -0.0039034, -0.0086417, -0.0042762, -0.0033323, 0.0038378
5: 0.0105231, 0.0122597, 0.0104650, 0.0121185, -0.0012622, 0.0014537
6: 0.0002090, 0.0068357, 0.0007479, 0.0070578, -0.0055473, 0.0048165
7: 0.9782056, 0.9828426, 0.9785826, 0.9829980, -0.0038817, 0.0033703
8: -0.0099314, -0.0049597, -0.0095270, -0.0047931, -0.0041618, 0.0036135
9: -0.0017234, 0.0015606, -0.0018335, 0.0012936, -0.0023869, 0.0027491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0018352
time: 1.52 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0018587
time: 1.73 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.51 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017550
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017656
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0018080
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0018282
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0017845
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0017981
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0017525, upper bound: 0.0018352
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.51
Output dim: 7, lower bound: -0.0018587, upper bound: 0.0018587

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019662, -0.0004236, -0.0019928, -0.0003938, -0.0012344, 0.0012300
1: -0.0093001, -0.0053855, -0.0093676, -0.0053099, -0.0031325, 0.0031214
2: 0.0292602, 0.0316889, 0.0292184, 0.0317358, -0.0019434, 0.0019365
3: 0.0003914, 0.0049263, 0.0003038, 0.0050044, -0.0036160, 0.0036289
4: -0.0083528, -0.0043709, -0.0084214, -0.0042940, -0.0031863, 0.0031750
5: 0.0105744, 0.0120826, 0.0105484, 0.0121117, -0.0012069, 0.0012026
6: 0.0008848, 0.0066402, 0.0007737, 0.0067394, -0.0045891, 0.0046056
7: 0.9786785, 0.9827058, 0.9786007, 0.9827752, -0.0032112, 0.0032227
8: -0.0094243, -0.0051064, -0.0095077, -0.0050319, -0.0034430, 0.0034553
9: -0.0016265, 0.0012257, -0.0016757, 0.0012808, -0.0022824, 0.0022743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017117
time: 1.34 seconds

## Relational analysis of IS_A1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017117
time: 1.45 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020010, -0.0004174, -0.0020666, -0.0003897, -0.0012644, 0.0012671
1: -0.0093884, -0.0053699, -0.0095550, -0.0052995, -0.0032086, 0.0032154
2: 0.0292054, 0.0316985, 0.0291021, 0.0317422, -0.0019906, 0.0019949
3: 0.0003733, 0.0050286, 0.0002918, 0.0052216, -0.0037249, 0.0037170
4: -0.0084426, -0.0043551, -0.0086120, -0.0042835, -0.0032636, 0.0032706
5: 0.0105403, 0.0120886, 0.0104762, 0.0121157, -0.0012362, 0.0012388
6: 0.0008619, 0.0067701, 0.0007585, 0.0070150, -0.0047274, 0.0047173
7: 0.9786624, 0.9827967, 0.9785900, 0.9829680, -0.0033080, 0.0033009
8: -0.0094415, -0.0050089, -0.0095191, -0.0048252, -0.0035467, 0.0035391
9: -0.0016909, 0.0012370, -0.0018123, 0.0012883, -0.0023378, 0.0023428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017217
time: 1.31 seconds

## Relational analysis of IS_A1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017217
time: 1.35 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0019433, -0.0002737, -0.0019654, -0.0003980, -0.0012535, 0.0013567
1: -0.0092419, -0.0050052, -0.0092980, -0.0053204, -0.0031811, 0.0034428
2: 0.0292963, 0.0319248, 0.0292615, 0.0317292, -0.0019735, 0.0021359
3: -0.0000492, 0.0048588, 0.0003160, 0.0049238, -0.0039883, 0.0036851
4: -0.0082935, -0.0039841, -0.0083506, -0.0043048, -0.0032357, 0.0035019
5: 0.0105968, 0.0122291, 0.0105752, 0.0121076, -0.0012256, 0.0013264
6: 0.0003257, 0.0065546, 0.0007892, 0.0066371, -0.0050617, 0.0046769
7: 0.9782872, 0.9826459, 0.9786115, 0.9827036, -0.0035419, 0.0032726
8: -0.0098438, -0.0051706, -0.0094960, -0.0051087, -0.0037975, 0.0035088
9: -0.0015841, 0.0015028, -0.0016250, 0.0012731, -0.0023178, 0.0025085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017590
time: 1.83 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017673
time: 1.60 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0019800, -0.0002677, -0.0020409, -0.0003946, -0.0012826, 0.0013978
1: -0.0093351, -0.0049899, -0.0094896, -0.0053118, -0.0032548, 0.0035472
2: 0.0292385, 0.0319343, 0.0291426, 0.0317345, -0.0020193, 0.0022007
3: -0.0000669, 0.0049668, 0.0003061, 0.0051459, -0.0041093, 0.0037705
4: -0.0083883, -0.0039686, -0.0085456, -0.0042960, -0.0033107, 0.0036081
5: 0.0105609, 0.0122350, 0.0105013, 0.0121110, -0.0012540, 0.0013667
6: 0.0003032, 0.0066916, 0.0007766, 0.0069189, -0.0052152, 0.0047853
7: 0.9782715, 0.9827418, 0.9786026, 0.9829007, -0.0036494, 0.0033485
8: -0.0098607, -0.0050678, -0.0095055, -0.0048973, -0.0039127, 0.0035901
9: -0.0016520, 0.0015139, -0.0017647, 0.0012793, -0.0023715, 0.0025846

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017839
time: 1.96 seconds

## Relational analysis of IS_A1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017840
time: 1.42 seconds

## BFS IS instance: IS_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020045, -0.0003991, -0.0020083, -0.0003920, -0.0012322, 0.0012648
1: -0.0093972, -0.0053234, -0.0094070, -0.0053054, -0.0031269, 0.0032095
2: 0.0292000, 0.0317274, 0.0291939, 0.0317385, -0.0019400, 0.0019912
3: 0.0003195, 0.0050388, 0.0002986, 0.0050502, -0.0037181, 0.0036224
4: -0.0084515, -0.0043078, -0.0084615, -0.0042895, -0.0031806, 0.0032646
5: 0.0105370, 0.0121065, 0.0105332, 0.0121134, -0.0012047, 0.0012365
6: 0.0007936, 0.0067830, 0.0007671, 0.0067974, -0.0047187, 0.0045973
7: 0.9786146, 0.9828056, 0.9785960, 0.9828158, -0.0033019, 0.0032170
8: -0.0094928, -0.0049993, -0.0095126, -0.0049884, -0.0035402, 0.0034491
9: -0.0016973, 0.0012709, -0.0017045, 0.0012840, -0.0022783, 0.0023385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017420
time: 1.29 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017420
time: 1.88 seconds

## BFS IS instance: IS_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020388, -0.0003928, -0.0020816, -0.0003875, -0.0012644, 0.0013029
1: -0.0094842, -0.0053075, -0.0095929, -0.0052939, -0.0032086, 0.0033063
2: 0.0291460, 0.0317373, 0.0290785, 0.0317456, -0.0019906, 0.0020513
3: 0.0003010, 0.0051396, 0.0002854, 0.0052655, -0.0038302, 0.0037170
4: -0.0085401, -0.0042916, -0.0086507, -0.0042779, -0.0032637, 0.0033631
5: 0.0105034, 0.0121126, 0.0104615, 0.0121178, -0.0012362, 0.0012738
6: 0.0007701, 0.0069109, 0.0007503, 0.0070708, -0.0048611, 0.0047173
7: 0.9785982, 0.9828951, 0.9785843, 0.9830071, -0.0034015, 0.0033010
8: -0.0095104, -0.0049033, -0.0095253, -0.0047833, -0.0036470, 0.0035391
9: -0.0017607, 0.0012825, -0.0018399, 0.0012924, -0.0023378, 0.0024090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017553
time: 2.02 seconds

## Relational analysis of IS_A2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017553
time: 1.82 seconds

## BFS IS instance: IS_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0019785, -0.0002497, -0.0019811, -0.0003962, -0.0012495, 0.0013930
1: -0.0093312, -0.0049441, -0.0093378, -0.0053159, -0.0031707, 0.0035350
2: 0.0292409, 0.0319627, 0.0292368, 0.0317320, -0.0019671, 0.0021931
3: -0.0001199, 0.0049623, 0.0003108, 0.0049700, -0.0040951, 0.0036731
4: -0.0083844, -0.0039220, -0.0083911, -0.0043002, -0.0032251, 0.0035957
5: 0.0105624, 0.0122526, 0.0105599, 0.0121094, -0.0012216, 0.0013619
6: 0.0002360, 0.0066860, 0.0007826, 0.0066956, -0.0051972, 0.0046617
7: 0.9782244, 0.9827378, 0.9786069, 0.9827446, -0.0036367, 0.0032620
8: -0.0099111, -0.0050720, -0.0095010, -0.0050648, -0.0038992, 0.0034974
9: -0.0016492, 0.0015472, -0.0016540, 0.0012764, -0.0023102, 0.0025756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017916
time: 1.53 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017924
time: 1.36 seconds

## BFS IS instance: IS_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020138, -0.0002435, -0.0020552, -0.0003924, -0.0012806, 0.0014320
1: -0.0094210, -0.0049286, -0.0095259, -0.0053063, -0.0032497, 0.0036338
2: 0.0291852, 0.0319723, 0.0291201, 0.0317380, -0.0020161, 0.0022544
3: -0.0001379, 0.0050663, 0.0002996, 0.0051879, -0.0042096, 0.0037647
4: -0.0084757, -0.0039062, -0.0085824, -0.0042904, -0.0033055, 0.0036962
5: 0.0105278, 0.0122586, 0.0104874, 0.0121131, -0.0012520, 0.0014000
6: 0.0002132, 0.0068180, 0.0007684, 0.0069722, -0.0053425, 0.0047778
7: 0.9782084, 0.9828302, 0.9785970, 0.9829380, -0.0037384, 0.0033433
8: -0.0099282, -0.0049730, -0.0095117, -0.0048573, -0.0040082, 0.0035845
9: -0.0017146, 0.0015586, -0.0017911, 0.0012834, -0.0023678, 0.0026476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018118
time: 1.29 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018120
time: 2.06 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.66 seconds
IS_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017117
IS_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017117
IS_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017217
IS_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017217
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017590
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017673
IS_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017839
IS_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017840
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017085, upper bound: 0.0017420
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017420
IS_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018118, upper bound: 0.0017553
IS_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0017553
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017916
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0017111, upper bound: 0.0017924
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018118
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.66
Output dim: 7, lower bound: -0.0018120, upper bound: 0.0018120

## BFS IS instance: IS_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019628, -0.0004677, -0.0019916, -0.0004093, -0.0012077, 0.0011618
1: -0.0092916, -0.0054975, -0.0093646, -0.0053491, -0.0030646, 0.0029483
2: 0.0292655, 0.0316193, 0.0292202, 0.0317114, -0.0019013, 0.0018291
3: 0.0005212, 0.0049164, 0.0003493, 0.0050010, -0.0034154, 0.0035502
4: -0.0083441, -0.0044849, -0.0084184, -0.0043340, -0.0031172, 0.0029989
5: 0.0105777, 0.0120394, 0.0105495, 0.0120966, -0.0011807, 0.0011359
6: 0.0010496, 0.0066277, 0.0008314, 0.0067350, -0.0043346, 0.0045056
7: 0.9787937, 0.9826970, 0.9786411, 0.9827721, -0.0030332, 0.0031528
8: -0.0093007, -0.0051158, -0.0094644, -0.0050352, -0.0032520, 0.0033803
9: -0.0016203, 0.0011440, -0.0016735, 0.0012521, -0.0022329, 0.0021482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017117
time: 1.96 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017117
time: 2.25 seconds

## BFS IS instance: IS_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020137, -0.0004478, -0.0019912, -0.0004168, -0.0013012, 0.0011938
1: -0.0094207, -0.0054470, -0.0093635, -0.0053684, -0.0033020, 0.0030294
2: 0.0291854, 0.0316507, 0.0292209, 0.0316995, -0.0020486, 0.0018795
3: 0.0004626, 0.0050660, 0.0003716, 0.0049997, -0.0035094, 0.0038252
4: -0.0084755, -0.0044335, -0.0084172, -0.0043536, -0.0033587, 0.0030814
5: 0.0105279, 0.0120589, 0.0105500, 0.0120892, -0.0012722, 0.0011672
6: 0.0009752, 0.0068176, 0.0008597, 0.0067334, -0.0044539, 0.0048547
7: 0.9787417, 0.9828299, 0.9786609, 0.9827710, -0.0031167, 0.0033971
8: -0.0093565, -0.0049733, -0.0094432, -0.0050365, -0.0033415, 0.0036422
9: -0.0017145, 0.0011809, -0.0016727, 0.0012381, -0.0024059, 0.0022073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017117
time: 1.94 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017117
time: 2.13 seconds

## BFS IS instance: IS_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019976, -0.0004615, -0.0020655, -0.0004048, -0.0012374, 0.0011983
1: -0.0093799, -0.0054818, -0.0095520, -0.0053378, -0.0031400, 0.0030410
2: 0.0292107, 0.0316291, 0.0291039, 0.0317184, -0.0019480, 0.0018866
3: 0.0005029, 0.0050187, 0.0003362, 0.0052181, -0.0035228, 0.0036375
4: -0.0084339, -0.0044689, -0.0086090, -0.0043225, -0.0031939, 0.0030932
5: 0.0105436, 0.0120455, 0.0104773, 0.0121009, -0.0012098, 0.0011716
6: 0.0010264, 0.0067575, 0.0008147, 0.0070105, -0.0044709, 0.0046165
7: 0.9787775, 0.9827878, 0.9786294, 0.9829649, -0.0031285, 0.0032304
8: -0.0093181, -0.0050183, -0.0094769, -0.0048285, -0.0033543, 0.0034635
9: -0.0016847, 0.0011555, -0.0018101, 0.0012604, -0.0022878, 0.0022157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017682, upper bound: 0.0017217
time: 1.40 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017682, upper bound: 0.0017217
time: 1.97 seconds

## BFS IS instance: IS_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020507, -0.0004416, -0.0020650, -0.0004128, -0.0013310, 0.0012328
1: -0.0095146, -0.0054311, -0.0095509, -0.0053581, -0.0033776, 0.0031283
2: 0.0291271, 0.0316605, 0.0291046, 0.0317058, -0.0020955, 0.0019408
3: 0.0004443, 0.0051748, 0.0003597, 0.0052168, -0.0036240, 0.0039128
4: -0.0085710, -0.0044174, -0.0086079, -0.0043431, -0.0034356, 0.0031820
5: 0.0104917, 0.0120650, 0.0104777, 0.0120931, -0.0013013, 0.0012053
6: 0.0009519, 0.0069556, 0.0008447, 0.0070090, -0.0045993, 0.0049659
7: 0.9787254, 0.9829265, 0.9786503, 0.9829639, -0.0032184, 0.0034749
8: -0.0093740, -0.0048698, -0.0094545, -0.0048297, -0.0034506, 0.0037256
9: -0.0017828, 0.0011924, -0.0018093, 0.0012456, -0.0024610, 0.0022793

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017709, upper bound: 0.0017217
time: 1.90 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017709, upper bound: 0.0017217
time: 2.15 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019420, -0.0002889, -0.0019620, -0.0004407, -0.0011876, 0.0013307
1: -0.0092386, -0.0050436, -0.0092893, -0.0054289, -0.0030138, 0.0033769
2: 0.0292984, 0.0319009, 0.0292669, 0.0316619, -0.0018698, 0.0020950
3: -0.0000046, 0.0048550, 0.0004417, 0.0049138, -0.0039120, 0.0034913
4: -0.0082902, -0.0040232, -0.0083418, -0.0044152, -0.0030655, 0.0034349
5: 0.0105981, 0.0122143, 0.0105785, 0.0120658, -0.0011611, 0.0013010
6: 0.0003823, 0.0065498, 0.0009487, 0.0066244, -0.0049648, 0.0044309
7: 0.9783268, 0.9826424, 0.9787232, 0.9826947, -0.0034741, 0.0031006
8: -0.0098014, -0.0051742, -0.0093764, -0.0051182, -0.0037248, 0.0033243
9: -0.0015817, 0.0014748, -0.0016187, 0.0011940, -0.0021959, 0.0024604

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017089
time: 1.54 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017070
time: 1.62 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019415, -0.0002961, -0.0020091, -0.0004254, -0.0012254, 0.0014118
1: -0.0092374, -0.0050620, -0.0094091, -0.0053900, -0.0031097, 0.0035826
2: 0.0292991, 0.0318895, 0.0291926, 0.0316860, -0.0019292, 0.0022227
3: 0.0000167, 0.0048537, 0.0003966, 0.0050525, -0.0041503, 0.0036024
4: -0.0082891, -0.0040420, -0.0084636, -0.0043756, -0.0031630, 0.0036441
5: 0.0105985, 0.0122072, 0.0105324, 0.0120808, -0.0011981, 0.0013803
6: 0.0004093, 0.0065481, 0.0008915, 0.0068004, -0.0052672, 0.0045719
7: 0.9783457, 0.9826413, 0.9786832, 0.9828179, -0.0036857, 0.0031992
8: -0.0097811, -0.0051755, -0.0094193, -0.0049862, -0.0039517, 0.0034300
9: -0.0015809, 0.0014613, -0.0017060, 0.0012224, -0.0022657, 0.0026103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017153
time: 1.46 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017138
time: 1.93 seconds

## BFS IS instance: IS_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019762, -0.0003099, -0.0020397, -0.0004097, -0.0012552, 0.0013360
1: -0.0093254, -0.0050969, -0.0094866, -0.0053502, -0.0031853, 0.0033903
2: 0.0292445, 0.0318679, 0.0291445, 0.0317107, -0.0019762, 0.0021034
3: 0.0000571, 0.0049556, 0.0003505, 0.0051424, -0.0039275, 0.0036900
4: -0.0083785, -0.0040774, -0.0085425, -0.0043351, -0.0032400, 0.0034485
5: 0.0105646, 0.0121938, 0.0105025, 0.0120962, -0.0012272, 0.0013062
6: 0.0004606, 0.0066774, 0.0008330, 0.0069144, -0.0049846, 0.0046831
7: 0.9783816, 0.9827319, 0.9786422, 0.9828977, -0.0034880, 0.0032770
8: -0.0097426, -0.0050784, -0.0094632, -0.0049006, -0.0037396, 0.0035134
9: -0.0016450, 0.0014359, -0.0017625, 0.0012514, -0.0023208, 0.0024702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017319
time: 1.88 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017314
time: 1.46 seconds

## BFS IS instance: IS_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020277, -0.0002926, -0.0020393, -0.0004177, -0.0013496, 0.0013749
1: -0.0094561, -0.0050530, -0.0094855, -0.0053706, -0.0034247, 0.0034891
2: 0.0291634, 0.0318951, 0.0291452, 0.0316981, -0.0021247, 0.0021646
3: 0.0000063, 0.0051071, 0.0003741, 0.0051411, -0.0040419, 0.0039674
4: -0.0085115, -0.0040328, -0.0085414, -0.0043558, -0.0034835, 0.0035490
5: 0.0105143, 0.0122107, 0.0105029, 0.0120883, -0.0013195, 0.0013443
6: 0.0003961, 0.0068697, 0.0008629, 0.0069129, -0.0051297, 0.0050351
7: 0.9783365, 0.9828663, 0.9786631, 0.9828966, -0.0035895, 0.0035233
8: -0.0097910, -0.0049342, -0.0094407, -0.0049018, -0.0038485, 0.0037776
9: -0.0017403, 0.0014679, -0.0017617, 0.0012365, -0.0024953, 0.0025422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017319
time: 1.44 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017314
time: 1.50 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020011, -0.0004418, -0.0020072, -0.0004075, -0.0012046, 0.0011964
1: -0.0093885, -0.0054317, -0.0094040, -0.0053446, -0.0030568, 0.0030359
2: 0.0292053, 0.0316602, 0.0291957, 0.0317142, -0.0018965, 0.0018835
3: 0.0004450, 0.0050288, 0.0003441, 0.0050467, -0.0035170, 0.0035412
4: -0.0084428, -0.0044180, -0.0084585, -0.0043294, -0.0031093, 0.0030880
5: 0.0105403, 0.0120648, 0.0105343, 0.0120983, -0.0011777, 0.0011697
6: 0.0009528, 0.0067703, 0.0008248, 0.0067930, -0.0044635, 0.0044942
7: 0.9787260, 0.9827968, 0.9786364, 0.9828128, -0.0031233, 0.0031448
8: -0.0093733, -0.0050088, -0.0094694, -0.0049917, -0.0033487, 0.0033717
9: -0.0016910, 0.0011920, -0.0017023, 0.0012554, -0.0022272, 0.0022120

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017420
time: 1.56 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017420
time: 2.10 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020466, -0.0004263, -0.0020067, -0.0004151, -0.0013000, 0.0012267
1: -0.0095040, -0.0053924, -0.0094029, -0.0053638, -0.0032988, 0.0031129
2: 0.0291337, 0.0316846, 0.0291964, 0.0317023, -0.0020466, 0.0019313
3: 0.0003994, 0.0051625, 0.0003663, 0.0050454, -0.0036061, 0.0038215
4: -0.0085602, -0.0043780, -0.0084573, -0.0043489, -0.0033555, 0.0031663
5: 0.0104958, 0.0120799, 0.0105348, 0.0120909, -0.0012710, 0.0011993
6: 0.0008950, 0.0069400, 0.0008530, 0.0067913, -0.0045767, 0.0048500
7: 0.9786856, 0.9829155, 0.9786562, 0.9828115, -0.0032025, 0.0033938
8: -0.0094167, -0.0048814, -0.0094482, -0.0049930, -0.0034336, 0.0036387
9: -0.0017751, 0.0012207, -0.0017014, 0.0012415, -0.0024036, 0.0022681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017420
time: 1.92 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017420
time: 2.18 seconds

## BFS IS instance: IS_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020354, -0.0004354, -0.0020804, -0.0004026, -0.0012372, 0.0012361
1: -0.0094756, -0.0054156, -0.0095899, -0.0053321, -0.0031396, 0.0031368
2: 0.0291513, 0.0316702, 0.0290804, 0.0317219, -0.0019478, 0.0019461
3: 0.0004263, 0.0051296, 0.0003296, 0.0052621, -0.0036338, 0.0036370
4: -0.0085313, -0.0044016, -0.0086476, -0.0043167, -0.0031935, 0.0031906
5: 0.0105068, 0.0120710, 0.0104627, 0.0121031, -0.0012096, 0.0012085
6: 0.0009291, 0.0068983, 0.0008064, 0.0070664, -0.0046118, 0.0046159
7: 0.9787094, 0.9828863, 0.9786236, 0.9830039, -0.0032271, 0.0032300
8: -0.0093911, -0.0049128, -0.0094831, -0.0047867, -0.0034600, 0.0034630
9: -0.0017544, 0.0012037, -0.0018377, 0.0012646, -0.0022875, 0.0022855

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017685, upper bound: 0.0017553
time: 1.33 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017685, upper bound: 0.0017553
time: 1.87 seconds

## BFS IS instance: IS_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020832, -0.0004199, -0.0020800, -0.0004106, -0.0013323, 0.0012704
1: -0.0095969, -0.0053762, -0.0095888, -0.0053525, -0.0033808, 0.0032239
2: 0.0290761, 0.0316946, 0.0290811, 0.0317093, -0.0020974, 0.0020001
3: 0.0003806, 0.0052701, 0.0003531, 0.0052608, -0.0037348, 0.0039165
4: -0.0086547, -0.0043615, -0.0086465, -0.0043374, -0.0034388, 0.0032793
5: 0.0104600, 0.0120862, 0.0104631, 0.0120953, -0.0013025, 0.0012421
6: 0.0008712, 0.0070766, 0.0008363, 0.0070647, -0.0047399, 0.0049705
7: 0.9786689, 0.9830111, 0.9786445, 0.9830028, -0.0033168, 0.0034781
8: -0.0094346, -0.0047790, -0.0094607, -0.0047879, -0.0035561, 0.0037291
9: -0.0018428, 0.0012325, -0.0018369, 0.0012497, -0.0024633, 0.0023490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017710, upper bound: 0.0017553
time: 1.43 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017710, upper bound: 0.0017553
time: 2.03 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019772, -0.0002644, -0.0019777, -0.0004388, -0.0011814, 0.0013671
1: -0.0093279, -0.0049816, -0.0093292, -0.0054242, -0.0029979, 0.0034691
2: 0.0292430, 0.0319394, 0.0292422, 0.0316648, -0.0018599, 0.0021523
3: -0.0000765, 0.0049585, 0.0004363, 0.0049600, -0.0040188, 0.0034730
4: -0.0083811, -0.0039601, -0.0083824, -0.0044104, -0.0030494, 0.0035287
5: 0.0105637, 0.0122382, 0.0105632, 0.0120676, -0.0011550, 0.0013366
6: 0.0002911, 0.0066811, 0.0009418, 0.0066830, -0.0051004, 0.0044076
7: 0.9782630, 0.9827344, 0.9787183, 0.9827357, -0.0035690, 0.0030843
8: -0.0098698, -0.0050757, -0.0093816, -0.0050743, -0.0038265, 0.0033068
9: -0.0016468, 0.0015199, -0.0016477, 0.0011975, -0.0021843, 0.0025276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017426
time: 1.32 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017389
time: 1.91 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019767, -0.0002728, -0.0020231, -0.0004234, -0.0012193, 0.0014448
1: -0.0093266, -0.0050029, -0.0094444, -0.0053850, -0.0030941, 0.0036663
2: 0.0292437, 0.0319262, 0.0291706, 0.0316892, -0.0019196, 0.0022746
3: -0.0000518, 0.0049570, 0.0003908, 0.0050935, -0.0042472, 0.0035844
4: -0.0083798, -0.0039818, -0.0084996, -0.0043704, -0.0031473, 0.0037292
5: 0.0105641, 0.0122300, 0.0105188, 0.0120828, -0.0011921, 0.0014125
6: 0.0003224, 0.0066793, 0.0008841, 0.0068525, -0.0053903, 0.0045491
7: 0.9782848, 0.9827331, 0.9786779, 0.9828543, -0.0037719, 0.0031832
8: -0.0098463, -0.0050771, -0.0094249, -0.0049471, -0.0040440, 0.0034129
9: -0.0016459, 0.0015044, -0.0017317, 0.0012261, -0.0022544, 0.0026713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017431
time: 1.79 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017390
time: 1.59 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0020125, -0.0002583, -0.0020517, -0.0004345, -0.0012123, 0.0014068
1: -0.0094176, -0.0049660, -0.0095171, -0.0054133, -0.0030764, 0.0035699
2: 0.0291873, 0.0319491, 0.0291256, 0.0316716, -0.0019086, 0.0022148
3: -0.0000945, 0.0050625, 0.0004236, 0.0051777, -0.0041356, 0.0035639
4: -0.0084723, -0.0039443, -0.0085735, -0.0043993, -0.0031292, 0.0036312
5: 0.0105291, 0.0122442, 0.0104908, 0.0120719, -0.0011853, 0.0013754
6: 0.0002682, 0.0068130, 0.0009258, 0.0069593, -0.0052486, 0.0045230
7: 0.9782469, 0.9828267, 0.9787071, 0.9829291, -0.0036727, 0.0031650
8: -0.0098870, -0.0049767, -0.0093936, -0.0048670, -0.0039377, 0.0033934
9: -0.0017122, 0.0015313, -0.0017847, 0.0012054, -0.0022415, 0.0026011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B2_B1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017623
time: 1.52 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017581
time: 1.50 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0020120, -0.0002668, -0.0021011, -0.0004196, -0.0012453, 0.0014844
1: -0.0094164, -0.0049875, -0.0096425, -0.0053753, -0.0031601, 0.0037668
2: 0.0291881, 0.0319357, 0.0290478, 0.0316952, -0.0019605, 0.0023369
3: -0.0000696, 0.0050610, 0.0003796, 0.0053230, -0.0043636, 0.0036608
4: -0.0084711, -0.0039662, -0.0087011, -0.0043606, -0.0032143, 0.0038314
5: 0.0105296, 0.0122359, 0.0104425, 0.0120865, -0.0012175, 0.0014512
6: 0.0002998, 0.0068112, 0.0008698, 0.0071436, -0.0055380, 0.0046460
7: 0.9782690, 0.9828255, 0.9786679, 0.9830580, -0.0038752, 0.0032510
8: -0.0098632, -0.0049781, -0.0094356, -0.0047287, -0.0041549, 0.0034856
9: -0.0017113, 0.0015156, -0.0018760, 0.0012331, -0.0023025, 0.0027445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A2_B2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017619
time: 1.68 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017579
time: 1.54 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.52 seconds
IS_A1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017117
IS_A1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017117
IS_A1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017117
IS_A1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017117
IS_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017682, upper bound: 0.0017217
IS_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017682, upper bound: 0.0017217
IS_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017709, upper bound: 0.0017217
IS_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017709, upper bound: 0.0017217
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017089
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017070
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017153
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017138
IS_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017319
IS_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017581, upper bound: 0.0017314
IS_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017319
IS_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017314
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017420
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016883, upper bound: 0.0017420
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017420
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016927, upper bound: 0.0017420
IS_A2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017685, upper bound: 0.0017553
IS_A2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017685, upper bound: 0.0017553
IS_A2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017710, upper bound: 0.0017553
IS_A2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017710, upper bound: 0.0017553
IS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017426
IS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016546, upper bound: 0.0017389
IS_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017431
IS_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017390
IS_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017623
IS_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017581
IS_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017619
IS_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.52
Output dim: 7, lower bound: -0.0017579, upper bound: 0.0017579

## BFS IS instance: IS_A1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019628, -0.0004677, -0.0019536, -0.0004131, -0.0012043, 0.0011261
1: -0.0092916, -0.0054975, -0.0092681, -0.0053589, -0.0030562, 0.0028577
2: 0.0292655, 0.0316193, 0.0292801, 0.0317053, -0.0018961, 0.0017729
3: 0.0005212, 0.0049164, 0.0003607, 0.0048892, -0.0033105, 0.0035404
4: -0.0083441, -0.0044849, -0.0083202, -0.0043440, -0.0031086, 0.0029068
5: 0.0105777, 0.0120394, 0.0105867, 0.0120928, -0.0011775, 0.0011010
6: 0.0010496, 0.0066277, 0.0008458, 0.0065932, -0.0042015, 0.0044933
7: 0.9787937, 0.9826970, 0.9786512, 0.9826729, -0.0029400, 0.0031442
8: -0.0093007, -0.0051158, -0.0094536, -0.0051417, -0.0031521, 0.0033710
9: -0.0016203, 0.0011440, -0.0016032, 0.0012450, -0.0022268, 0.0020822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016371, upper bound: 0.0016637
time: 1.77 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016631
time: 1.38 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019628, -0.0004677, -0.0019267, -0.0002625, -0.0013598, 0.0011078
1: -0.0092916, -0.0054975, -0.0091999, -0.0049768, -0.0034506, 0.0028111
2: 0.0292655, 0.0316193, 0.0293224, 0.0319424, -0.0021407, 0.0017440
3: 0.0005212, 0.0049164, -0.0000821, 0.0048102, -0.0032566, 0.0039973
4: -0.0083441, -0.0044849, -0.0082509, -0.0039552, -0.0035098, 0.0028594
5: 0.0105777, 0.0120394, 0.0106130, 0.0122400, -0.0013294, 0.0010831
6: 0.0010496, 0.0066277, 0.0002840, 0.0064929, -0.0041330, 0.0050731
7: 0.9787937, 0.9826970, 0.9782580, 0.9826027, -0.0028921, 0.0035499
8: -0.0093007, -0.0051158, -0.0098751, -0.0052169, -0.0031008, 0.0038061
9: -0.0016203, 0.0011440, -0.0015536, 0.0015235, -0.0025141, 0.0020482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016492, upper bound: 0.0016631
time: 2.03 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016631
time: 2.16 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020137, -0.0004478, -0.0019531, -0.0004207, -0.0012979, 0.0011581
1: -0.0094207, -0.0054470, -0.0092669, -0.0053781, -0.0032935, 0.0029388
2: 0.0291854, 0.0316507, 0.0292808, 0.0316934, -0.0020433, 0.0018233
3: 0.0004626, 0.0050660, 0.0003829, 0.0048879, -0.0034045, 0.0038154
4: -0.0084755, -0.0044335, -0.0083190, -0.0043635, -0.0033501, 0.0029893
5: 0.0105279, 0.0120589, 0.0105872, 0.0120854, -0.0012689, 0.0011323
6: 0.0009752, 0.0068176, 0.0008741, 0.0065915, -0.0043207, 0.0048423
7: 0.9787417, 0.9828299, 0.9786709, 0.9826716, -0.0030234, 0.0033884
8: -0.0093565, -0.0049733, -0.0094324, -0.0051430, -0.0032416, 0.0036329
9: -0.0017145, 0.0011809, -0.0016024, 0.0012310, -0.0023997, 0.0021413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016631
time: 1.41 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016631
time: 1.56 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020137, -0.0004478, -0.0019263, -0.0002713, -0.0014474, 0.0011397
1: -0.0094207, -0.0054470, -0.0091987, -0.0049992, -0.0036730, 0.0028922
2: 0.0291854, 0.0316507, 0.0293231, 0.0319285, -0.0022787, 0.0017943
3: 0.0004626, 0.0050660, -0.0000561, 0.0048089, -0.0033505, 0.0042550
4: -0.0084755, -0.0044335, -0.0082497, -0.0039780, -0.0037360, 0.0029419
5: 0.0105279, 0.0120589, 0.0106134, 0.0122314, -0.0014151, 0.0011143
6: 0.0009752, 0.0068176, 0.0003169, 0.0064912, -0.0042522, 0.0054001
7: 0.9787417, 0.9828299, 0.9782810, 0.9826015, -0.0029755, 0.0037787
8: -0.0093565, -0.0049733, -0.0098504, -0.0052182, -0.0031902, 0.0040514
9: -0.0017145, 0.0011809, -0.0015527, 0.0015072, -0.0026762, 0.0021073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016631
time: 2.00 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016631
time: 2.10 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019976, -0.0004615, -0.0020269, -0.0004091, -0.0012335, 0.0011623
1: -0.0093799, -0.0054818, -0.0094542, -0.0053487, -0.0031303, 0.0029496
2: 0.0292107, 0.0316291, 0.0291646, 0.0317116, -0.0019420, 0.0018299
3: 0.0005029, 0.0050187, 0.0003488, 0.0051049, -0.0034170, 0.0036263
4: -0.0084339, -0.0044689, -0.0085096, -0.0043336, -0.0031840, 0.0030002
5: 0.0105436, 0.0120455, 0.0105150, 0.0120967, -0.0012060, 0.0011364
6: 0.0010264, 0.0067575, 0.0008309, 0.0068669, -0.0043366, 0.0046022
7: 0.9787775, 0.9827878, 0.9786407, 0.9828644, -0.0030345, 0.0032204
8: -0.0093181, -0.0050183, -0.0094648, -0.0049363, -0.0032535, 0.0034528
9: -0.0016847, 0.0011555, -0.0017389, 0.0012524, -0.0022808, 0.0021491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017198, upper bound: 0.0016728
time: 1.53 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017198, upper bound: 0.0016732
time: 1.36 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019976, -0.0004615, -0.0020050, -0.0002593, -0.0013890, 0.0011450
1: -0.0093799, -0.0054818, -0.0093985, -0.0049686, -0.0035247, 0.0029056
2: 0.0292107, 0.0316291, 0.0291991, 0.0319475, -0.0021867, 0.0018027
3: 0.0005029, 0.0050187, -0.0000916, 0.0050403, -0.0033661, 0.0040832
4: -0.0084339, -0.0044689, -0.0084529, -0.0039469, -0.0035852, 0.0029555
5: 0.0105436, 0.0120455, 0.0105364, 0.0122432, -0.0013580, 0.0011195
6: 0.0010264, 0.0067575, 0.0002719, 0.0067850, -0.0042720, 0.0051821
7: 0.9787775, 0.9827878, 0.9782495, 0.9828070, -0.0029893, 0.0036262
8: -0.0093181, -0.0050183, -0.0098842, -0.0049978, -0.0032050, 0.0038878
9: -0.0016847, 0.0011555, -0.0016983, 0.0015295, -0.0025681, 0.0021171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017227, upper bound: 0.0016732
time: 1.93 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017198, upper bound: 0.0016732
time: 2.09 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020507, -0.0004416, -0.0020265, -0.0004171, -0.0013272, 0.0011967
1: -0.0095146, -0.0054311, -0.0094531, -0.0053691, -0.0033679, 0.0030369
2: 0.0291271, 0.0316605, 0.0291653, 0.0316990, -0.0020895, 0.0018841
3: 0.0004443, 0.0051748, 0.0003724, 0.0051036, -0.0035181, 0.0039015
4: -0.0085710, -0.0044174, -0.0085084, -0.0043543, -0.0034257, 0.0030890
5: 0.0104917, 0.0120650, 0.0105154, 0.0120889, -0.0012976, 0.0011700
6: 0.0009519, 0.0069556, 0.0008607, 0.0068652, -0.0044649, 0.0049516
7: 0.9787254, 0.9829265, 0.9786615, 0.9828632, -0.0031243, 0.0034649
8: -0.0093740, -0.0048698, -0.0094424, -0.0049376, -0.0033498, 0.0037149
9: -0.0017828, 0.0011924, -0.0017381, 0.0012376, -0.0024539, 0.0022127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017252, upper bound: 0.0016732
time: 1.88 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017214, upper bound: 0.0016732
time: 1.57 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020507, -0.0004416, -0.0020045, -0.0002678, -0.0014783, 0.0011794
1: -0.0095146, -0.0054311, -0.0093974, -0.0049902, -0.0037515, 0.0029929
2: 0.0291271, 0.0316605, 0.0291999, 0.0319341, -0.0023274, 0.0018568
3: 0.0004443, 0.0051748, -0.0000665, 0.0050390, -0.0034671, 0.0043459
4: -0.0085710, -0.0044174, -0.0084517, -0.0039689, -0.0038159, 0.0030443
5: 0.0104917, 0.0120650, 0.0105369, 0.0122349, -0.0014454, 0.0011531
6: 0.0009519, 0.0069556, 0.0003037, 0.0067832, -0.0044002, 0.0055155
7: 0.9787254, 0.9829265, 0.9782719, 0.9828058, -0.0030790, 0.0038595
8: -0.0093740, -0.0048698, -0.0098603, -0.0049991, -0.0033012, 0.0041380
9: -0.0017828, 0.0011924, -0.0016974, 0.0015137, -0.0027334, 0.0021806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017252, upper bound: 0.0016732
time: 2.01 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017214, upper bound: 0.0016732
time: 2.23 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019410, -0.0003043, -0.0019592, -0.0004813, -0.0011425, 0.0013094
1: -0.0092362, -0.0050829, -0.0092823, -0.0055319, -0.0028993, 0.0033228
2: 0.0292998, 0.0318766, 0.0292713, 0.0315980, -0.0017987, 0.0020615
3: 0.0000409, 0.0048523, 0.0005611, 0.0049057, -0.0038493, 0.0033587
4: -0.0082878, -0.0040632, -0.0083347, -0.0045199, -0.0029490, 0.0033798
5: 0.0105990, 0.0121992, 0.0105812, 0.0120261, -0.0011170, 0.0012802
6: 0.0004400, 0.0065463, 0.0011002, 0.0066140, -0.0048852, 0.0042626
7: 0.9783671, 0.9826400, 0.9788291, 0.9826874, -0.0034184, 0.0029827
8: -0.0097581, -0.0051768, -0.0092627, -0.0051260, -0.0036651, 0.0031980
9: -0.0015800, 0.0014462, -0.0016136, 0.0011190, -0.0021124, 0.0024210

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017089
time: 1.88 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017089
time: 1.32 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019406, -0.0003161, -0.0019695, -0.0004960, -0.0011453, 0.0013292
1: -0.0092350, -0.0051127, -0.0093085, -0.0055693, -0.0029064, 0.0033731
2: 0.0293006, 0.0318581, 0.0292550, 0.0315748, -0.0018032, 0.0020927
3: 0.0000754, 0.0048509, 0.0006043, 0.0049360, -0.0039076, 0.0033670
4: -0.0082866, -0.0040935, -0.0083613, -0.0045579, -0.0029563, 0.0034310
5: 0.0105994, 0.0121877, 0.0105711, 0.0120118, -0.0011198, 0.0012996
6: 0.0004838, 0.0065445, 0.0011551, 0.0066526, -0.0049593, 0.0042731
7: 0.9783977, 0.9826388, 0.9788675, 0.9827144, -0.0034702, 0.0029901
8: -0.0097252, -0.0051781, -0.0092216, -0.0050971, -0.0037207, 0.0032059
9: -0.0015791, 0.0014245, -0.0016327, 0.0010918, -0.0021177, 0.0024577

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017070
time: 2.14 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017070
time: 2.03 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0019406, -0.0003116, -0.0020063, -0.0004659, -0.0011565, 0.0013901
1: -0.0092351, -0.0051012, -0.0094020, -0.0054928, -0.0029349, 0.0035276
2: 0.0293005, 0.0318652, 0.0291970, 0.0316223, -0.0018208, 0.0021886
3: 0.0000621, 0.0048510, 0.0005157, 0.0050443, -0.0040866, 0.0033999
4: -0.0082867, -0.0040818, -0.0084564, -0.0044801, -0.0029853, 0.0035882
5: 0.0105994, 0.0121921, 0.0105351, 0.0120412, -0.0011307, 0.0013591
6: 0.0004670, 0.0065447, 0.0010426, 0.0067900, -0.0051864, 0.0043150
7: 0.9783860, 0.9826389, 0.9787889, 0.9828106, -0.0036292, 0.0030194
8: -0.0097378, -0.0051781, -0.0093060, -0.0049940, -0.0038911, 0.0032373
9: -0.0015792, 0.0014328, -0.0017008, 0.0011475, -0.0021384, 0.0025703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017153
time: 1.50 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017153
time: 2.13 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0019401, -0.0003232, -0.0020168, -0.0004806, -0.0011597, 0.0014060
1: -0.0092339, -0.0051308, -0.0094284, -0.0055303, -0.0029428, 0.0035679
2: 0.0293013, 0.0318469, 0.0291806, 0.0315990, -0.0018258, 0.0022136
3: 0.0000963, 0.0048496, 0.0005591, 0.0050750, -0.0041333, 0.0034091
4: -0.0082854, -0.0041119, -0.0084833, -0.0045182, -0.0029934, 0.0036292
5: 0.0105999, 0.0121807, 0.0105249, 0.0120268, -0.0011338, 0.0013746
6: 0.0005104, 0.0065429, 0.0010977, 0.0068289, -0.0052457, 0.0043266
7: 0.9784163, 0.9826377, 0.9788275, 0.9828379, -0.0036707, 0.0030276
8: -0.0097053, -0.0051794, -0.0092646, -0.0049648, -0.0039355, 0.0032460
9: -0.0015783, 0.0014113, -0.0017201, 0.0011202, -0.0021442, 0.0025996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017138
time: 1.67 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017138
time: 1.62 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0019753, -0.0003256, -0.0020370, -0.0004530, -0.0012071, 0.0013153
1: -0.0093232, -0.0051369, -0.0094797, -0.0054601, -0.0030631, 0.0033378
2: 0.0292459, 0.0318431, 0.0291488, 0.0316426, -0.0019004, 0.0020708
3: 0.0001034, 0.0049530, 0.0004778, 0.0051343, -0.0038667, 0.0035485
4: -0.0083762, -0.0041181, -0.0085355, -0.0044468, -0.0031157, 0.0033951
5: 0.0105655, 0.0121784, 0.0105052, 0.0120538, -0.0011801, 0.0012860
6: 0.0005194, 0.0066741, 0.0009945, 0.0069043, -0.0049073, 0.0045035
7: 0.9784227, 0.9827295, 0.9787552, 0.9828905, -0.0034339, 0.0031513
8: -0.0096985, -0.0050809, -0.0093420, -0.0049083, -0.0036817, 0.0033787
9: -0.0016434, 0.0014068, -0.0017574, 0.0011713, -0.0022318, 0.0024319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016281
time: 1.80 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0017319
time: 1.97 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0019748, -0.0003375, -0.0020494, -0.0004661, -0.0012097, 0.0013351
1: -0.0093220, -0.0051670, -0.0095113, -0.0054933, -0.0030697, 0.0033880
2: 0.0292466, 0.0318244, 0.0291292, 0.0316219, -0.0019044, 0.0021019
3: 0.0001383, 0.0049517, 0.0005163, 0.0051710, -0.0039248, 0.0035561
4: -0.0083751, -0.0041487, -0.0085676, -0.0044807, -0.0031224, 0.0034461
5: 0.0105659, 0.0121668, 0.0104930, 0.0120410, -0.0011827, 0.0013053
6: 0.0005636, 0.0066724, 0.0010434, 0.0069508, -0.0049811, 0.0045131
7: 0.9784536, 0.9827284, 0.9787894, 0.9829232, -0.0034855, 0.0031580
8: -0.0096653, -0.0050822, -0.0093053, -0.0048734, -0.0037370, 0.0033859
9: -0.0016425, 0.0013849, -0.0017805, 0.0011471, -0.0022366, 0.0024685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016226
time: 2.07 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0017314
time: 1.97 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020268, -0.0003081, -0.0020366, -0.0004610, -0.0013015, 0.0013534
1: -0.0094538, -0.0050925, -0.0094786, -0.0054804, -0.0033027, 0.0034345
2: 0.0291648, 0.0318706, 0.0291494, 0.0316300, -0.0020490, 0.0021308
3: 0.0000519, 0.0051044, 0.0005014, 0.0051331, -0.0039787, 0.0038260
4: -0.0085091, -0.0040729, -0.0085344, -0.0044675, -0.0033594, 0.0034935
5: 0.0105152, 0.0121955, 0.0105056, 0.0120460, -0.0012724, 0.0013232
6: 0.0004540, 0.0068662, 0.0010244, 0.0069027, -0.0050495, 0.0048556
7: 0.9783769, 0.9828639, 0.9787761, 0.9828894, -0.0035334, 0.0033977
8: -0.0097475, -0.0049368, -0.0093196, -0.0049094, -0.0037884, 0.0036429
9: -0.0017386, 0.0014392, -0.0017566, 0.0011565, -0.0024064, 0.0025024

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016281
time: 2.04 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017319
time: 2.01 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020263, -0.0003199, -0.0020490, -0.0004748, -0.0013039, 0.0013736
1: -0.0094526, -0.0051223, -0.0095102, -0.0055154, -0.0033089, 0.0034857
2: 0.0291656, 0.0318521, 0.0291298, 0.0316083, -0.0020528, 0.0021626
3: 0.0000865, 0.0051030, 0.0005419, 0.0051698, -0.0040380, 0.0038332
4: -0.0085079, -0.0041032, -0.0085665, -0.0045031, -0.0033657, 0.0035456
5: 0.0105156, 0.0121840, 0.0104934, 0.0120325, -0.0012748, 0.0013430
6: 0.0004979, 0.0068645, 0.0010758, 0.0069492, -0.0051248, 0.0048648
7: 0.9784077, 0.9828627, 0.9788121, 0.9829220, -0.0035861, 0.0034042
8: -0.0097146, -0.0049381, -0.0092810, -0.0048746, -0.0038448, 0.0036498
9: -0.0017377, 0.0014174, -0.0017797, 0.0011310, -0.0024109, 0.0025397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016226
time: 2.05 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017314
time: 2.05 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020011, -0.0004418, -0.0019694, -0.0004113, -0.0012012, 0.0011618
1: -0.0093885, -0.0054317, -0.0093082, -0.0053544, -0.0030481, 0.0029483
2: 0.0292053, 0.0316602, 0.0292552, 0.0317081, -0.0018911, 0.0018291
3: 0.0004450, 0.0050288, 0.0003554, 0.0049356, -0.0034155, 0.0035311
4: -0.0084428, -0.0044180, -0.0083610, -0.0043393, -0.0031004, 0.0029989
5: 0.0105403, 0.0120648, 0.0105713, 0.0120946, -0.0011744, 0.0011359
6: 0.0009528, 0.0067703, 0.0008391, 0.0066521, -0.0043347, 0.0044814
7: 0.9787260, 0.9827968, 0.9786465, 0.9827141, -0.0030332, 0.0031359
8: -0.0093733, -0.0050088, -0.0094586, -0.0050975, -0.0032521, 0.0033621
9: -0.0016910, 0.0011920, -0.0016324, 0.0012483, -0.0022209, 0.0021482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016371, upper bound: 0.0016965
time: 1.90 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016927
time: 1.78 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020011, -0.0004418, -0.0019423, -0.0002606, -0.0013597, 0.0011402
1: -0.0093885, -0.0054317, -0.0092395, -0.0049719, -0.0034504, 0.0028933
2: 0.0292053, 0.0316602, 0.0292978, 0.0319455, -0.0021406, 0.0017950
3: 0.0004450, 0.0050288, -0.0000877, 0.0048561, -0.0033518, 0.0039971
4: -0.0084428, -0.0044180, -0.0082912, -0.0039503, -0.0035096, 0.0029430
5: 0.0105403, 0.0120648, 0.0105977, 0.0122419, -0.0013294, 0.0011147
6: 0.0009528, 0.0067703, 0.0002768, 0.0065511, -0.0042538, 0.0050729
7: 0.9787260, 0.9827968, 0.9782529, 0.9826434, -0.0029766, 0.0035498
8: -0.0093733, -0.0050088, -0.0098805, -0.0051732, -0.0031914, 0.0038059
9: -0.0016910, 0.0011920, -0.0015824, 0.0015270, -0.0025140, 0.0021081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016492, upper bound: 0.0016927
time: 2.06 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016927
time: 2.10 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020466, -0.0004263, -0.0019689, -0.0004189, -0.0012965, 0.0011921
1: -0.0095040, -0.0053924, -0.0093069, -0.0053736, -0.0032901, 0.0030252
2: 0.0291337, 0.0316846, 0.0292560, 0.0316962, -0.0020412, 0.0018769
3: 0.0003994, 0.0051625, 0.0003776, 0.0049342, -0.0035046, 0.0038114
4: -0.0085602, -0.0043780, -0.0083597, -0.0043589, -0.0033465, 0.0030772
5: 0.0104958, 0.0120799, 0.0105717, 0.0120872, -0.0012676, 0.0011655
6: 0.0008950, 0.0069400, 0.0008674, 0.0066503, -0.0044478, 0.0048371
7: 0.9786856, 0.9829155, 0.9786662, 0.9827127, -0.0031123, 0.0033848
8: -0.0094167, -0.0048814, -0.0094374, -0.0050988, -0.0033369, 0.0036290
9: -0.0017751, 0.0012207, -0.0016315, 0.0012344, -0.0023972, 0.0022042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016927
time: 1.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016927
time: 1.55 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020466, -0.0004263, -0.0019418, -0.0002694, -0.0014491, 0.0011704
1: -0.0095040, -0.0053924, -0.0092382, -0.0049942, -0.0036772, 0.0029702
2: 0.0291337, 0.0316846, 0.0292986, 0.0319316, -0.0022814, 0.0018427
3: 0.0003994, 0.0051625, -0.0000619, 0.0048546, -0.0034408, 0.0042599
4: -0.0085602, -0.0043780, -0.0082899, -0.0039729, -0.0037403, 0.0030212
5: 0.0104958, 0.0120799, 0.0105982, 0.0122333, -0.0014167, 0.0011443
6: 0.0008950, 0.0069400, 0.0003096, 0.0065493, -0.0043668, 0.0054063
7: 0.9786856, 0.9829155, 0.9782759, 0.9826422, -0.0030557, 0.0037831
8: -0.0094167, -0.0048814, -0.0098559, -0.0051746, -0.0032762, 0.0040561
9: -0.0017751, 0.0012207, -0.0015815, 0.0015108, -0.0026793, 0.0021641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016927
time: 1.94 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016927
time: 2.12 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0020354, -0.0004354, -0.0020422, -0.0004069, -0.0012331, 0.0012011
1: -0.0094756, -0.0054156, -0.0094930, -0.0053431, -0.0031293, 0.0030480
2: 0.0291513, 0.0316702, 0.0291405, 0.0317152, -0.0019414, 0.0018910
3: 0.0004263, 0.0051296, 0.0003423, 0.0051498, -0.0035310, 0.0036251
4: -0.0085313, -0.0044016, -0.0085490, -0.0043278, -0.0031830, 0.0031004
5: 0.0105068, 0.0120710, 0.0105000, 0.0120989, -0.0012056, 0.0011743
6: 0.0009291, 0.0068983, 0.0008225, 0.0069239, -0.0044813, 0.0046007
7: 0.9787094, 0.9828863, 0.9786348, 0.9829043, -0.0031358, 0.0032194
8: -0.0093911, -0.0049128, -0.0094711, -0.0048936, -0.0033621, 0.0034517
9: -0.0017544, 0.0012037, -0.0017671, 0.0012566, -0.0022800, 0.0022208

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B2_A1_B1_B1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017200, upper bound: 0.0017090
time: 1.88 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017200, upper bound: 0.0017049
time: 1.53 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0020354, -0.0004354, -0.0020190, -0.0002570, -0.0013861, 0.0011813
1: -0.0094756, -0.0054156, -0.0094340, -0.0049629, -0.0035175, 0.0029978
2: 0.0291513, 0.0316702, 0.0291771, 0.0319510, -0.0021823, 0.0018599
3: 0.0004263, 0.0051296, -0.0000982, 0.0050814, -0.0034728, 0.0040748
4: -0.0085313, -0.0044016, -0.0084890, -0.0039411, -0.0035779, 0.0030493
5: 0.0105068, 0.0120710, 0.0105228, 0.0122454, -0.0013552, 0.0011550
6: 0.0009291, 0.0068983, 0.0002635, 0.0068371, -0.0044075, 0.0051715
7: 0.9787094, 0.9828863, 0.9782437, 0.9828436, -0.0030841, 0.0036188
8: -0.0093911, -0.0049128, -0.0098904, -0.0049586, -0.0033067, 0.0038799
9: -0.0017544, 0.0012037, -0.0017241, 0.0015336, -0.0025629, 0.0021843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0017049
time: 2.07 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017200, upper bound: 0.0017049
time: 2.26 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0020832, -0.0004199, -0.0020418, -0.0004149, -0.0013282, 0.0012354
1: -0.0095969, -0.0053762, -0.0094918, -0.0053634, -0.0033705, 0.0031351
2: 0.0290761, 0.0316946, 0.0291413, 0.0317025, -0.0020911, 0.0019450
3: 0.0003806, 0.0052701, 0.0003658, 0.0051484, -0.0036319, 0.0039046
4: -0.0086547, -0.0043615, -0.0085478, -0.0043485, -0.0034284, 0.0031889
5: 0.0104600, 0.0120862, 0.0105005, 0.0120911, -0.0012986, 0.0012079
6: 0.0008712, 0.0070766, 0.0008524, 0.0069221, -0.0046093, 0.0049554
7: 0.9786689, 0.9830111, 0.9786557, 0.9829031, -0.0032254, 0.0034675
8: -0.0094346, -0.0047790, -0.0094486, -0.0048949, -0.0034581, 0.0037178
9: -0.0018428, 0.0012325, -0.0017663, 0.0012418, -0.0024558, 0.0022843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017253, upper bound: 0.0017049
time: 1.99 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017215, upper bound: 0.0017049
time: 1.86 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0020832, -0.0004199, -0.0020185, -0.0002656, -0.0014759, 0.0012156
1: -0.0095969, -0.0053762, -0.0094328, -0.0049845, -0.0037453, 0.0030849
2: 0.0290761, 0.0316946, 0.0291779, 0.0319376, -0.0023236, 0.0019139
3: 0.0003806, 0.0052701, -0.0000731, 0.0050800, -0.0035737, 0.0043388
4: -0.0086547, -0.0043615, -0.0084878, -0.0039631, -0.0038096, 0.0031378
5: 0.0104600, 0.0120862, 0.0105232, 0.0122371, -0.0014430, 0.0011885
6: 0.0008712, 0.0070766, 0.0002953, 0.0068353, -0.0045355, 0.0055065
7: 0.9786689, 0.9830111, 0.9782659, 0.9828423, -0.0031737, 0.0038532
8: -0.0094346, -0.0047790, -0.0098666, -0.0049600, -0.0034027, 0.0041312
9: -0.0018428, 0.0012325, -0.0017232, 0.0015178, -0.0027289, 0.0022477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 50

## Relational analysis of IS_A2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017253, upper bound: 0.0017049
time: 2.04 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0017215, upper bound: 0.0017048
time: 2.03 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019762, -0.0002795, -0.0019749, -0.0004795, -0.0011318, 0.0013464
1: -0.0093256, -0.0050198, -0.0093221, -0.0055274, -0.0028720, 0.0034168
2: 0.0292444, 0.0319157, 0.0292466, 0.0316008, -0.0017818, 0.0021198
3: -0.0000323, 0.0049558, 0.0005558, 0.0049518, -0.0039582, 0.0033271
4: -0.0083787, -0.0039990, -0.0083752, -0.0045153, -0.0029213, 0.0034754
5: 0.0105646, 0.0122235, 0.0105659, 0.0120279, -0.0011065, 0.0013164
6: 0.0003472, 0.0066777, 0.0010934, 0.0066726, -0.0050234, 0.0042225
7: 0.9783022, 0.9827320, 0.9788244, 0.9827284, -0.0035151, 0.0029547
8: -0.0098277, -0.0050783, -0.0092678, -0.0050821, -0.0037688, 0.0031679
9: -0.0016451, 0.0014921, -0.0016426, 0.0011223, -0.0020926, 0.0024895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017426
time: 1.89 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017426
time: 1.52 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019758, -0.0002922, -0.0019838, -0.0004941, -0.0011342, 0.0013614
1: -0.0093244, -0.0050521, -0.0093447, -0.0055645, -0.0028783, 0.0034547
2: 0.0292452, 0.0318957, 0.0292325, 0.0315778, -0.0017857, 0.0021433
3: 0.0000052, 0.0049544, 0.0005988, 0.0049780, -0.0040021, 0.0033344
4: -0.0083775, -0.0040319, -0.0083982, -0.0045531, -0.0029277, 0.0035140
5: 0.0105650, 0.0122110, 0.0105572, 0.0120136, -0.0011089, 0.0013310
6: 0.0003947, 0.0066759, 0.0011481, 0.0067058, -0.0050791, 0.0042317
7: 0.9783355, 0.9827307, 0.9788626, 0.9827517, -0.0035541, 0.0029612
8: -0.0097920, -0.0050796, -0.0092268, -0.0050571, -0.0038106, 0.0031748
9: -0.0016442, 0.0014686, -0.0016591, 0.0010952, -0.0020972, 0.0025171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017389
time: 1.87 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017389
time: 2.12 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0019758, -0.0002881, -0.0020203, -0.0004639, -0.0011458, 0.0014237
1: -0.0093243, -0.0050418, -0.0094374, -0.0054878, -0.0029076, 0.0036128
2: 0.0292452, 0.0319021, 0.0291750, 0.0316254, -0.0018039, 0.0022414
3: -0.0000068, 0.0049544, 0.0005099, 0.0050853, -0.0041853, 0.0033683
4: -0.0083774, -0.0040214, -0.0084924, -0.0044750, -0.0029575, 0.0036749
5: 0.0105650, 0.0122150, 0.0105215, 0.0120432, -0.0011202, 0.0013919
6: 0.0003795, 0.0066759, 0.0010353, 0.0068421, -0.0053117, 0.0042748
7: 0.9783249, 0.9827308, 0.9787837, 0.9828470, -0.0037169, 0.0029913
8: -0.0098034, -0.0050796, -0.0093114, -0.0049549, -0.0039851, 0.0032071
9: -0.0016442, 0.0014761, -0.0017266, 0.0011511, -0.0021185, 0.0026324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A2_B1_B2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017431
time: 1.77 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017431
time: 2.04 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0019753, -0.0003006, -0.0020296, -0.0004785, -0.0011498, 0.0014349
1: -0.0093231, -0.0050734, -0.0094610, -0.0055249, -0.0029177, 0.0036413
2: 0.0292459, 0.0318825, 0.0291604, 0.0316023, -0.0018102, 0.0022591
3: 0.0000298, 0.0049530, 0.0005529, 0.0051127, -0.0042183, 0.0033801
4: -0.0083762, -0.0040535, -0.0085165, -0.0045128, -0.0029678, 0.0037038
5: 0.0105655, 0.0122028, 0.0105124, 0.0120289, -0.0011241, 0.0014029
6: 0.0004260, 0.0066741, 0.0010899, 0.0068768, -0.0053536, 0.0042897
7: 0.9783573, 0.9827295, 0.9788219, 0.9828714, -0.0037462, 0.0030017
8: -0.0097686, -0.0050810, -0.0092705, -0.0049288, -0.0040165, 0.0032183
9: -0.0016433, 0.0014531, -0.0017438, 0.0011241, -0.0021259, 0.0026531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A2_B1_B2_B2_B1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017390
time: 1.76 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017390
time: 1.62 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0020116, -0.0002733, -0.0020491, -0.0004786, -0.0011611, 0.0013866
1: -0.0094154, -0.0050042, -0.0095104, -0.0055250, -0.0029465, 0.0035187
2: 0.0291887, 0.0319254, 0.0291297, 0.0316023, -0.0018280, 0.0021830
3: -0.0000503, 0.0050598, 0.0005530, 0.0051699, -0.0040763, 0.0034134
4: -0.0084700, -0.0039832, -0.0085667, -0.0045129, -0.0029971, 0.0035792
5: 0.0105300, 0.0122295, 0.0104934, 0.0120288, -0.0011352, 0.0013557
6: 0.0003243, 0.0068097, 0.0010900, 0.0069494, -0.0051733, 0.0043320
7: 0.9782863, 0.9828244, 0.9788221, 0.9829221, -0.0036201, 0.0030313
8: -0.0098448, -0.0049792, -0.0092704, -0.0048744, -0.0038813, 0.0032501
9: -0.0017105, 0.0015035, -0.0017798, 0.0011240, -0.0021469, 0.0025638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016657
time: 1.87 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017623
time: 2.04 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0020112, -0.0002862, -0.0020610, -0.0004913, -0.0011627, 0.0014003
1: -0.0094142, -0.0050367, -0.0095407, -0.0055574, -0.0029505, 0.0035535
2: 0.0291894, 0.0319052, 0.0291109, 0.0315822, -0.0018305, 0.0022046
3: -0.0000126, 0.0050585, 0.0005905, 0.0052050, -0.0041165, 0.0034180
4: -0.0084688, -0.0040162, -0.0085975, -0.0045458, -0.0030012, 0.0036145
5: 0.0105304, 0.0122169, 0.0104817, 0.0120164, -0.0011368, 0.0013691
6: 0.0003721, 0.0068079, 0.0011376, 0.0069939, -0.0052244, 0.0043379
7: 0.9783196, 0.9828232, 0.9788553, 0.9829533, -0.0036558, 0.0030355
8: -0.0098090, -0.0049805, -0.0092347, -0.0048410, -0.0039196, 0.0032545
9: -0.0017097, 0.0014798, -0.0018018, 0.0011004, -0.0021498, 0.0025891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016522
time: 2.04 seconds

## Relational analysis of IS_A2_A2_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017581
time: 2.06 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0020111, -0.0002821, -0.0020984, -0.0004629, -0.0011778, 0.0014638
1: -0.0094141, -0.0050265, -0.0096355, -0.0054852, -0.0029887, 0.0037147
2: 0.0291895, 0.0319116, 0.0290521, 0.0316270, -0.0018542, 0.0023046
3: -0.0000245, 0.0050584, 0.0005070, 0.0053148, -0.0043033, 0.0034623
4: -0.0084688, -0.0040058, -0.0086939, -0.0044724, -0.0030401, 0.0037785
5: 0.0105304, 0.0122209, 0.0104452, 0.0120441, -0.0011515, 0.0014312
6: 0.0003570, 0.0068079, 0.0010315, 0.0071333, -0.0054614, 0.0043941
7: 0.9783092, 0.9828231, 0.9787810, 0.9830508, -0.0038217, 0.0030748
8: -0.0098203, -0.0049806, -0.0093143, -0.0047364, -0.0040974, 0.0032967
9: -0.0017096, 0.0014873, -0.0018709, 0.0011530, -0.0021776, 0.0027066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016661
time: 2.09 seconds

## Relational analysis of IS_A2_A2_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017619
time: 2.09 seconds

## BFS IS instance: IS_A2_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0020107, -0.0002946, -0.0021082, -0.0004765, -0.0011811, 0.0014741
1: -0.0094129, -0.0050582, -0.0096603, -0.0055197, -0.0029973, 0.0037408
2: 0.0291902, 0.0318919, 0.0290367, 0.0316056, -0.0018595, 0.0023208
3: 0.0000122, 0.0050570, 0.0005469, 0.0053436, -0.0043336, 0.0034722
4: -0.0084675, -0.0040380, -0.0087192, -0.0045075, -0.0030487, 0.0038051
5: 0.0105309, 0.0122087, 0.0104356, 0.0120309, -0.0011548, 0.0014413
6: 0.0004036, 0.0068061, 0.0010822, 0.0071699, -0.0054999, 0.0044066
7: 0.9783417, 0.9828219, 0.9788166, 0.9830764, -0.0038486, 0.0030836
8: -0.0097853, -0.0049819, -0.0092762, -0.0047090, -0.0041263, 0.0033061
9: -0.0017088, 0.0014642, -0.0018890, 0.0011279, -0.0021838, 0.0027256

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A2_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016547
time: 2.11 seconds

## Relational analysis of IS_A2_A2_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017579
time: 2.06 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.51 seconds
IS_A1_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016371, upper bound: 0.0016637
IS_A1_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016631
IS_A1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016492, upper bound: 0.0016631
IS_A1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016631
IS_A1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016631
IS_A1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016631
IS_A1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016631
IS_A1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016631
IS_A1_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017198, upper bound: 0.0016728
IS_A1_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017198, upper bound: 0.0016732
IS_A1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017227, upper bound: 0.0016732
IS_A1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017198, upper bound: 0.0016732
IS_A1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017252, upper bound: 0.0016732
IS_A1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017214, upper bound: 0.0016732
IS_A1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017252, upper bound: 0.0016732
IS_A1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017214, upper bound: 0.0016732
IS_A1_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017089
IS_A1_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017089
IS_A1_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017070
IS_A1_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017070
IS_A1_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017153
IS_A1_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017153
IS_A1_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017138
IS_A1_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017138
IS_A1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016281
IS_A1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0017319
IS_A1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016226
IS_A1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0017314
IS_A1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016281
IS_A1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017319
IS_A1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016226
IS_A1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017314
IS_A2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016371, upper bound: 0.0016965
IS_A2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016927
IS_A2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016492, upper bound: 0.0016927
IS_A2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016372, upper bound: 0.0016927
IS_A2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016927
IS_A2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016927
IS_A2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016522, upper bound: 0.0016927
IS_A2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016418, upper bound: 0.0016927
IS_A2_A1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017200, upper bound: 0.0017090
IS_A2_A1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017200, upper bound: 0.0017049
IS_A2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017229, upper bound: 0.0017049
IS_A2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017200, upper bound: 0.0017049
IS_A2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017253, upper bound: 0.0017049
IS_A2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017215, upper bound: 0.0017049
IS_A2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017253, upper bound: 0.0017049
IS_A2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0017215, upper bound: 0.0017048
IS_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017426
IS_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017426
IS_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017389
IS_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017389
IS_A2_A2_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017431
IS_A2_A2_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017431
IS_A2_A2_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017390
IS_A2_A2_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017390
IS_A2_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016657
IS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017623
IS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016522
IS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017581
IS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016661
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017619
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016547
IS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 5.51
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017579

## BFS IS instance: IS_A1_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019619, -0.0004841, -0.0019507, -0.0004533, -0.0011591, 0.0011031
1: -0.0092891, -0.0055390, -0.0092609, -0.0054608, -0.0029413, 0.0027994
2: 0.0292670, 0.0315936, 0.0292845, 0.0316421, -0.0018248, 0.0017367
3: 0.0005692, 0.0049136, 0.0004786, 0.0048808, -0.0032429, 0.0034073
4: -0.0083416, -0.0045271, -0.0083129, -0.0044476, -0.0029918, 0.0028474
5: 0.0105786, 0.0120234, 0.0105895, 0.0120536, -0.0011332, 0.0010785
6: 0.0011106, 0.0066241, 0.0009956, 0.0065825, -0.0041157, 0.0043243
7: 0.9788364, 0.9826945, 0.9787560, 0.9826654, -0.0028800, 0.0030260
8: -0.0092550, -0.0051185, -0.0093412, -0.0051496, -0.0030878, 0.0032443
9: -0.0016186, 0.0011138, -0.0015980, 0.0011708, -0.0021431, 0.0020397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016572, upper bound: 0.0016800
time: 2.27 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016575, upper bound: 0.0016907
time: 1.50 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019614, -0.0004951, -0.0019608, -0.0004679, -0.0011620, 0.0011270
1: -0.0092878, -0.0055670, -0.0092863, -0.0054979, -0.0029488, 0.0028599
2: 0.0292678, 0.0315763, 0.0292688, 0.0316191, -0.0018294, 0.0017743
3: 0.0006016, 0.0049121, 0.0005216, 0.0049103, -0.0033131, 0.0034160
4: -0.0083403, -0.0045556, -0.0083387, -0.0044853, -0.0029994, 0.0029090
5: 0.0105791, 0.0120127, 0.0105797, 0.0120393, -0.0011361, 0.0011019
6: 0.0011517, 0.0066222, 0.0010501, 0.0066199, -0.0042047, 0.0043354
7: 0.9788651, 0.9826931, 0.9787942, 0.9826916, -0.0029423, 0.0030337
8: -0.0092241, -0.0051199, -0.0093003, -0.0051216, -0.0031546, 0.0032526
9: -0.0016176, 0.0010934, -0.0016165, 0.0011438, -0.0021485, 0.0020838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016573, upper bound: 0.0016793
time: 2.35 seconds

## Relational analysis of IS_A1_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016576, upper bound: 0.0016930
time: 1.46 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019601, -0.0005133, -0.0019258, -0.0002770, -0.0013397, 0.0010572
1: -0.0092845, -0.0056131, -0.0091975, -0.0050134, -0.0033996, 0.0026828
2: 0.0292699, 0.0315476, 0.0293238, 0.0319197, -0.0021092, 0.0016644
3: 0.0006551, 0.0049082, -0.0000396, 0.0048075, -0.0031079, 0.0039383
4: -0.0083369, -0.0046025, -0.0082485, -0.0039925, -0.0034580, 0.0027289
5: 0.0105804, 0.0119949, 0.0106139, 0.0122259, -0.0013098, 0.0010336
6: 0.0012195, 0.0066173, 0.0003379, 0.0064894, -0.0039443, 0.0049982
7: 0.9789127, 0.9826897, 0.9782956, 0.9826002, -0.0027600, 0.0034975
8: -0.0091732, -0.0051235, -0.0098347, -0.0052195, -0.0029592, 0.0037499
9: -0.0016152, 0.0010598, -0.0015518, 0.0014967, -0.0024770, 0.0019547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016358, upper bound: 0.0016410
time: 1.80 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016516, upper bound: 0.0016435
time: 1.34 seconds

## BFS IS instance: IS_A1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019719, -0.0005238, -0.0019253, -0.0002902, -0.0013600, 0.0010597
1: -0.0093145, -0.0056397, -0.0091963, -0.0050469, -0.0034513, 0.0026890
2: 0.0292513, 0.0315311, 0.0293246, 0.0318989, -0.0021412, 0.0016683
3: 0.0006859, 0.0049430, -0.0000008, 0.0048061, -0.0031151, 0.0039981
4: -0.0083674, -0.0046296, -0.0082472, -0.0040266, -0.0035105, 0.0027352
5: 0.0105688, 0.0119846, 0.0106144, 0.0122130, -0.0013297, 0.0010360
6: 0.0012587, 0.0066614, 0.0003871, 0.0064876, -0.0039535, 0.0050742
7: 0.9789400, 0.9827206, 0.9783301, 0.9825990, -0.0027665, 0.0035506
8: -0.0091438, -0.0050905, -0.0097977, -0.0052209, -0.0029661, 0.0038069
9: -0.0016371, 0.0010404, -0.0015509, 0.0014724, -0.0025146, 0.0019593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016220, upper bound: 0.0016410
time: 1.98 seconds

## Relational analysis of IS_A1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016383, upper bound: 0.0016436
time: 1.39 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020109, -0.0004930, -0.0019521, -0.0004351, -0.0012768, 0.0011059
1: -0.0094135, -0.0055617, -0.0092644, -0.0054147, -0.0032401, 0.0028065
2: 0.0291899, 0.0315795, 0.0292823, 0.0316707, -0.0020102, 0.0017412
3: 0.0005955, 0.0050576, 0.0004253, 0.0048850, -0.0032512, 0.0037535
4: -0.0084681, -0.0045502, -0.0083165, -0.0044007, -0.0032957, 0.0028547
5: 0.0105307, 0.0120147, 0.0105881, 0.0120713, -0.0012483, 0.0010813
6: 0.0011439, 0.0068069, 0.0009279, 0.0065878, -0.0041262, 0.0047636
7: 0.9788597, 0.9828224, 0.9787085, 0.9826691, -0.0028873, 0.0033334
8: -0.0092299, -0.0049813, -0.0093920, -0.0051457, -0.0030956, 0.0035739
9: -0.0017092, 0.0010973, -0.0016006, 0.0012044, -0.0023608, 0.0020448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016708, upper bound: 0.0016793
time: 2.08 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016710, upper bound: 0.0016931
time: 1.97 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020233, -0.0005040, -0.0019516, -0.0004481, -0.0012937, 0.0010937
1: -0.0094450, -0.0055894, -0.0092631, -0.0054477, -0.0032829, 0.0027754
2: 0.0291703, 0.0315623, 0.0292832, 0.0316502, -0.0020367, 0.0017219
3: 0.0006277, 0.0050942, 0.0004635, 0.0048834, -0.0032152, 0.0038031
4: -0.0085002, -0.0045784, -0.0083152, -0.0044343, -0.0033393, 0.0028231
5: 0.0105185, 0.0120040, 0.0105886, 0.0120586, -0.0012648, 0.0010693
6: 0.0011847, 0.0068533, 0.0009763, 0.0065858, -0.0040805, 0.0048266
7: 0.9788883, 0.9828549, 0.9787425, 0.9826677, -0.0028553, 0.0033774
8: -0.0091993, -0.0049465, -0.0093557, -0.0051472, -0.0030614, 0.0036211
9: -0.0017322, 0.0010771, -0.0015996, 0.0011803, -0.0023920, 0.0020222

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016613, upper bound: 0.0016793
time: 1.44 seconds

## Relational analysis of IS_A1_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016615, upper bound: 0.0016931
time: 1.52 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020109, -0.0004930, -0.0019253, -0.0002860, -0.0014270, 0.0010876
1: -0.0094135, -0.0055617, -0.0091963, -0.0050364, -0.0036212, 0.0027599
2: 0.0291899, 0.0315795, 0.0293246, 0.0319054, -0.0022466, 0.0017122
3: 0.0005955, 0.0050576, -0.0000130, 0.0048061, -0.0031972, 0.0041950
4: -0.0084681, -0.0045502, -0.0082472, -0.0040159, -0.0036833, 0.0028072
5: 0.0105307, 0.0120147, 0.0106143, 0.0122171, -0.0013951, 0.0010633
6: 0.0011439, 0.0068069, 0.0003716, 0.0064877, -0.0040576, 0.0053239
7: 0.9788597, 0.9828224, 0.9783193, 0.9825991, -0.0028393, 0.0037254
8: -0.0092299, -0.0049813, -0.0098093, -0.0052208, -0.0030442, 0.0039943
9: -0.0017092, 0.0010973, -0.0015510, 0.0014800, -0.0026384, 0.0020109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016410, upper bound: 0.0016410
time: 1.46 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016517, upper bound: 0.0016435
time: 1.35 seconds

## BFS IS instance: IS_A1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020233, -0.0005040, -0.0019248, -0.0002989, -0.0014429, 0.0010771
1: -0.0094450, -0.0055894, -0.0091951, -0.0050691, -0.0036616, 0.0027333
2: 0.0291703, 0.0315623, 0.0293253, 0.0318851, -0.0022717, 0.0016957
3: 0.0006277, 0.0050942, 0.0000249, 0.0048047, -0.0031663, 0.0042418
4: -0.0085002, -0.0045784, -0.0082460, -0.0040492, -0.0037245, 0.0027802
5: 0.0105185, 0.0120040, 0.0106148, 0.0122045, -0.0014107, 0.0010531
6: 0.0011847, 0.0068533, 0.0004197, 0.0064859, -0.0040185, 0.0053834
7: 0.9788883, 0.9828549, 0.9783529, 0.9825978, -0.0028120, 0.0037671
8: -0.0091993, -0.0049465, -0.0097733, -0.0052222, -0.0030149, 0.0040389
9: -0.0017322, 0.0010771, -0.0015501, 0.0014562, -0.0026679, 0.0019915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016283, upper bound: 0.0016410
time: 1.76 seconds

## Relational analysis of IS_A1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016406, upper bound: 0.0016435
time: 2.07 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019967, -0.0004779, -0.0020242, -0.0004524, -0.0011855, 0.0011405
1: -0.0093775, -0.0055233, -0.0094473, -0.0054587, -0.0030083, 0.0028942
2: 0.0292122, 0.0316034, 0.0291689, 0.0316434, -0.0018664, 0.0017956
3: 0.0005510, 0.0050159, 0.0004762, 0.0050968, -0.0033528, 0.0034850
4: -0.0084315, -0.0045111, -0.0085025, -0.0044454, -0.0030599, 0.0029439
5: 0.0105446, 0.0120295, 0.0105177, 0.0120544, -0.0011590, 0.0011151
6: 0.0010875, 0.0067540, 0.0009925, 0.0068566, -0.0042551, 0.0044229
7: 0.9788203, 0.9827853, 0.9787539, 0.9828572, -0.0029775, 0.0030949
8: -0.0092723, -0.0050210, -0.0093435, -0.0049440, -0.0031924, 0.0033182
9: -0.0016829, 0.0011253, -0.0017338, 0.0011723, -0.0021919, 0.0021088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016482
time: 1.97 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017133
time: 1.62 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019962, -0.0004889, -0.0020374, -0.0004655, -0.0011878, 0.0011643
1: -0.0093762, -0.0055513, -0.0094807, -0.0054919, -0.0030142, 0.0029546
2: 0.0292130, 0.0315860, 0.0291482, 0.0316228, -0.0018700, 0.0018331
3: 0.0005835, 0.0050144, 0.0005147, 0.0051355, -0.0034228, 0.0034918
4: -0.0084302, -0.0045396, -0.0085365, -0.0044792, -0.0030659, 0.0030054
5: 0.0105451, 0.0120187, 0.0105048, 0.0120416, -0.0011613, 0.0011384
6: 0.0011287, 0.0067521, 0.0010413, 0.0069057, -0.0043440, 0.0044315
7: 0.9788491, 0.9827841, 0.9787880, 0.9828916, -0.0030397, 0.0031009
8: -0.0092414, -0.0050224, -0.0093069, -0.0049072, -0.0032591, 0.0033247
9: -0.0016820, 0.0011048, -0.0017581, 0.0011482, -0.0021962, 0.0021528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016453
time: 2.00 seconds

## Relational analysis of IS_A1_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017152
time: 1.91 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019949, -0.0005072, -0.0020041, -0.0002743, -0.0013683, 0.0010969
1: -0.0093729, -0.0055977, -0.0093963, -0.0050068, -0.0034722, 0.0027834
2: 0.0292150, 0.0315572, 0.0292005, 0.0319238, -0.0021541, 0.0017269
3: 0.0006373, 0.0050107, -0.0000473, 0.0050377, -0.0032245, 0.0040223
4: -0.0084269, -0.0045868, -0.0084506, -0.0039857, -0.0035318, 0.0028312
5: 0.0105463, 0.0120008, 0.0105373, 0.0122285, -0.0013377, 0.0010724
6: 0.0011969, 0.0067473, 0.0003281, 0.0067816, -0.0040923, 0.0051048
7: 0.9788968, 0.9827808, 0.9782888, 0.9828047, -0.0028636, 0.0035721
8: -0.0091902, -0.0050260, -0.0098420, -0.0050003, -0.0030702, 0.0038299
9: -0.0016796, 0.0010710, -0.0016966, 0.0015016, -0.0025299, 0.0020281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0015903
time: 2.13 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016732
time: 2.21 seconds

## BFS IS instance: IS_A1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020081, -0.0005178, -0.0020036, -0.0002871, -0.0013889, 0.0010999
1: -0.0094064, -0.0056247, -0.0093951, -0.0050392, -0.0035246, 0.0027912
2: 0.0291943, 0.0315404, 0.0292013, 0.0319037, -0.0021867, 0.0017317
3: 0.0006685, 0.0050494, -0.0000097, 0.0050363, -0.0032335, 0.0040831
4: -0.0084609, -0.0046143, -0.0084494, -0.0040188, -0.0035851, 0.0028391
5: 0.0105334, 0.0119904, 0.0105378, 0.0122160, -0.0013579, 0.0010754
6: 0.0012365, 0.0067965, 0.0003758, 0.0067799, -0.0041037, 0.0051820
7: 0.9789246, 0.9828152, 0.9783222, 0.9828035, -0.0028716, 0.0036261
8: -0.0091604, -0.0049891, -0.0098062, -0.0050016, -0.0030788, 0.0038878
9: -0.0017040, 0.0010514, -0.0016958, 0.0014780, -0.0025681, 0.0020337

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015903
time: 2.01 seconds

## Relational analysis of IS_A1_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016732
time: 2.03 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020479, -0.0004869, -0.0020256, -0.0004327, -0.0013053, 0.0011458
1: -0.0095073, -0.0055462, -0.0094507, -0.0054085, -0.0033125, 0.0029077
2: 0.0291316, 0.0315891, 0.0291668, 0.0316745, -0.0020551, 0.0018039
3: 0.0005776, 0.0051664, 0.0004181, 0.0051008, -0.0033684, 0.0038374
4: -0.0085636, -0.0045345, -0.0085060, -0.0043944, -0.0033694, 0.0029576
5: 0.0104945, 0.0120206, 0.0105163, 0.0120737, -0.0012762, 0.0011203
6: 0.0011212, 0.0069449, 0.0009188, 0.0068617, -0.0042749, 0.0048701
7: 0.9788439, 0.9829190, 0.9787021, 0.9828607, -0.0029914, 0.0034079
8: -0.0092470, -0.0048778, -0.0093989, -0.0049402, -0.0032072, 0.0036538
9: -0.0017776, 0.0011086, -0.0017363, 0.0012089, -0.0024135, 0.0021186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016900, upper bound: 0.0016453
time: 1.39 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016900, upper bound: 0.0017152
time: 1.98 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020607, -0.0004980, -0.0020250, -0.0004448, -0.0013234, 0.0011299
1: -0.0095399, -0.0055742, -0.0094494, -0.0054392, -0.0033584, 0.0028672
2: 0.0291114, 0.0315718, 0.0291676, 0.0316555, -0.0020835, 0.0017788
3: 0.0006101, 0.0052041, 0.0004536, 0.0050993, -0.0033215, 0.0038905
4: -0.0085967, -0.0045629, -0.0085046, -0.0044256, -0.0034160, 0.0029164
5: 0.0104820, 0.0120099, 0.0105169, 0.0120619, -0.0012939, 0.0011047
6: 0.0011624, 0.0069927, 0.0009638, 0.0068597, -0.0042154, 0.0049375
7: 0.9788727, 0.9829524, 0.9787338, 0.9828594, -0.0029498, 0.0034551
8: -0.0092161, -0.0048419, -0.0093650, -0.0049417, -0.0031626, 0.0037044
9: -0.0018013, 0.0010882, -0.0017353, 0.0011865, -0.0024469, 0.0020891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016453
time: 2.09 seconds

## Relational analysis of IS_A1_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017152
time: 2.24 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020479, -0.0004869, -0.0020036, -0.0002832, -0.0014573, 0.0011285
1: -0.0095073, -0.0055462, -0.0093951, -0.0050291, -0.0036981, 0.0028637
2: 0.0291316, 0.0315891, 0.0292013, 0.0319099, -0.0022943, 0.0017766
3: 0.0005776, 0.0051664, -0.0000214, 0.0050364, -0.0033174, 0.0042840
4: -0.0085636, -0.0045345, -0.0084494, -0.0040085, -0.0037615, 0.0029128
5: 0.0104945, 0.0120206, 0.0105378, 0.0122199, -0.0014248, 0.0011033
6: 0.0011212, 0.0069449, 0.0003609, 0.0067799, -0.0042102, 0.0054370
7: 0.9788439, 0.9829190, 0.9783118, 0.9828035, -0.0029461, 0.0038045
8: -0.0092470, -0.0048778, -0.0098174, -0.0050016, -0.0031587, 0.0040791
9: -0.0017776, 0.0011086, -0.0016958, 0.0014853, -0.0026944, 0.0020865

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0015903
time: 2.01 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0016732
time: 1.96 seconds

## BFS IS instance: IS_A1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020607, -0.0004980, -0.0020032, -0.0002956, -0.0014738, 0.0011141
1: -0.0095399, -0.0055742, -0.0093939, -0.0050608, -0.0037400, 0.0028273
2: 0.0291114, 0.0315718, 0.0292020, 0.0318903, -0.0023203, 0.0017541
3: 0.0006101, 0.0052041, 0.0000152, 0.0050350, -0.0032753, 0.0043326
4: -0.0085967, -0.0045629, -0.0084482, -0.0040407, -0.0038042, 0.0028758
5: 0.0104820, 0.0120099, 0.0105382, 0.0122077, -0.0014409, 0.0010893
6: 0.0011624, 0.0069927, 0.0004074, 0.0067781, -0.0041567, 0.0054986
7: 0.9788727, 0.9829524, 0.9783444, 0.9828023, -0.0029087, 0.0038476
8: -0.0092161, -0.0048419, -0.0097825, -0.0050029, -0.0031186, 0.0041253
9: -0.0018013, 0.0010882, -0.0016949, 0.0014623, -0.0027250, 0.0020600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015903
time: 2.07 seconds

## Relational analysis of IS_A1_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016732
time: 2.04 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019410, -0.0003043, -0.0019486, -0.0004810, -0.0011061, 0.0012991
1: -0.0092362, -0.0050829, -0.0092554, -0.0055313, -0.0028069, 0.0032967
2: 0.0292998, 0.0318766, 0.0292879, 0.0315984, -0.0017414, 0.0020453
3: 0.0000409, 0.0048523, 0.0005603, 0.0048745, -0.0038191, 0.0032517
4: -0.0082878, -0.0040632, -0.0083073, -0.0045193, -0.0028551, 0.0033533
5: 0.0105990, 0.0121992, 0.0105916, 0.0120264, -0.0010814, 0.0012701
6: 0.0004400, 0.0065463, 0.0010992, 0.0065745, -0.0048469, 0.0041268
7: 0.9783671, 0.9826400, 0.9788285, 0.9826598, -0.0033916, 0.0028877
8: -0.0097581, -0.0051768, -0.0092635, -0.0051557, -0.0036363, 0.0030961
9: -0.0015800, 0.0014462, -0.0015940, 0.0011194, -0.0020451, 0.0024020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016721
time: 1.31 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0016916
time: 1.84 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019410, -0.0003043, -0.0019216, -0.0003291, -0.0011717, 0.0011932
1: -0.0092362, -0.0050829, -0.0091869, -0.0051458, -0.0029734, 0.0030278
2: 0.0292998, 0.0318766, 0.0293304, 0.0318376, -0.0018447, 0.0018785
3: 0.0000409, 0.0048523, 0.0001137, 0.0047952, -0.0035076, 0.0034445
4: -0.0082878, -0.0040632, -0.0082376, -0.0041271, -0.0030245, 0.0030798
5: 0.0105990, 0.0121992, 0.0106180, 0.0121749, -0.0011456, 0.0011665
6: 0.0004400, 0.0065463, 0.0005324, 0.0064738, -0.0044516, 0.0043716
7: 0.9783671, 0.9826400, 0.9784318, 0.9825894, -0.0031150, 0.0030590
8: -0.0097581, -0.0051768, -0.0096887, -0.0052312, -0.0033398, 0.0032797
9: -0.0015800, 0.0014462, -0.0015441, 0.0014003, -0.0021665, 0.0022061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016721
time: 1.88 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0016916
time: 2.05 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0019406, -0.0003161, -0.0019586, -0.0004956, -0.0011089, 0.0013208
1: -0.0092350, -0.0051127, -0.0092808, -0.0055683, -0.0028141, 0.0033517
2: 0.0293006, 0.0318581, 0.0292722, 0.0315754, -0.0017459, 0.0020794
3: 0.0000754, 0.0048509, 0.0006032, 0.0049040, -0.0038828, 0.0032600
4: -0.0082866, -0.0040935, -0.0083332, -0.0045569, -0.0028624, 0.0034093
5: 0.0105994, 0.0121877, 0.0105818, 0.0120121, -0.0010842, 0.0012913
6: 0.0004838, 0.0065445, 0.0011536, 0.0066119, -0.0049278, 0.0041373
7: 0.9783977, 0.9826388, 0.9788665, 0.9826859, -0.0034482, 0.0028951
8: -0.0097252, -0.0051781, -0.0092227, -0.0051276, -0.0036971, 0.0031040
9: -0.0015791, 0.0014245, -0.0016125, 0.0010925, -0.0020504, 0.0024421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016680
time: 1.52 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016904
time: 1.57 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0019406, -0.0003161, -0.0019326, -0.0003455, -0.0011745, 0.0012155
1: -0.0092350, -0.0051127, -0.0092148, -0.0051873, -0.0029805, 0.0030846
2: 0.0293006, 0.0318581, 0.0293131, 0.0318118, -0.0018491, 0.0019137
3: 0.0000754, 0.0048509, 0.0001618, 0.0048275, -0.0035734, 0.0034528
4: -0.0082866, -0.0040935, -0.0082660, -0.0041694, -0.0030317, 0.0031376
5: 0.0105994, 0.0121877, 0.0106072, 0.0121589, -0.0011483, 0.0011884
6: 0.0004838, 0.0065445, 0.0005935, 0.0065148, -0.0045351, 0.0043820
7: 0.9783977, 0.9826388, 0.9784746, 0.9826180, -0.0031734, 0.0030663
8: -0.0097252, -0.0051781, -0.0096429, -0.0052005, -0.0034024, 0.0032876
9: -0.0015791, 0.0014245, -0.0015644, 0.0013701, -0.0021716, 0.0022475

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016680
time: 2.30 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016904
time: 2.08 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019406, -0.0003116, -0.0019940, -0.0004654, -0.0011227, 0.0013852
1: -0.0092351, -0.0051012, -0.0093707, -0.0054916, -0.0028489, 0.0035151
2: 0.0293005, 0.0318652, 0.0292164, 0.0316230, -0.0017675, 0.0021808
3: 0.0000621, 0.0048510, 0.0005143, 0.0050081, -0.0040720, 0.0033004
4: -0.0082867, -0.0040818, -0.0084246, -0.0044789, -0.0028979, 0.0035754
5: 0.0105994, 0.0121921, 0.0105472, 0.0120417, -0.0010976, 0.0013543
6: 0.0004670, 0.0065447, 0.0010409, 0.0067440, -0.0051679, 0.0041886
7: 0.9783860, 0.9826389, 0.9787877, 0.9827784, -0.0036163, 0.0029310
8: -0.0097378, -0.0051781, -0.0093072, -0.0050285, -0.0038772, 0.0031425
9: -0.0015792, 0.0014328, -0.0016780, 0.0011484, -0.0020758, 0.0025611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016796
time: 1.70 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016980
time: 1.39 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019406, -0.0003116, -0.0019712, -0.0003168, -0.0011881, 0.0012851
1: -0.0092351, -0.0051012, -0.0093128, -0.0051146, -0.0030148, 0.0032612
2: 0.0293005, 0.0318652, 0.0292523, 0.0318569, -0.0018704, 0.0020233
3: 0.0000621, 0.0048510, 0.0000776, 0.0049410, -0.0037780, 0.0034926
4: -0.0082867, -0.0040818, -0.0083657, -0.0040954, -0.0030666, 0.0033172
5: 0.0105994, 0.0121921, 0.0105695, 0.0121869, -0.0011615, 0.0012565
6: 0.0004670, 0.0065447, 0.0004866, 0.0066589, -0.0047947, 0.0044325
7: 0.9783860, 0.9826389, 0.9783998, 0.9827189, -0.0033551, 0.0031017
8: -0.0097378, -0.0051781, -0.0097231, -0.0050923, -0.0035972, 0.0033255
9: -0.0015792, 0.0014328, -0.0016358, 0.0014230, -0.0021967, 0.0023762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016796
time: 2.18 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016980
time: 2.17 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0019401, -0.0003232, -0.0020055, -0.0004801, -0.0011266, 0.0014024
1: -0.0092339, -0.0051308, -0.0093999, -0.0055288, -0.0028589, 0.0035589
2: 0.0293013, 0.0318469, 0.0291983, 0.0315999, -0.0017737, 0.0022079
3: 0.0000963, 0.0048496, 0.0005575, 0.0050419, -0.0041228, 0.0033119
4: -0.0082854, -0.0041119, -0.0084543, -0.0045168, -0.0029079, 0.0036200
5: 0.0105999, 0.0121807, 0.0105359, 0.0120273, -0.0011015, 0.0013711
6: 0.0005104, 0.0065429, 0.0010956, 0.0067870, -0.0052323, 0.0042032
7: 0.9784163, 0.9826377, 0.9788260, 0.9828084, -0.0036613, 0.0029412
8: -0.0097053, -0.0051794, -0.0092662, -0.0049963, -0.0039255, 0.0031534
9: -0.0015783, 0.0014113, -0.0016993, 0.0011212, -0.0020830, 0.0025930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016768
time: 1.70 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016971
time: 1.89 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0019401, -0.0003232, -0.0019813, -0.0003327, -0.0011912, 0.0013030
1: -0.0092339, -0.0051308, -0.0093385, -0.0051548, -0.0030229, 0.0033065
2: 0.0293013, 0.0318469, 0.0292364, 0.0318320, -0.0018754, 0.0020514
3: 0.0000963, 0.0048496, 0.0001242, 0.0049708, -0.0038305, 0.0035019
4: -0.0082854, -0.0041119, -0.0083919, -0.0041363, -0.0030748, 0.0033633
5: 0.0105999, 0.0121807, 0.0105596, 0.0121714, -0.0011646, 0.0012739
6: 0.0005104, 0.0065429, 0.0005457, 0.0066967, -0.0048614, 0.0044443
7: 0.9784163, 0.9826377, 0.9784411, 0.9827453, -0.0034018, 0.0031099
8: -0.0097053, -0.0051794, -0.0096787, -0.0050640, -0.0036472, 0.0033343
9: -0.0015783, 0.0014113, -0.0016546, 0.0013937, -0.0022025, 0.0024092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016768
time: 2.09 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016971
time: 2.38 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0018792, -0.0003341, -0.0020370, -0.0004530, -0.0011276, 0.0013446
1: -0.0090794, -0.0051584, -0.0094797, -0.0054601, -0.0028613, 0.0034122
2: 0.0293971, 0.0318297, 0.0291488, 0.0316426, -0.0017752, 0.0021169
3: 0.0001284, 0.0046706, 0.0004778, 0.0051343, -0.0039529, 0.0033147
4: -0.0081283, -0.0041400, -0.0085355, -0.0044468, -0.0029105, 0.0034708
5: 0.0106594, 0.0121701, 0.0105052, 0.0120538, -0.0011024, 0.0013146
6: 0.0005510, 0.0063157, 0.0009945, 0.0069043, -0.0050167, 0.0042068
7: 0.9784449, 0.9824787, 0.9787552, 0.9828905, -0.0035105, 0.0029437
8: -0.0096747, -0.0053498, -0.0093420, -0.0049083, -0.0037638, 0.0031561
9: -0.0014657, 0.0013911, -0.0017574, 0.0011713, -0.0020848, 0.0024862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016281
time: 2.16 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016281
time: 2.07 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0019577, -0.0003301, -0.0020370, -0.0004530, -0.0011616, 0.0013111
1: -0.0092785, -0.0051483, -0.0094797, -0.0054601, -0.0029478, 0.0033270
2: 0.0292736, 0.0318360, 0.0291488, 0.0316426, -0.0018288, 0.0020641
3: 0.0001166, 0.0049012, 0.0004778, 0.0051343, -0.0038542, 0.0034149
4: -0.0083308, -0.0041297, -0.0085355, -0.0044468, -0.0029984, 0.0033841
5: 0.0105827, 0.0121740, 0.0105052, 0.0120538, -0.0011357, 0.0012818
6: 0.0005361, 0.0066084, 0.0009945, 0.0069043, -0.0048915, 0.0043339
7: 0.9784344, 0.9826836, 0.9787552, 0.9828905, -0.0034228, 0.0030327
8: -0.0096859, -0.0051302, -0.0093420, -0.0049083, -0.0036698, 0.0032515
9: -0.0016108, 0.0013985, -0.0017574, 0.0011713, -0.0021478, 0.0024241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017319
time: 2.04 seconds

## Relational analysis of IS_A1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017319
time: 2.13 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0018788, -0.0003457, -0.0020494, -0.0004661, -0.0011301, 0.0013653
1: -0.0090782, -0.0051878, -0.0095113, -0.0054933, -0.0028678, 0.0034646
2: 0.0293979, 0.0318115, 0.0291292, 0.0316219, -0.0017792, 0.0021494
3: 0.0001625, 0.0046692, 0.0005163, 0.0051710, -0.0040135, 0.0033222
4: -0.0081271, -0.0041699, -0.0085676, -0.0044807, -0.0029170, 0.0035240
5: 0.0106599, 0.0121587, 0.0104930, 0.0120410, -0.0011049, 0.0013348
6: 0.0005943, 0.0063140, 0.0010434, 0.0069508, -0.0050937, 0.0042163
7: 0.9784752, 0.9824775, 0.9787894, 0.9829232, -0.0035643, 0.0029503
8: -0.0096423, -0.0053511, -0.0093053, -0.0048734, -0.0038215, 0.0031632
9: -0.0014649, 0.0013697, -0.0017805, 0.0011471, -0.0020895, 0.0025243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016226
time: 2.21 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016226
time: 2.07 seconds

## BFS IS instance: IS_A1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0019572, -0.0003419, -0.0020494, -0.0004661, -0.0011642, 0.0013309
1: -0.0092773, -0.0051783, -0.0095113, -0.0054933, -0.0029542, 0.0033772
2: 0.0292743, 0.0318174, 0.0291292, 0.0316219, -0.0018328, 0.0020953
3: 0.0001514, 0.0048999, 0.0005163, 0.0051710, -0.0039124, 0.0034223
4: -0.0083296, -0.0041602, -0.0085676, -0.0044807, -0.0030049, 0.0034352
5: 0.0105832, 0.0121624, 0.0104930, 0.0120410, -0.0011382, 0.0013012
6: 0.0005802, 0.0066067, 0.0010434, 0.0069508, -0.0049653, 0.0043434
7: 0.9784653, 0.9826823, 0.9787894, 0.9829232, -0.0034745, 0.0030393
8: -0.0096528, -0.0051315, -0.0093053, -0.0048734, -0.0037252, 0.0032586
9: -0.0016100, 0.0013766, -0.0017805, 0.0011471, -0.0021525, 0.0024607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017314
time: 2.32 seconds

## Relational analysis of IS_A1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017314
time: 2.23 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019306, -0.0003154, -0.0020366, -0.0004610, -0.0012202, 0.0013832
1: -0.0092099, -0.0051109, -0.0094786, -0.0054804, -0.0030964, 0.0035100
2: 0.0293162, 0.0318592, 0.0291494, 0.0316300, -0.0019210, 0.0021776
3: 0.0000733, 0.0048218, 0.0005014, 0.0051331, -0.0040662, 0.0035871
4: -0.0082610, -0.0040916, -0.0085344, -0.0044675, -0.0031496, 0.0035703
5: 0.0106091, 0.0121884, 0.0105056, 0.0120460, -0.0011930, 0.0013523
6: 0.0004811, 0.0065076, 0.0010244, 0.0069027, -0.0051605, 0.0045525
7: 0.9783959, 0.9826129, 0.9787761, 0.9828894, -0.0036111, 0.0031856
8: -0.0097272, -0.0052059, -0.0093196, -0.0049094, -0.0038716, 0.0034155
9: -0.0015608, 0.0014258, -0.0017566, 0.0011565, -0.0022561, 0.0025574

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016281
time: 1.67 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016281
time: 2.25 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020097, -0.0003125, -0.0020366, -0.0004610, -0.0012551, 0.0013490
1: -0.0094105, -0.0051036, -0.0094786, -0.0054804, -0.0031849, 0.0034233
2: 0.0291917, 0.0318637, 0.0291494, 0.0316300, -0.0019759, 0.0021238
3: 0.0000649, 0.0050542, 0.0005014, 0.0051331, -0.0039657, 0.0036896
4: -0.0084651, -0.0040843, -0.0085344, -0.0044675, -0.0032396, 0.0034821
5: 0.0105318, 0.0121912, 0.0105056, 0.0120460, -0.0012271, 0.0013189
6: 0.0004705, 0.0068026, 0.0010244, 0.0069027, -0.0050330, 0.0046825
7: 0.9783885, 0.9828194, 0.9787761, 0.9828894, -0.0035219, 0.0032766
8: -0.0097352, -0.0049846, -0.0093196, -0.0049094, -0.0037760, 0.0035130
9: -0.0017070, 0.0014310, -0.0017566, 0.0011565, -0.0023206, 0.0024943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017319
time: 2.16 seconds

## Relational analysis of IS_A1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017319
time: 2.24 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019302, -0.0003268, -0.0020490, -0.0004748, -0.0012227, 0.0014040
1: -0.0092087, -0.0051399, -0.0095102, -0.0055154, -0.0031027, 0.0035629
2: 0.0293169, 0.0318412, 0.0291298, 0.0316083, -0.0019249, 0.0022104
3: 0.0001069, 0.0048204, 0.0005419, 0.0051698, -0.0041274, 0.0035944
4: -0.0082598, -0.0041212, -0.0085665, -0.0045031, -0.0031560, 0.0036240
5: 0.0106096, 0.0121772, 0.0104934, 0.0120325, -0.0011954, 0.0013727
6: 0.0005238, 0.0065058, 0.0010758, 0.0069492, -0.0052382, 0.0045617
7: 0.9784258, 0.9826118, 0.9788121, 0.9829220, -0.0036655, 0.0031921
8: -0.0096952, -0.0052072, -0.0092810, -0.0048746, -0.0039299, 0.0034224
9: -0.0015600, 0.0014046, -0.0017797, 0.0011310, -0.0022607, 0.0025959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016226
time: 1.94 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016226
time: 2.23 seconds

## BFS IS instance: IS_A1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020092, -0.0003242, -0.0020490, -0.0004748, -0.0012576, 0.0013692
1: -0.0094093, -0.0051333, -0.0095102, -0.0055154, -0.0031913, 0.0034745
2: 0.0291924, 0.0318453, 0.0291298, 0.0316083, -0.0019799, 0.0021556
3: 0.0000993, 0.0050528, 0.0005419, 0.0051698, -0.0040250, 0.0036970
4: -0.0084639, -0.0041145, -0.0085665, -0.0045031, -0.0032461, 0.0035341
5: 0.0105323, 0.0121797, 0.0104934, 0.0120325, -0.0012295, 0.0013386
6: 0.0005141, 0.0068008, 0.0010758, 0.0069492, -0.0051083, 0.0046920
7: 0.9784191, 0.9828182, 0.9788121, 0.9829220, -0.0035745, 0.0032832
8: -0.0097024, -0.0049859, -0.0092810, -0.0048746, -0.0038325, 0.0035201
9: -0.0017061, 0.0014094, -0.0017797, 0.0011310, -0.0023253, 0.0025316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017314
time: 1.38 seconds

## Relational analysis of IS_A1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017314
time: 2.13 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0020001, -0.0004576, -0.0019666, -0.0004515, -0.0011523, 0.0011401
1: -0.0093861, -0.0054718, -0.0093010, -0.0054564, -0.0029240, 0.0028933
2: 0.0292068, 0.0316353, 0.0292596, 0.0316449, -0.0018141, 0.0017950
3: 0.0004914, 0.0050259, 0.0004735, 0.0049274, -0.0033517, 0.0033874
4: -0.0084403, -0.0044588, -0.0083537, -0.0044431, -0.0029742, 0.0029429
5: 0.0105412, 0.0120493, 0.0105740, 0.0120553, -0.0011266, 0.0011147
6: 0.0010118, 0.0067667, 0.0009891, 0.0066416, -0.0042537, 0.0042990
7: 0.9787673, 0.9827942, 0.9787514, 0.9827068, -0.0029766, 0.0030082
8: -0.0093291, -0.0050115, -0.0093461, -0.0051053, -0.0031914, 0.0032253
9: -0.0016892, 0.0011628, -0.0016272, 0.0011740, -0.0021305, 0.0021081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016572, upper bound: 0.0017072
time: 1.85 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016576, upper bound: 0.0017257
time: 1.94 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019996, -0.0004693, -0.0019758, -0.0004661, -0.0011546, 0.0011578
1: -0.0093848, -0.0055015, -0.0093245, -0.0054934, -0.0029300, 0.0029381
2: 0.0292077, 0.0316168, 0.0292450, 0.0316219, -0.0018178, 0.0018228
3: 0.0005259, 0.0050244, 0.0005164, 0.0049546, -0.0034037, 0.0033943
4: -0.0084389, -0.0044890, -0.0083776, -0.0044807, -0.0029803, 0.0029886
5: 0.0105417, 0.0120379, 0.0105650, 0.0120410, -0.0011289, 0.0011320
6: 0.0010555, 0.0067647, 0.0010435, 0.0066762, -0.0043197, 0.0043078
7: 0.9787979, 0.9827928, 0.9787894, 0.9827309, -0.0030227, 0.0030144
8: -0.0092963, -0.0050130, -0.0093053, -0.0050794, -0.0032409, 0.0032319
9: -0.0016883, 0.0011411, -0.0016444, 0.0011470, -0.0021348, 0.0021408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016573, upper bound: 0.0017021
time: 1.51 seconds

## Relational analysis of IS_A2_A1_B1_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016577, upper bound: 0.0017223
time: 2.03 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0019983, -0.0004858, -0.0019414, -0.0002750, -0.0013387, 0.0010929
1: -0.0093815, -0.0055433, -0.0092371, -0.0050085, -0.0033971, 0.0027733
2: 0.0292097, 0.0315909, 0.0292993, 0.0319227, -0.0021076, 0.0017206
3: 0.0005742, 0.0050206, -0.0000453, 0.0048534, -0.0032128, 0.0039354
4: -0.0084356, -0.0045315, -0.0082888, -0.0039875, -0.0034555, 0.0028210
5: 0.0105430, 0.0120218, 0.0105986, 0.0122278, -0.0013088, 0.0010685
6: 0.0011169, 0.0067599, 0.0003306, 0.0065477, -0.0040774, 0.0049946
7: 0.9788409, 0.9827895, 0.9782906, 0.9826410, -0.0028532, 0.0034950
8: -0.0092502, -0.0050166, -0.0098401, -0.0051758, -0.0030591, 0.0037471
9: -0.0016859, 0.0011107, -0.0015807, 0.0015003, -0.0024752, 0.0020207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016358, upper bound: 0.0016739
time: 1.72 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016516, upper bound: 0.0016754
time: 1.82 seconds

## BFS IS instance: IS_A2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020076, -0.0004984, -0.0019409, -0.0002883, -0.0013616, 0.0010955
1: -0.0094052, -0.0055752, -0.0092359, -0.0050421, -0.0034552, 0.0027799
2: 0.0291950, 0.0315711, 0.0293000, 0.0319019, -0.0021436, 0.0017246
3: 0.0006112, 0.0050480, -0.0000064, 0.0048520, -0.0032204, 0.0040027
4: -0.0084597, -0.0045640, -0.0082875, -0.0040217, -0.0035145, 0.0028276
5: 0.0105339, 0.0120095, 0.0105991, 0.0122149, -0.0013312, 0.0010710
6: 0.0011639, 0.0067947, 0.0003800, 0.0065459, -0.0040871, 0.0050800
7: 0.9788737, 0.9828139, 0.9783252, 0.9826398, -0.0028599, 0.0035547
8: -0.0092150, -0.0049904, -0.0098030, -0.0051771, -0.0030663, 0.0038112
9: -0.0017031, 0.0010874, -0.0015798, 0.0014759, -0.0025175, 0.0020255

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016220, upper bound: 0.0016739
time: 1.42 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016383, upper bound: 0.0016755
time: 1.74 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020438, -0.0004696, -0.0019679, -0.0004333, -0.0012741, 0.0011275
1: -0.0094969, -0.0055022, -0.0093045, -0.0054101, -0.0032333, 0.0028612
2: 0.0291381, 0.0316164, 0.0292575, 0.0316736, -0.0020060, 0.0017751
3: 0.0005266, 0.0051543, 0.0004200, 0.0049314, -0.0033145, 0.0037456
4: -0.0085530, -0.0044897, -0.0083572, -0.0043960, -0.0032888, 0.0029103
5: 0.0104986, 0.0120376, 0.0105727, 0.0120731, -0.0012457, 0.0011023
6: 0.0010565, 0.0069296, 0.0009211, 0.0066467, -0.0042066, 0.0047537
7: 0.9787985, 0.9829082, 0.9787038, 0.9827103, -0.0029436, 0.0033264
8: -0.0092956, -0.0048893, -0.0093971, -0.0051015, -0.0031560, 0.0035664
9: -0.0017699, 0.0011406, -0.0016297, 0.0012077, -0.0023558, 0.0020847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016708, upper bound: 0.0017021
time: 1.36 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016710, upper bound: 0.0017224
time: 1.88 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020540, -0.0004830, -0.0019674, -0.0004463, -0.0012937, 0.0011317
1: -0.0095229, -0.0055363, -0.0093032, -0.0054432, -0.0032831, 0.0028719
2: 0.0291220, 0.0315953, 0.0292583, 0.0316530, -0.0020368, 0.0017817
3: 0.0005661, 0.0051844, 0.0004583, 0.0049299, -0.0033269, 0.0038033
4: -0.0085794, -0.0045243, -0.0083559, -0.0044297, -0.0033394, 0.0029212
5: 0.0104885, 0.0120245, 0.0105732, 0.0120603, -0.0012649, 0.0011065
6: 0.0011065, 0.0069678, 0.0009697, 0.0066448, -0.0042223, 0.0048269
7: 0.9788336, 0.9829350, 0.9787378, 0.9827089, -0.0029545, 0.0033776
8: -0.0092580, -0.0048606, -0.0093606, -0.0051030, -0.0031677, 0.0036213
9: -0.0017889, 0.0011158, -0.0016288, 0.0011836, -0.0023921, 0.0020925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016612, upper bound: 0.0017021
time: 1.44 seconds

## Relational analysis of IS_A2_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016616, upper bound: 0.0017224
time: 1.81 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020438, -0.0004696, -0.0019409, -0.0002841, -0.0014276, 0.0011077
1: -0.0094969, -0.0055022, -0.0092359, -0.0050314, -0.0036227, 0.0028108
2: 0.0291381, 0.0316164, 0.0293000, 0.0319085, -0.0022476, 0.0017438
3: 0.0005266, 0.0051543, -0.0000188, 0.0048519, -0.0032562, 0.0041968
4: -0.0085530, -0.0044897, -0.0082875, -0.0040108, -0.0036849, 0.0028591
5: 0.0104986, 0.0120376, 0.0105991, 0.0122190, -0.0013958, 0.0010829
6: 0.0010565, 0.0069296, 0.0003643, 0.0065458, -0.0041326, 0.0053262
7: 0.9787985, 0.9829082, 0.9783142, 0.9826397, -0.0028918, 0.0037270
8: -0.0092956, -0.0048893, -0.0098148, -0.0051772, -0.0031004, 0.0039960
9: -0.0017699, 0.0011406, -0.0015798, 0.0014837, -0.0026396, 0.0020480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016411, upper bound: 0.0016739
time: 1.43 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016518, upper bound: 0.0016754
time: 1.78 seconds

## BFS IS instance: IS_A2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020540, -0.0004830, -0.0019404, -0.0002970, -0.0014458, 0.0011118
1: -0.0095229, -0.0055363, -0.0092347, -0.0050642, -0.0036688, 0.0028213
2: 0.0291220, 0.0315953, 0.0293008, 0.0318882, -0.0022762, 0.0017503
3: 0.0005661, 0.0051844, 0.0000192, 0.0048505, -0.0032683, 0.0042502
4: -0.0085794, -0.0045243, -0.0082862, -0.0040441, -0.0037318, 0.0028697
5: 0.0104885, 0.0120245, 0.0105996, 0.0122064, -0.0014135, 0.0010870
6: 0.0011065, 0.0069678, 0.0004125, 0.0065440, -0.0041479, 0.0053940
7: 0.9788336, 0.9829350, 0.9783480, 0.9826384, -0.0029025, 0.0037745
8: -0.0092580, -0.0048606, -0.0097787, -0.0051785, -0.0031120, 0.0040468
9: -0.0017889, 0.0011158, -0.0015789, 0.0014598, -0.0026732, 0.0020556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 254

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016283, upper bound: 0.0016738
time: 1.69 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016406, upper bound: 0.0016754
time: 2.02 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0020344, -0.0004513, -0.0020394, -0.0004503, -0.0011824, 0.0011806
1: -0.0094732, -0.0054557, -0.0094859, -0.0054532, -0.0030005, 0.0029959
2: 0.0291528, 0.0316453, 0.0291449, 0.0316468, -0.0018615, 0.0018587
3: 0.0004727, 0.0051268, 0.0004699, 0.0051416, -0.0034706, 0.0034759
4: -0.0085288, -0.0044424, -0.0085418, -0.0044399, -0.0030520, 0.0030473
5: 0.0105077, 0.0120555, 0.0105028, 0.0120565, -0.0011560, 0.0011542
6: 0.0009881, 0.0068947, 0.0009845, 0.0069135, -0.0044047, 0.0044114
7: 0.9787507, 0.9828838, 0.9787482, 0.9828970, -0.0030822, 0.0030869
8: -0.0093468, -0.0049154, -0.0093495, -0.0049014, -0.0033046, 0.0033096
9: -0.0017527, 0.0011745, -0.0017620, 0.0011763, -0.0021862, 0.0021829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016901
time: 2.00 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017463
time: 1.89 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0020339, -0.0004630, -0.0020522, -0.0004633, -0.0011838, 0.0011971
1: -0.0094718, -0.0054856, -0.0095183, -0.0054863, -0.0030040, 0.0030379
2: 0.0291537, 0.0316268, 0.0291248, 0.0316263, -0.0018637, 0.0018847
3: 0.0005073, 0.0051253, 0.0005082, 0.0051791, -0.0035192, 0.0034800
4: -0.0085275, -0.0044728, -0.0085747, -0.0044735, -0.0030556, 0.0030900
5: 0.0105082, 0.0120440, 0.0104903, 0.0120437, -0.0011574, 0.0011704
6: 0.0010320, 0.0068927, 0.0010331, 0.0069610, -0.0044664, 0.0044166
7: 0.9787814, 0.9828824, 0.9787821, 0.9829302, -0.0031253, 0.0030905
8: -0.0093139, -0.0049169, -0.0093131, -0.0048657, -0.0033509, 0.0033135
9: -0.0017517, 0.0011528, -0.0017855, 0.0011522, -0.0021888, 0.0022134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016804
time: 2.22 seconds

## Relational analysis of IS_A2_A1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017429
time: 2.12 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020326, -0.0004795, -0.0020181, -0.0002721, -0.0013644, 0.0011369
1: -0.0094686, -0.0055275, -0.0094318, -0.0050012, -0.0034624, 0.0028850
2: 0.0291557, 0.0316007, 0.0291785, 0.0319273, -0.0021481, 0.0017898
3: 0.0005559, 0.0051215, -0.0000538, 0.0050788, -0.0033421, 0.0040110
4: -0.0085242, -0.0045154, -0.0084867, -0.0039801, -0.0035219, 0.0029345
5: 0.0105094, 0.0120279, 0.0105236, 0.0122306, -0.0013340, 0.0011115
6: 0.0010937, 0.0068880, 0.0003198, 0.0068338, -0.0042416, 0.0050905
7: 0.9788246, 0.9828791, 0.9782830, 0.9828413, -0.0029680, 0.0035621
8: -0.0092676, -0.0049205, -0.0098482, -0.0049611, -0.0031822, 0.0038191
9: -0.0017493, 0.0011222, -0.0017225, 0.0015057, -0.0025228, 0.0021020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016247
time: 1.91 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0017049
time: 1.97 seconds

## BFS IS instance: IS_A2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020440, -0.0004923, -0.0020176, -0.0002849, -0.0013873, 0.0011397
1: -0.0094974, -0.0055599, -0.0094306, -0.0050337, -0.0035205, 0.0028922
2: 0.0291378, 0.0315807, 0.0291793, 0.0319071, -0.0021841, 0.0017943
3: 0.0005934, 0.0051549, -0.0000162, 0.0050774, -0.0033504, 0.0040783
4: -0.0085535, -0.0045483, -0.0084855, -0.0040131, -0.0035810, 0.0029418
5: 0.0104984, 0.0120154, 0.0105241, 0.0122181, -0.0013564, 0.0011143
6: 0.0011412, 0.0069303, 0.0003676, 0.0068320, -0.0042521, 0.0051760
7: 0.9788578, 0.9829088, 0.9783165, 0.9828399, -0.0029754, 0.0036219
8: -0.0092320, -0.0048887, -0.0098123, -0.0049625, -0.0031901, 0.0038832
9: -0.0017703, 0.0010986, -0.0017216, 0.0014820, -0.0025651, 0.0021073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016247
time: 1.85 seconds

## Relational analysis of IS_A2_A1_B2_A1_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0017049
time: 1.89 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0020803, -0.0004634, -0.0020408, -0.0004305, -0.0013058, 0.0011692
1: -0.0095897, -0.0054865, -0.0094894, -0.0054030, -0.0033138, 0.0029670
2: 0.0290805, 0.0316262, 0.0291428, 0.0316780, -0.0020559, 0.0018408
3: 0.0005084, 0.0052618, 0.0004116, 0.0051456, -0.0034372, 0.0038388
4: -0.0086474, -0.0044737, -0.0085453, -0.0043887, -0.0033707, 0.0030180
5: 0.0104628, 0.0120437, 0.0105014, 0.0120758, -0.0012767, 0.0011431
6: 0.0010333, 0.0070661, 0.0009105, 0.0069185, -0.0043622, 0.0048720
7: 0.9787824, 0.9830038, 0.9786965, 0.9829005, -0.0030525, 0.0034092
8: -0.0093129, -0.0047869, -0.0094050, -0.0048976, -0.0032727, 0.0036552
9: -0.0018376, 0.0011521, -0.0017645, 0.0012129, -0.0024144, 0.0021618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016901, upper bound: 0.0016804
time: 2.03 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016901, upper bound: 0.0017429
time: 1.93 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0020903, -0.0004769, -0.0020403, -0.0004425, -0.0013252, 0.0011730
1: -0.0096151, -0.0055207, -0.0094880, -0.0054336, -0.0033629, 0.0029767
2: 0.0290648, 0.0316050, 0.0291436, 0.0316590, -0.0020863, 0.0018467
3: 0.0005480, 0.0052912, 0.0004471, 0.0051440, -0.0034483, 0.0038957
4: -0.0086732, -0.0045085, -0.0085440, -0.0044199, -0.0034206, 0.0030278
5: 0.0104530, 0.0120305, 0.0105020, 0.0120640, -0.0012956, 0.0011468
6: 0.0010836, 0.0071033, 0.0009556, 0.0069165, -0.0043764, 0.0049442
7: 0.9788176, 0.9830298, 0.9787279, 0.9828991, -0.0030624, 0.0034597
8: -0.0092752, -0.0047589, -0.0093712, -0.0048991, -0.0032833, 0.0037093
9: -0.0018560, 0.0011272, -0.0017635, 0.0011906, -0.0024502, 0.0021688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016804
time: 2.15 seconds

## Relational analysis of IS_A2_A1_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017429
time: 2.17 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0020803, -0.0004634, -0.0020176, -0.0002809, -0.0014541, 0.0011518
1: -0.0095897, -0.0054865, -0.0094305, -0.0050235, -0.0036900, 0.0029228
2: 0.0290805, 0.0316262, 0.0291793, 0.0319134, -0.0022893, 0.0018133
3: 0.0005084, 0.0052618, -0.0000279, 0.0050774, -0.0033860, 0.0042747
4: -0.0086474, -0.0044737, -0.0084855, -0.0040028, -0.0037534, 0.0029730
5: 0.0104628, 0.0120437, 0.0105241, 0.0122220, -0.0014217, 0.0011261
6: 0.0010333, 0.0070661, 0.0003527, 0.0068320, -0.0042972, 0.0054252
7: 0.9787824, 0.9830038, 0.9783061, 0.9828399, -0.0030070, 0.0037963
8: -0.0093129, -0.0047869, -0.0098236, -0.0049625, -0.0032240, 0.0040702
9: -0.0018376, 0.0011521, -0.0017216, 0.0014894, -0.0026886, 0.0021296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0016247
time: 1.35 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0017049
time: 2.77 seconds

## BFS IS instance: IS_A2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0020903, -0.0004769, -0.0020171, -0.0002934, -0.0014723, 0.0011555
1: -0.0096151, -0.0055207, -0.0094293, -0.0050552, -0.0037363, 0.0029323
2: 0.0290648, 0.0316050, 0.0291800, 0.0318938, -0.0023180, 0.0018192
3: 0.0005480, 0.0052912, 0.0000088, 0.0050760, -0.0033969, 0.0043283
4: -0.0086732, -0.0045085, -0.0084843, -0.0040350, -0.0038004, 0.0029827
5: 0.0104530, 0.0120305, 0.0105246, 0.0122098, -0.0014395, 0.0011297
6: 0.0010836, 0.0071033, 0.0003992, 0.0068303, -0.0043112, 0.0054932
7: 0.9788176, 0.9830298, 0.9783386, 0.9828388, -0.0030167, 0.0038439
8: -0.0092752, -0.0047589, -0.0097886, -0.0049638, -0.0032344, 0.0041212
9: -0.0018560, 0.0011272, -0.0017207, 0.0014663, -0.0027223, 0.0021365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 254
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016247
time: 2.04 seconds

## Relational analysis of IS_A2_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0017049
time: 2.13 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019762, -0.0002795, -0.0019644, -0.0004792, -0.0010971, 0.0013379
1: -0.0093256, -0.0050198, -0.0092955, -0.0055267, -0.0027839, 0.0033950
2: 0.0292444, 0.0319157, 0.0292630, 0.0316012, -0.0017272, 0.0021063
3: -0.0000323, 0.0049558, 0.0005550, 0.0049210, -0.0039330, 0.0032251
4: -0.0083787, -0.0039990, -0.0083481, -0.0045146, -0.0028317, 0.0034533
5: 0.0105646, 0.0122235, 0.0105761, 0.0120282, -0.0010726, 0.0013080
6: 0.0003472, 0.0066777, 0.0010925, 0.0066335, -0.0049914, 0.0040930
7: 0.9783022, 0.9827320, 0.9788237, 0.9827011, -0.0034928, 0.0028641
8: -0.0098277, -0.0050783, -0.0092685, -0.0051114, -0.0037448, 0.0030708
9: -0.0016451, 0.0014921, -0.0016232, 0.0011228, -0.0020284, 0.0024737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B1_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016965
time: 1.82 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017285
time: 1.82 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019762, -0.0002795, -0.0019372, -0.0003272, -0.0011621, 0.0012304
1: -0.0093256, -0.0050198, -0.0092266, -0.0051408, -0.0029490, 0.0031223
2: 0.0292444, 0.0319157, 0.0293058, 0.0318406, -0.0018295, 0.0019371
3: -0.0000323, 0.0049558, 0.0001080, 0.0048411, -0.0036170, 0.0034162
4: -0.0083787, -0.0039990, -0.0082780, -0.0041221, -0.0029996, 0.0031759
5: 0.0105646, 0.0122235, 0.0106027, 0.0121768, -0.0011362, 0.0012029
6: 0.0003472, 0.0066777, 0.0005252, 0.0065321, -0.0045904, 0.0043356
7: 0.9783022, 0.9827320, 0.9784268, 0.9826301, -0.0032122, 0.0030339
8: -0.0098277, -0.0050783, -0.0096942, -0.0051875, -0.0034439, 0.0032528
9: -0.0016451, 0.0014921, -0.0015730, 0.0014039, -0.0021487, 0.0022749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B1_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016965
time: 2.07 seconds

## Relational analysis of IS_A2_A2_B1_B1_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017285
time: 2.05 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0019758, -0.0002922, -0.0019737, -0.0004938, -0.0010994, 0.0013540
1: -0.0093244, -0.0050521, -0.0093191, -0.0055636, -0.0027899, 0.0034359
2: 0.0292452, 0.0318957, 0.0292484, 0.0315783, -0.0017309, 0.0021317
3: 0.0000052, 0.0049544, 0.0005978, 0.0049483, -0.0039803, 0.0032320
4: -0.0083775, -0.0040319, -0.0083721, -0.0045522, -0.0028378, 0.0034949
5: 0.0105650, 0.0122110, 0.0105671, 0.0120139, -0.0010749, 0.0013238
6: 0.0003947, 0.0066759, 0.0011468, 0.0066681, -0.0050516, 0.0041018
7: 0.9783355, 0.9827307, 0.9788617, 0.9827253, -0.0035348, 0.0028703
8: -0.0097920, -0.0050796, -0.0092278, -0.0050854, -0.0037899, 0.0030774
9: -0.0016442, 0.0014686, -0.0016404, 0.0010959, -0.0020328, 0.0025034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B1_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016905
time: 1.84 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017247
time: 1.52 seconds

## BFS IS instance: IS_A2_A2_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0019758, -0.0002922, -0.0019468, -0.0003436, -0.0011642, 0.0012471
1: -0.0093244, -0.0050521, -0.0092509, -0.0051825, -0.0029544, 0.0031646
2: 0.0292452, 0.0318957, 0.0292907, 0.0318148, -0.0018329, 0.0019633
3: 0.0000052, 0.0049544, 0.0001563, 0.0048693, -0.0036660, 0.0034225
4: -0.0083775, -0.0040319, -0.0083027, -0.0041645, -0.0030051, 0.0032189
5: 0.0105650, 0.0122110, 0.0105933, 0.0121608, -0.0011382, 0.0012192
6: 0.0003947, 0.0066759, 0.0005865, 0.0065678, -0.0046527, 0.0043436
7: 0.9783355, 0.9827307, 0.9784697, 0.9826551, -0.0032557, 0.0030394
8: -0.0097920, -0.0050796, -0.0096481, -0.0051607, -0.0034906, 0.0032587
9: -0.0016442, 0.0014686, -0.0015907, 0.0013736, -0.0021526, 0.0023058

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B1_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016905
time: 2.09 seconds

## Relational analysis of IS_A2_A2_B1_B1_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017247
time: 2.19 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0019758, -0.0002881, -0.0020083, -0.0004635, -0.0011138, 0.0014195
1: -0.0093243, -0.0050418, -0.0094069, -0.0054867, -0.0028263, 0.0036021
2: 0.0292452, 0.0319021, 0.0291940, 0.0316261, -0.0017535, 0.0022348
3: -0.0000068, 0.0049544, 0.0005086, 0.0050500, -0.0041729, 0.0032742
4: -0.0083774, -0.0040214, -0.0084614, -0.0044739, -0.0028749, 0.0036640
5: 0.0105650, 0.0122150, 0.0105332, 0.0120436, -0.0010889, 0.0013878
6: 0.0003795, 0.0066759, 0.0010336, 0.0067972, -0.0052960, 0.0041554
7: 0.9783249, 0.9827308, 0.9787825, 0.9828156, -0.0037059, 0.0029077
8: -0.0098034, -0.0050796, -0.0093127, -0.0049886, -0.0039733, 0.0031175
9: -0.0016442, 0.0014761, -0.0017043, 0.0011520, -0.0020593, 0.0026246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0017023
time: 1.61 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017280
time: 2.03 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0019758, -0.0002881, -0.0019844, -0.0003149, -0.0011773, 0.0013162
1: -0.0093243, -0.0050418, -0.0093464, -0.0051096, -0.0029876, 0.0033399
2: 0.0292452, 0.0319021, 0.0292315, 0.0318600, -0.0018535, 0.0020721
3: -0.0000068, 0.0049544, 0.0000718, 0.0049799, -0.0038692, 0.0034610
4: -0.0083774, -0.0040214, -0.0083998, -0.0040904, -0.0030389, 0.0033973
5: 0.0105650, 0.0122150, 0.0105565, 0.0121889, -0.0011511, 0.0012868
6: 0.0003795, 0.0066759, 0.0004793, 0.0067082, -0.0049105, 0.0043925
7: 0.9783249, 0.9827308, 0.9783947, 0.9827533, -0.0034361, 0.0030737
8: -0.0098034, -0.0050796, -0.0097286, -0.0050553, -0.0036841, 0.0032955
9: -0.0016442, 0.0014761, -0.0016603, 0.0014267, -0.0021768, 0.0024335

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0017023
time: 1.95 seconds

## Relational analysis of IS_A2_A2_B1_B2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017280
time: 1.53 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0019753, -0.0003006, -0.0020189, -0.0004779, -0.0011178, 0.0014329
1: -0.0093231, -0.0050734, -0.0094339, -0.0055234, -0.0028366, 0.0036362
2: 0.0292459, 0.0318825, 0.0291772, 0.0316033, -0.0017598, 0.0022559
3: 0.0000298, 0.0049530, 0.0005512, 0.0050813, -0.0042124, 0.0032861
4: -0.0083762, -0.0040535, -0.0084888, -0.0045113, -0.0028853, 0.0036987
5: 0.0105655, 0.0122028, 0.0105228, 0.0120294, -0.0010929, 0.0014010
6: 0.0004260, 0.0066741, 0.0010876, 0.0068369, -0.0053461, 0.0041705
7: 0.9783573, 0.9827295, 0.9788203, 0.9828433, -0.0037409, 0.0029183
8: -0.0097686, -0.0050810, -0.0092721, -0.0049588, -0.0040109, 0.0031289
9: -0.0016433, 0.0014531, -0.0017240, 0.0011252, -0.0020668, 0.0026494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 251

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016966
time: 1.45 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017238
time: 1.38 seconds

## BFS IS instance: IS_A2_A2_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0019753, -0.0003006, -0.0019944, -0.0003306, -0.0011811, 0.0013306
1: -0.0093231, -0.0050734, -0.0093716, -0.0051495, -0.0029971, 0.0033766
2: 0.0292459, 0.0318825, 0.0292159, 0.0318352, -0.0018594, 0.0020948
3: 0.0000298, 0.0049530, 0.0001181, 0.0050091, -0.0039116, 0.0034720
4: -0.0083762, -0.0040535, -0.0084255, -0.0041310, -0.0030485, 0.0034345
5: 0.0105655, 0.0122028, 0.0105468, 0.0121735, -0.0011547, 0.0013009
6: 0.0004260, 0.0066741, 0.0005380, 0.0067453, -0.0049643, 0.0044064
7: 0.9783573, 0.9827295, 0.9784358, 0.9827793, -0.0034738, 0.0030834
8: -0.0097686, -0.0050810, -0.0096845, -0.0050275, -0.0037244, 0.0033059
9: -0.0016433, 0.0014531, -0.0016786, 0.0013976, -0.0021837, 0.0024602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 221
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 221
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 233
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 112
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 254

## Relational analysis of IS_A2_A2_B1_B2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016966
time: 2.09 seconds

## Relational analysis of IS_A2_A2_B1_B2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0017238
time: 2.07 seconds

## BFS IS instance: IS_A2_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0019181, -0.0002803, -0.0020491, -0.0004786, -0.0010817, 0.0014154
1: -0.0091781, -0.0050219, -0.0095104, -0.0055250, -0.0027450, 0.0035919
2: 0.0293359, 0.0319144, 0.0291297, 0.0316023, -0.0017030, 0.0022284
3: -0.0000298, 0.0047850, 0.0005530, 0.0051699, -0.0041610, 0.0031800
4: -0.0082287, -0.0040011, -0.0085667, -0.0045129, -0.0027922, 0.0036536
5: 0.0106214, 0.0122227, 0.0104934, 0.0120288, -0.0010576, 0.0013839
6: 0.0003503, 0.0064609, 0.0010900, 0.0069494, -0.0052809, 0.0040358
7: 0.9783044, 0.9825803, 0.9788221, 0.9829221, -0.0036953, 0.0028241
8: -0.0098253, -0.0052409, -0.0092704, -0.0048744, -0.0039620, 0.0030279
9: -0.0015377, 0.0014906, -0.0017798, 0.0011240, -0.0020001, 0.0026171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 254
type: B, layer: 1, pos: 254
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 221
type: B, layer: 1, pos: 221
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 233
type: B, layer: 1, pos: 233
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 112
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 112
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_A2_B2_B1_B1_A1_B1

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016657
time: 1.40 seconds

## Relational analysis of IS_A2_A2_B2_B1_B1_A1_B2

### Relational analysis result of IS_A2_A2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016657
time: 2.12 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.89 seconds
IS_A1_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016572, upper bound: 0.0016800
IS_A1_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016575, upper bound: 0.0016907
IS_A1_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016573, upper bound: 0.0016793
IS_A1_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016576, upper bound: 0.0016930
IS_A1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016358, upper bound: 0.0016410
IS_A1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016516, upper bound: 0.0016435
IS_A1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016220, upper bound: 0.0016410
IS_A1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016383, upper bound: 0.0016436
IS_A1_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016708, upper bound: 0.0016793
IS_A1_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016710, upper bound: 0.0016931
IS_A1_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016613, upper bound: 0.0016793
IS_A1_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016615, upper bound: 0.0016931
IS_A1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016410, upper bound: 0.0016410
IS_A1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016517, upper bound: 0.0016435
IS_A1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016283, upper bound: 0.0016410
IS_A1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016406, upper bound: 0.0016435
IS_A1_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016482
IS_A1_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017133
IS_A1_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016453
IS_A1_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017152
IS_A1_A1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0015903
IS_A1_A1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016732
IS_A1_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0015903
IS_A1_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016732
IS_A1_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016900, upper bound: 0.0016453
IS_A1_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016900, upper bound: 0.0017152
IS_A1_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016453
IS_A1_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017152
IS_A1_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0015903
IS_A1_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0016732
IS_A1_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0015903
IS_A1_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016732
IS_A1_A2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016721
IS_A1_A2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0016916
IS_A1_A2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016721
IS_A1_A2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0016916
IS_A1_A2_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016680
IS_A1_A2_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016904
IS_A1_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016680
IS_A1_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016904
IS_A1_A2_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016796
IS_A1_A2_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016980
IS_A1_A2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016796
IS_A1_A2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016980
IS_A1_A2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016768
IS_A1_A2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016971
IS_A1_A2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016768
IS_A1_A2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0016971
IS_A1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016281
IS_A1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016281
IS_A1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017319
IS_A1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017319
IS_A1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016226
IS_A1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0016226
IS_A1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017314
IS_A1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016208, upper bound: 0.0017314
IS_A1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016281
IS_A1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016281
IS_A1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017319
IS_A1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017319
IS_A1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016226
IS_A1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016226
IS_A1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017314
IS_A1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0017314
IS_A2_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016572, upper bound: 0.0017072
IS_A2_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016576, upper bound: 0.0017257
IS_A2_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016573, upper bound: 0.0017021
IS_A2_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016577, upper bound: 0.0017223
IS_A2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016358, upper bound: 0.0016739
IS_A2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016516, upper bound: 0.0016754
IS_A2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016220, upper bound: 0.0016739
IS_A2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016383, upper bound: 0.0016755
IS_A2_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016708, upper bound: 0.0017021
IS_A2_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016710, upper bound: 0.0017224
IS_A2_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016612, upper bound: 0.0017021
IS_A2_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016616, upper bound: 0.0017224
IS_A2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016411, upper bound: 0.0016739
IS_A2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016518, upper bound: 0.0016754
IS_A2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016283, upper bound: 0.0016738
IS_A2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016406, upper bound: 0.0016754
IS_A2_A1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0016901
IS_A2_A1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016752, upper bound: 0.0017463
IS_A2_A1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0016804
IS_A2_A1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016753, upper bound: 0.0017429
IS_A2_A1_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0016247
IS_A2_A1_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016657, upper bound: 0.0017049
IS_A2_A1_B2_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0016247
IS_A2_A1_B2_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016523, upper bound: 0.0017049
IS_A2_A1_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016901, upper bound: 0.0016804
IS_A2_A1_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016901, upper bound: 0.0017429
IS_A2_A1_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0016804
IS_A2_A1_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016804, upper bound: 0.0017429
IS_A2_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0016247
IS_A2_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016665, upper bound: 0.0017049
IS_A2_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0016247
IS_A2_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016552, upper bound: 0.0017049
IS_A2_A2_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016965
IS_A2_A2_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017285
IS_A2_A2_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016965
IS_A2_A2_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017285
IS_A2_A2_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016905
IS_A2_A2_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017247
IS_A2_A2_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016905
IS_A2_A2_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017247
IS_A2_A2_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0017023
IS_A2_A2_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017280
IS_A2_A2_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0017023
IS_A2_A2_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017280
IS_A2_A2_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016966
IS_A2_A2_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016092, upper bound: 0.0017238
IS_A2_A2_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016083, upper bound: 0.0016966
IS_A2_A2_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016091, upper bound: 0.0017238
IS_A2_A2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016657
IS_A2_A2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -0.0016247, upper bound: 0.0016657
IS_A2_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017623
IS_A2_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016522
IS_A2_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017581
IS_A2_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016661
IS_A2_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017619
IS_A2_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0016547
IS_A2_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.89
Output dim: 7, lower bound: -0.0016547, upper bound: 0.0017579

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.05 + 597.21 = 601.25 seconds
