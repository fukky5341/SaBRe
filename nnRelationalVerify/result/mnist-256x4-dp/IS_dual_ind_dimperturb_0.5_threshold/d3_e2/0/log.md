## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.001618947


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0030711, 0.0009685, -0.0030711, 0.0009685, -0.0037671, 0.0037671)
1: (-0.0046474, -0.0032927, -0.0046474, -0.0032927, -0.0013293, 0.0013293)
2: (0.0107054, 0.0161666, 0.0107054, 0.0161666, -0.0050605, 0.0050605)
3: (1.0066155, 1.0100789, 1.0066155, 1.0100789, -0.0034634, 0.0034634)
4: (-0.0042776, -0.0033740, -0.0042776, -0.0033740, -0.0008332, 0.0008332)
5: (0.0016041, 0.0047204, 0.0016041, 0.0047204, -0.0029028, 0.0029028)
6: (-0.0026153, -0.0022953, -0.0026153, -0.0022953, -0.0003200, 0.0003200)
7: (-0.0131095, -0.0080398, -0.0131095, -0.0080398, -0.0050399, 0.0050399)
8: (-0.0140015, -0.0041438, -0.0140015, -0.0041438, -0.0090697, 0.0090697)
9: (-0.0021439, 0.0027784, -0.0021439, 0.0027784, -0.0045139, 0.0045139)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.54 + 2.11 = 3.65 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -0.0021372, upper bound: 0.0021372

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 146
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 146

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020821, upper bound: 0.0020956
time: 1.33 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020970, upper bound: 0.0020970
time: 1.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.85 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.85
Output dim: 3, lower bound: -0.0020821, upper bound: 0.0020956
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.85
Output dim: 3, lower bound: -0.0020970, upper bound: 0.0020970

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0029949, 0.0008332, -0.0030686, 0.0009545, -0.0036665, 0.0036353
1: -0.0046288, -0.0033592, -0.0046469, -0.0032997, -0.0012973, 0.0012665
2: 0.0107986, 0.0159586, 0.0107085, 0.0161450, -0.0049261, 0.0048589
3: 1.0067815, 1.0100325, 1.0066327, 1.0100777, -0.0032963, 0.0033998
4: -0.0042389, -0.0033881, -0.0042736, -0.0033744, -0.0007957, 0.0008112
5: 0.0016620, 0.0046139, 0.0016060, 0.0047093, -0.0028256, 0.0027991
6: -0.0026040, -0.0022991, -0.0026142, -0.0022954, -0.0003086, 0.0003151
7: -0.0130936, -0.0081725, -0.0131078, -0.0080453, -0.0050192, 0.0049033
8: -0.0135507, -0.0042911, -0.0139547, -0.0041482, -0.0086350, 0.0088274
9: -0.0020752, 0.0025375, -0.0021421, 0.0027534, -0.0043942, 0.0042822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020821, upper bound: 0.0020821
time: 1.62 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020821, upper bound: 0.0020955
time: 1.43 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0030605, 0.0008842, -0.0030705, 0.0009642, -0.0037529, 0.0036458
1: -0.0046454, -0.0033337, -0.0046473, -0.0032948, -0.0013260, 0.0012751
2: 0.0107182, 0.0160369, 0.0107061, 0.0161599, -0.0050428, 0.0048723
3: 1.0067124, 1.0100740, 1.0066206, 1.0100785, -0.0033661, 0.0034534
4: -0.0042535, -0.0033758, -0.0042764, -0.0033741, -0.0007983, 0.0008305
5: 0.0016121, 0.0046539, 0.0016045, 0.0047169, -0.0028919, 0.0028070
6: -0.0026094, -0.0022957, -0.0026150, -0.0022953, -0.0003141, 0.0003193
7: -0.0130996, -0.0080644, -0.0131090, -0.0080411, -0.0050256, 0.0050149
8: -0.0137203, -0.0041618, -0.0139869, -0.0041447, -0.0086613, 0.0090427
9: -0.0021365, 0.0026281, -0.0021435, 0.0027706, -0.0045020, 0.0042965

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 146
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 146

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020956, upper bound: 0.0020821
time: 1.46 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020956, upper bound: 0.0020970
time: 1.60 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.62 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 3, lower bound: -0.0020821, upper bound: 0.0020821
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 3, lower bound: -0.0020821, upper bound: 0.0020955
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 3, lower bound: -0.0020956, upper bound: 0.0020821
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.62
Output dim: 3, lower bound: -0.0020956, upper bound: 0.0020970

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029949, 0.0008332, -0.0029949, 0.0008332, -0.0035515, 0.0035515
1: -0.0046288, -0.0033592, -0.0046288, -0.0033592, -0.0012413, 0.0012413
2: 0.0107986, 0.0159586, 0.0107986, 0.0159586, -0.0047495, 0.0047495
3: 1.0067815, 1.0100325, 1.0067815, 1.0100325, -0.0032511, 0.0032511
4: -0.0042389, -0.0033881, -0.0042389, -0.0033881, -0.0007783, 0.0007783
5: 0.0016620, 0.0046139, 0.0016620, 0.0046139, -0.0027352, 0.0027352
6: -0.0026040, -0.0022991, -0.0026040, -0.0022991, -0.0003049, 0.0003049
7: -0.0130936, -0.0081725, -0.0130936, -0.0081725, -0.0048899, 0.0048899
8: -0.0135507, -0.0042911, -0.0135507, -0.0042911, -0.0084444, 0.0084444
9: -0.0020752, 0.0025375, -0.0020752, 0.0025375, -0.0041895, 0.0041895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020590, upper bound: 0.0019865
time: 1.47 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020612, upper bound: 0.0020614
time: 1.24 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029949, 0.0008332, -0.0030605, 0.0008842, -0.0036142, 0.0036271
1: -0.0046288, -0.0033592, -0.0046454, -0.0033337, -0.0012703, 0.0012654
2: 0.0107986, 0.0159586, 0.0107182, 0.0160369, -0.0048458, 0.0048495
3: 1.0067815, 1.0100325, 1.0067124, 1.0100740, -0.0032926, 0.0033201
4: -0.0042389, -0.0033881, -0.0042535, -0.0033758, -0.0007945, 0.0007962
5: 0.0016620, 0.0046139, 0.0016121, 0.0046539, -0.0027845, 0.0027929
6: -0.0026040, -0.0022991, -0.0026094, -0.0022957, -0.0003083, 0.0003103
7: -0.0130936, -0.0081725, -0.0130996, -0.0080644, -0.0050001, 0.0048972
8: -0.0135507, -0.0042911, -0.0137203, -0.0041618, -0.0086237, 0.0086533
9: -0.0020752, 0.0025375, -0.0021365, 0.0026281, -0.0043011, 0.0042781

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020590, upper bound: 0.0020026
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020612, upper bound: 0.0020729
time: 1.73 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0030605, 0.0008842, -0.0029949, 0.0008332, -0.0036271, 0.0036142
1: -0.0046454, -0.0033337, -0.0046288, -0.0033592, -0.0012654, 0.0012703
2: 0.0107182, 0.0160369, 0.0107986, 0.0159586, -0.0048495, 0.0048458
3: 1.0067124, 1.0100740, 1.0067815, 1.0100325, -0.0033201, 0.0032926
4: -0.0042535, -0.0033758, -0.0042389, -0.0033881, -0.0007962, 0.0007945
5: 0.0016121, 0.0046539, 0.0016620, 0.0046139, -0.0027929, 0.0027845
6: -0.0026094, -0.0022957, -0.0026040, -0.0022991, -0.0003103, 0.0003083
7: -0.0130996, -0.0080644, -0.0130936, -0.0081725, -0.0048972, 0.0050001
8: -0.0137203, -0.0041618, -0.0135507, -0.0042911, -0.0086533, 0.0086237
9: -0.0021365, 0.0026281, -0.0020752, 0.0025375, -0.0042781, 0.0043011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020703, upper bound: 0.0019848
time: 1.40 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020729, upper bound: 0.0020612
time: 1.35 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0030605, 0.0008842, -0.0030605, 0.0008842, -0.0036352, 0.0036352
1: -0.0046454, -0.0033337, -0.0046454, -0.0033337, -0.0012737, 0.0012737
2: 0.0107182, 0.0160369, 0.0107182, 0.0160369, -0.0048602, 0.0048602
3: 1.0067124, 1.0100740, 1.0067124, 1.0100740, -0.0033616, 0.0033616
4: -0.0042535, -0.0033758, -0.0042535, -0.0033758, -0.0007966, 0.0007966
5: 0.0016121, 0.0046539, 0.0016121, 0.0046539, -0.0027990, 0.0027990
6: -0.0026094, -0.0022957, -0.0026094, -0.0022957, -0.0003137, 0.0003137
7: -0.0130996, -0.0080644, -0.0130996, -0.0080644, -0.0050023, 0.0050023
8: -0.0137203, -0.0041618, -0.0137203, -0.0041618, -0.0086462, 0.0086462
9: -0.0021365, 0.0026281, -0.0021365, 0.0026281, -0.0042908, 0.0042908

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 200

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020703, upper bound: 0.0019864
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020729, upper bound: 0.0020650
time: 1.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.77 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020590, upper bound: 0.0019865
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020612, upper bound: 0.0020614
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020590, upper bound: 0.0020026
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020612, upper bound: 0.0020729
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020703, upper bound: 0.0019848
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020729, upper bound: 0.0020612
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020703, upper bound: 0.0019864
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.77
Output dim: 3, lower bound: -0.0020729, upper bound: 0.0020650

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027447, 0.0007975, -0.0029543, 0.0008301, -0.0032809, 0.0034435
1: -0.0045649, -0.0033772, -0.0046182, -0.0033609, -0.0011640, 0.0011853
2: 0.0111151, 0.0159036, 0.0108500, 0.0159538, -0.0043987, 0.0045931
3: 1.0068339, 1.0098733, 1.0067877, 1.0100063, -0.0031724, 0.0030856
4: -0.0042286, -0.0034365, -0.0042380, -0.0033959, -0.0007509, 0.0007231
5: 0.0018531, 0.0045857, 0.0016930, 0.0046114, -0.0025278, 0.0026510
6: -0.0025957, -0.0023122, -0.0026026, -0.0023013, -0.0002945, 0.0002903
7: -0.0130894, -0.0086144, -0.0130932, -0.0082447, -0.0048100, 0.0044458
8: -0.0134315, -0.0047916, -0.0135403, -0.0043728, -0.0081349, 0.0078641
9: -0.0018397, 0.0024738, -0.0020365, 0.0025319, -0.0039127, 0.0040272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019505
time: 1.67 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019394
time: 1.25 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029399, 0.0008272, -0.0029949, 0.0008332, -0.0033865, 0.0035276
1: -0.0046153, -0.0033624, -0.0046288, -0.0033592, -0.0011765, 0.0012216
2: 0.0108683, 0.0159493, 0.0107986, 0.0159586, -0.0045060, 0.0047126
3: 1.0067922, 1.0099990, 1.0067815, 1.0100325, -0.0032403, 0.0032176
4: -0.0042371, -0.0033987, -0.0042389, -0.0033881, -0.0007714, 0.0007346
5: 0.0017041, 0.0046091, 0.0016620, 0.0046139, -0.0026063, 0.0027163
6: -0.0026019, -0.0023019, -0.0026040, -0.0022991, -0.0003028, 0.0003021
7: -0.0130929, -0.0082728, -0.0130936, -0.0081725, -0.0048871, 0.0047780
8: -0.0135305, -0.0044004, -0.0135507, -0.0042911, -0.0083647, 0.0079490
9: -0.0020252, 0.0025267, -0.0020752, 0.0025375, -0.0039329, 0.0041469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019865, upper bound: 0.0020592
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019865, upper bound: 0.0020614
time: 1.63 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027447, 0.0007975, -0.0030218, 0.0008810, -0.0033436, 0.0035235
1: -0.0045649, -0.0033772, -0.0046354, -0.0033354, -0.0011930, 0.0012104
2: 0.0111151, 0.0159036, 0.0107670, 0.0160320, -0.0044951, 0.0047001
3: 1.0068339, 1.0098733, 1.0067186, 1.0100492, -0.0032153, 0.0031546
4: -0.0042286, -0.0034365, -0.0042525, -0.0033833, -0.0007681, 0.0007411
5: 0.0018531, 0.0045857, 0.0016416, 0.0046514, -0.0025771, 0.0027122
6: -0.0025957, -0.0023122, -0.0026080, -0.0022977, -0.0002980, 0.0002958
7: -0.0130894, -0.0086144, -0.0130992, -0.0081330, -0.0049235, 0.0044532
8: -0.0134315, -0.0047916, -0.0137097, -0.0042392, -0.0083224, 0.0080729
9: -0.0018397, 0.0024738, -0.0020999, 0.0026225, -0.0040244, 0.0041194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019654
time: 1.52 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020072, upper bound: 0.0019601
time: 1.51 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0029399, 0.0008272, -0.0030605, 0.0008842, -0.0034567, 0.0036032
1: -0.0046153, -0.0033624, -0.0046454, -0.0033337, -0.0012057, 0.0012457
2: 0.0108683, 0.0159493, 0.0107182, 0.0160369, -0.0046140, 0.0048126
3: 1.0067922, 1.0099990, 1.0067124, 1.0100740, -0.0032818, 0.0032866
4: -0.0042371, -0.0033987, -0.0042535, -0.0033758, -0.0007876, 0.0007547
5: 0.0017041, 0.0046091, 0.0016121, 0.0046539, -0.0026615, 0.0027740
6: -0.0026019, -0.0023019, -0.0026094, -0.0022957, -0.0003062, 0.0003075
7: -0.0130929, -0.0082728, -0.0130996, -0.0080644, -0.0049973, 0.0047863
8: -0.0135305, -0.0044004, -0.0137203, -0.0041618, -0.0085439, 0.0081830
9: -0.0020252, 0.0025267, -0.0021365, 0.0026281, -0.0040579, 0.0042354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019848, upper bound: 0.0020703
time: 1.59 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019848, upper bound: 0.0020729
time: 1.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028231, 0.0008566, -0.0029543, 0.0008301, -0.0033844, 0.0035149
1: -0.0045844, -0.0033479, -0.0046182, -0.0033609, -0.0011945, 0.0012167
2: 0.0110185, 0.0159945, 0.0108500, 0.0159538, -0.0045406, 0.0047029
3: 1.0067562, 1.0099220, 1.0067877, 1.0100063, -0.0032501, 0.0031344
4: -0.0042456, -0.0034217, -0.0042380, -0.0033959, -0.0007713, 0.0007460
5: 0.0017935, 0.0046322, 0.0016930, 0.0046114, -0.0026075, 0.0027072
6: -0.0026020, -0.0023082, -0.0026026, -0.0023013, -0.0003007, 0.0002944
7: -0.0130963, -0.0084768, -0.0130932, -0.0082447, -0.0048183, 0.0045855
8: -0.0136285, -0.0046378, -0.0135403, -0.0043728, -0.0083728, 0.0081030
9: -0.0019118, 0.0025790, -0.0020365, 0.0025319, -0.0040252, 0.0041544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020264, upper bound: 0.0019499
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020264, upper bound: 0.0019371
time: 1.76 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0030014, 0.0008778, -0.0029949, 0.0008332, -0.0034770, 0.0035910
1: -0.0046300, -0.0033371, -0.0046288, -0.0033592, -0.0012079, 0.0012511
2: 0.0107923, 0.0160271, 0.0107986, 0.0159586, -0.0046352, 0.0048101
3: 1.0067239, 1.0100356, 1.0067815, 1.0100325, -0.0033087, 0.0032542
4: -0.0042516, -0.0033872, -0.0042389, -0.0033881, -0.0007896, 0.0007566
5: 0.0016571, 0.0046489, 0.0016620, 0.0046139, -0.0026761, 0.0027662
6: -0.0026072, -0.0022989, -0.0026040, -0.0022991, -0.0003080, 0.0003052
7: -0.0130988, -0.0081689, -0.0130936, -0.0081725, -0.0048945, 0.0048844
8: -0.0136991, -0.0042815, -0.0135507, -0.0042911, -0.0085760, 0.0081863
9: -0.0020797, 0.0026168, -0.0020752, 0.0025375, -0.0040483, 0.0042598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020026, upper bound: 0.0020590
time: 1.53 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020026, upper bound: 0.0020612
time: 1.72 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028231, 0.0008566, -0.0030218, 0.0008810, -0.0033852, 0.0035335
1: -0.0045844, -0.0033479, -0.0046354, -0.0033354, -0.0011998, 0.0012189
2: 0.0110185, 0.0159945, 0.0107670, 0.0160320, -0.0045408, 0.0047145
3: 1.0067562, 1.0099220, 1.0067186, 1.0100492, -0.0032930, 0.0032034
4: -0.0042456, -0.0034217, -0.0042525, -0.0033833, -0.0007709, 0.0007460
5: 0.0017935, 0.0046322, 0.0016416, 0.0046514, -0.0026082, 0.0027199
6: -0.0026020, -0.0023082, -0.0026080, -0.0022977, -0.0003043, 0.0002998
7: -0.0130963, -0.0084768, -0.0130992, -0.0081330, -0.0049260, 0.0045868
8: -0.0136285, -0.0046378, -0.0137097, -0.0042392, -0.0083536, 0.0081065
9: -0.0019118, 0.0025790, -0.0020999, 0.0026225, -0.0040272, 0.0041370

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019521
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019405
time: 1.65 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0030014, 0.0008778, -0.0030605, 0.0008842, -0.0034758, 0.0036116
1: -0.0046300, -0.0033371, -0.0046454, -0.0033337, -0.0012097, 0.0012540
2: 0.0107923, 0.0160271, 0.0107182, 0.0160369, -0.0046312, 0.0048239
3: 1.0067239, 1.0100356, 1.0067124, 1.0100740, -0.0033501, 0.0033232
4: -0.0042516, -0.0033872, -0.0042535, -0.0033758, -0.0007899, 0.0007557
5: 0.0016571, 0.0046489, 0.0016121, 0.0046539, -0.0026751, 0.0027805
6: -0.0026072, -0.0022989, -0.0026094, -0.0022957, -0.0003115, 0.0003105
7: -0.0130988, -0.0081689, -0.0130996, -0.0080644, -0.0049995, 0.0048867
8: -0.0136991, -0.0042815, -0.0137203, -0.0041618, -0.0085676, 0.0081769
9: -0.0020797, 0.0026168, -0.0021365, 0.0026281, -0.0040433, 0.0042489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020027, upper bound: 0.0020622
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020027, upper bound: 0.0020650
time: 1.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.82 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019505
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019394
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0019865, upper bound: 0.0020592
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0019865, upper bound: 0.0020614
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019654
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020072, upper bound: 0.0019601
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0019848, upper bound: 0.0020703
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0019848, upper bound: 0.0020729
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020264, upper bound: 0.0019499
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020264, upper bound: 0.0019371
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020026, upper bound: 0.0020590
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020026, upper bound: 0.0020612
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019521
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019405
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020027, upper bound: 0.0020622
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.82
Output dim: 3, lower bound: -0.0020027, upper bound: 0.0020650

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027447, 0.0007975, -0.0029512, 0.0007880, -0.0032356, 0.0034145
1: -0.0045649, -0.0033772, -0.0046177, -0.0033815, -0.0011432, 0.0011791
2: 0.0111151, 0.0159036, 0.0108536, 0.0158891, -0.0043290, 0.0045487
3: 1.0068339, 1.0098733, 1.0068362, 1.0100049, -0.0031710, 0.0030371
4: -0.0042286, -0.0034365, -0.0042259, -0.0033965, -0.0007420, 0.0007101
5: 0.0018531, 0.0045857, 0.0016953, 0.0045783, -0.0024921, 0.0026281
6: -0.0025957, -0.0023122, -0.0026002, -0.0023014, -0.0002944, 0.0002880
7: -0.0130894, -0.0086144, -0.0130883, -0.0082508, -0.0048018, 0.0044405
8: -0.0134315, -0.0047916, -0.0134001, -0.0043780, -0.0080270, 0.0077130
9: -0.0018397, 0.0024738, -0.0020344, 0.0024570, -0.0038320, 0.0039728

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019394
time: 1.49 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019394
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027431, 0.0007771, -0.0031474, 0.0007203, -0.0032215, 0.0036097
1: -0.0045646, -0.0033870, -0.0046578, -0.0034123, -0.0011522, 0.0012203
2: 0.0111171, 0.0158723, 0.0106207, 0.0157849, -0.0043078, 0.0047775
3: 1.0068563, 1.0098726, 1.0068871, 1.0101050, -0.0032487, 0.0029855
4: -0.0042228, -0.0034368, -0.0042065, -0.0033632, -0.0007747, 0.0007063
5: 0.0018544, 0.0045697, 0.0015468, 0.0045250, -0.0024810, 0.0027753
6: -0.0025946, -0.0023123, -0.0026056, -0.0022931, -0.0003015, 0.0002933
7: -0.0130870, -0.0086177, -0.0130803, -0.0078177, -0.0052359, 0.0044357
8: -0.0133637, -0.0047945, -0.0131744, -0.0040476, -0.0083584, 0.0076690
9: -0.0018385, 0.0024375, -0.0021831, 0.0023364, -0.0038089, 0.0041227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019049
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0018876
time: 1.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029399, 0.0008272, -0.0027447, 0.0007975, -0.0034400, 0.0032627
1: -0.0046153, -0.0033624, -0.0045649, -0.0033772, -0.0011885, 0.0011492
2: 0.0108683, 0.0159493, 0.0111151, 0.0159036, -0.0045977, 0.0043708
3: 1.0067922, 1.0099990, 1.0068339, 1.0098733, -0.0030811, 0.0031651
4: -0.0042371, -0.0033987, -0.0042286, -0.0034365, -0.0007179, 0.0007523
5: 0.0017041, 0.0046091, 0.0018531, 0.0045857, -0.0026489, 0.0025135
6: -0.0026019, -0.0023019, -0.0025957, -0.0023122, -0.0002896, 0.0002939
7: -0.0130929, -0.0082728, -0.0130894, -0.0086144, -0.0044437, 0.0047828
8: -0.0135305, -0.0044004, -0.0134315, -0.0047916, -0.0078035, 0.0081544
9: -0.0020252, 0.0025267, -0.0018397, 0.0024738, -0.0040389, 0.0038804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019505, upper bound: 0.0020087
time: 1.40 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020087
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029399, 0.0008272, -0.0029399, 0.0008272, -0.0033652, 0.0033652
1: -0.0046153, -0.0033624, -0.0046153, -0.0033624, -0.0011600, 0.0011600
2: 0.0108683, 0.0159493, 0.0108683, 0.0159493, -0.0044733, 0.0044733
3: 1.0067922, 1.0099990, 1.0067922, 1.0099990, -0.0032068, 0.0032068
4: -0.0042371, -0.0033987, -0.0042371, -0.0033987, -0.0007285, 0.0007285
5: 0.0017041, 0.0046091, 0.0017041, 0.0046091, -0.0025895, 0.0025895
6: -0.0026019, -0.0023019, -0.0026019, -0.0023019, -0.0003000, 0.0003000
7: -0.0130929, -0.0082728, -0.0130929, -0.0082728, -0.0047755, 0.0047755
8: -0.0135305, -0.0044004, -0.0135305, -0.0044004, -0.0078782, 0.0078782
9: -0.0020252, 0.0025267, -0.0020252, 0.0025267, -0.0038950, 0.0038950

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019506, upper bound: 0.0020102
time: 1.54 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020102
time: 1.74 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027447, 0.0007975, -0.0030157, 0.0008368, -0.0032937, 0.0035175
1: -0.0045649, -0.0033772, -0.0046341, -0.0033571, -0.0011703, 0.0012093
2: 0.0111151, 0.0159036, 0.0107747, 0.0159640, -0.0044184, 0.0046928
3: 1.0068339, 1.0098733, 1.0067695, 1.0100461, -0.0032122, 0.0031037
4: -0.0042286, -0.0034365, -0.0042399, -0.0033844, -0.0007671, 0.0007268
5: 0.0018531, 0.0045857, 0.0016464, 0.0046166, -0.0025379, 0.0027075
6: -0.0025957, -0.0023122, -0.0026054, -0.0022980, -0.0002977, 0.0002932
7: -0.0130894, -0.0086144, -0.0130940, -0.0081456, -0.0049109, 0.0044473
8: -0.0134315, -0.0047916, -0.0135624, -0.0042504, -0.0083128, 0.0079067
9: -0.0018397, 0.0024738, -0.0020951, 0.0025437, -0.0039355, 0.0041155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019600
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019600
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027431, 0.0007771, -0.0032347, 0.0007752, -0.0032826, 0.0037416
1: -0.0045646, -0.0033870, -0.0046763, -0.0033841, -0.0011793, 0.0012604
2: 0.0111171, 0.0158723, 0.0105146, 0.0158693, -0.0044018, 0.0049645
3: 1.0068563, 1.0098726, 1.0068102, 1.0101511, -0.0032948, 0.0030624
4: -0.0042228, -0.0034368, -0.0042223, -0.0033478, -0.0008071, 0.0007238
5: 0.0018544, 0.0045697, 0.0014805, 0.0045682, -0.0025291, 0.0028782
6: -0.0025946, -0.0023123, -0.0026117, -0.0022893, -0.0003053, 0.0002994
7: -0.0130870, -0.0086177, -0.0130868, -0.0076528, -0.0054032, 0.0044429
8: -0.0133637, -0.0047945, -0.0133573, -0.0038929, -0.0087185, 0.0078725
9: -0.0018385, 0.0024375, -0.0022518, 0.0024341, -0.0039177, 0.0043029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019261
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019083
time: 1.63 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029399, 0.0008272, -0.0028231, 0.0008566, -0.0035114, 0.0033662
1: -0.0046153, -0.0033624, -0.0045844, -0.0033479, -0.0012199, 0.0011797
2: 0.0108683, 0.0159493, 0.0110185, 0.0159945, -0.0047075, 0.0045127
3: 1.0067922, 1.0099990, 1.0067562, 1.0099220, -0.0031298, 0.0032429
4: -0.0042371, -0.0033987, -0.0042456, -0.0034217, -0.0007408, 0.0007727
5: 0.0017041, 0.0046091, 0.0017935, 0.0046322, -0.0027051, 0.0025931
6: -0.0026019, -0.0023019, -0.0026020, -0.0023082, -0.0002936, 0.0003001
7: -0.0130929, -0.0082728, -0.0130963, -0.0084768, -0.0045834, 0.0047911
8: -0.0135305, -0.0044004, -0.0136285, -0.0046378, -0.0080424, 0.0083923
9: -0.0020252, 0.0025267, -0.0019118, 0.0025790, -0.0041661, 0.0039928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019500, upper bound: 0.0020264
time: 1.73 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020264
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029399, 0.0008272, -0.0030014, 0.0008778, -0.0034365, 0.0034558
1: -0.0046153, -0.0033624, -0.0046300, -0.0033371, -0.0011892, 0.0011914
2: 0.0108683, 0.0159493, 0.0107923, 0.0160271, -0.0045828, 0.0046025
3: 1.0067922, 1.0099990, 1.0067239, 1.0100356, -0.0032434, 0.0032752
4: -0.0042371, -0.0033987, -0.0042516, -0.0033872, -0.0007505, 0.0007489
5: 0.0017041, 0.0046091, 0.0016571, 0.0046489, -0.0026456, 0.0026593
6: -0.0026019, -0.0023019, -0.0026072, -0.0022989, -0.0003030, 0.0003053
7: -0.0130929, -0.0082728, -0.0130988, -0.0081689, -0.0048819, 0.0047839
8: -0.0135305, -0.0044004, -0.0136991, -0.0042815, -0.0081155, 0.0081155
9: -0.0020252, 0.0025267, -0.0020797, 0.0026168, -0.0040218, 0.0040105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019500, upper bound: 0.0020285
time: 1.76 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020285
time: 1.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028231, 0.0008566, -0.0029512, 0.0007880, -0.0033391, 0.0034897
1: -0.0045844, -0.0033479, -0.0046177, -0.0033815, -0.0011737, 0.0012115
2: 0.0110185, 0.0159945, 0.0108536, 0.0158891, -0.0044709, 0.0046642
3: 1.0067562, 1.0099220, 1.0068362, 1.0100049, -0.0032487, 0.0030859
4: -0.0042456, -0.0034217, -0.0042259, -0.0033965, -0.0007636, 0.0007330
5: 0.0017935, 0.0046322, 0.0016953, 0.0045783, -0.0025718, 0.0026872
6: -0.0026020, -0.0023082, -0.0026002, -0.0023014, -0.0003006, 0.0002920
7: -0.0130963, -0.0084768, -0.0130883, -0.0082508, -0.0048106, 0.0045802
8: -0.0136285, -0.0046378, -0.0134001, -0.0043780, -0.0082772, 0.0079520
9: -0.0019118, 0.0025790, -0.0020344, 0.0024570, -0.0039445, 0.0041066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020263, upper bound: 0.0019371
time: 1.34 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020263, upper bound: 0.0019371
time: 1.59 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028216, 0.0008371, -0.0031474, 0.0007203, -0.0033251, 0.0036874
1: -0.0045841, -0.0033573, -0.0046578, -0.0034123, -0.0011719, 0.0012534
2: 0.0110204, 0.0159645, 0.0106207, 0.0157849, -0.0044499, 0.0048970
3: 1.0067778, 1.0099213, 1.0068871, 1.0101050, -0.0033273, 0.0030342
4: -0.0042400, -0.0034220, -0.0042065, -0.0033632, -0.0007970, 0.0007292
5: 0.0017947, 0.0046169, 0.0015468, 0.0045250, -0.0025608, 0.0028365
6: -0.0026009, -0.0023083, -0.0026056, -0.0022931, -0.0003078, 0.0002973
7: -0.0130940, -0.0084799, -0.0130803, -0.0078177, -0.0052450, 0.0045756
8: -0.0135635, -0.0046406, -0.0131744, -0.0040476, -0.0086173, 0.0079083
9: -0.0019107, 0.0025443, -0.0021831, 0.0023364, -0.0039215, 0.0042611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
time: 1.37 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0018872
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0030014, 0.0008778, -0.0027447, 0.0007975, -0.0035063, 0.0033262
1: -0.0046300, -0.0033371, -0.0045649, -0.0033772, -0.0012083, 0.0011787
2: 0.0107923, 0.0160271, 0.0111151, 0.0159036, -0.0046818, 0.0044683
3: 1.0067239, 1.0100356, 1.0068339, 1.0098733, -0.0031494, 0.0032017
4: -0.0042516, -0.0033872, -0.0042286, -0.0034365, -0.0007361, 0.0007660
5: 0.0016571, 0.0046489, 0.0018531, 0.0045857, -0.0026993, 0.0025634
6: -0.0026072, -0.0022989, -0.0025957, -0.0023122, -0.0002949, 0.0002969
7: -0.0130988, -0.0081689, -0.0130894, -0.0086144, -0.0044511, 0.0048884
8: -0.0136991, -0.0042815, -0.0134315, -0.0047916, -0.0080148, 0.0083023
9: -0.0020797, 0.0026168, -0.0018397, 0.0024738, -0.0041115, 0.0039933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019654, upper bound: 0.0020068
time: 1.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019601, upper bound: 0.0020072
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0030014, 0.0008778, -0.0029399, 0.0008272, -0.0034558, 0.0034365
1: -0.0046300, -0.0033371, -0.0046153, -0.0033624, -0.0011914, 0.0011892
2: 0.0107923, 0.0160271, 0.0108683, 0.0159493, -0.0046025, 0.0045828
3: 1.0067239, 1.0100356, 1.0067922, 1.0099990, -0.0032752, 0.0032434
4: -0.0042516, -0.0033872, -0.0042371, -0.0033987, -0.0007489, 0.0007505
5: 0.0016571, 0.0046489, 0.0017041, 0.0046091, -0.0026593, 0.0026456
6: -0.0026072, -0.0022989, -0.0026019, -0.0023019, -0.0003053, 0.0003030
7: -0.0130988, -0.0081689, -0.0130929, -0.0082728, -0.0047839, 0.0048819
8: -0.0136991, -0.0042815, -0.0135305, -0.0044004, -0.0081155, 0.0081155
9: -0.0020797, 0.0026168, -0.0020252, 0.0025267, -0.0040105, 0.0040218

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019654, upper bound: 0.0020083
time: 1.76 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019601, upper bound: 0.0020087
time: 1.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028231, 0.0008566, -0.0030157, 0.0008368, -0.0033386, 0.0035271
1: -0.0045844, -0.0033479, -0.0046341, -0.0033571, -0.0011785, 0.0012178
2: 0.0110185, 0.0159945, 0.0107747, 0.0159640, -0.0044692, 0.0047069
3: 1.0067562, 1.0099220, 1.0067695, 1.0100461, -0.0032899, 0.0031525
4: -0.0042456, -0.0034217, -0.0042399, -0.0033844, -0.0007698, 0.0007327
5: 0.0017935, 0.0046322, 0.0016464, 0.0046166, -0.0025715, 0.0027150
6: -0.0026020, -0.0023082, -0.0026054, -0.0022980, -0.0003040, 0.0002972
7: -0.0130963, -0.0084768, -0.0130940, -0.0081456, -0.0049133, 0.0045813
8: -0.0136285, -0.0046378, -0.0135624, -0.0042504, -0.0083432, 0.0079513
9: -0.0019118, 0.0025790, -0.0020951, 0.0025437, -0.0039442, 0.0041329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019406
time: 1.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019406
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028216, 0.0008371, -0.0032347, 0.0007752, -0.0033297, 0.0037494
1: -0.0045841, -0.0033573, -0.0046763, -0.0033841, -0.0011910, 0.0012653
2: 0.0110204, 0.0159645, 0.0105146, 0.0158693, -0.0044561, 0.0049732
3: 1.0067778, 1.0099213, 1.0068102, 1.0101511, -0.0033734, 0.0031111
4: -0.0042400, -0.0034220, -0.0042223, -0.0033478, -0.0008084, 0.0007303
5: 0.0017947, 0.0046169, 0.0014805, 0.0045682, -0.0025645, 0.0028841
6: -0.0026009, -0.0023083, -0.0026117, -0.0022893, -0.0003116, 0.0003034
7: -0.0130940, -0.0084799, -0.0130868, -0.0076528, -0.0054055, 0.0045773
8: -0.0135635, -0.0046406, -0.0133573, -0.0038929, -0.0087320, 0.0079247
9: -0.0019107, 0.0025443, -0.0022518, 0.0024341, -0.0039304, 0.0043079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.35 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0030014, 0.0008778, -0.0028231, 0.0008566, -0.0035243, 0.0033673
1: -0.0046300, -0.0033371, -0.0045844, -0.0033479, -0.0012202, 0.0011850
2: 0.0107923, 0.0160271, 0.0110185, 0.0159945, -0.0047076, 0.0045133
3: 1.0067239, 1.0100356, 1.0067562, 1.0099220, -0.0031981, 0.0032794
4: -0.0042516, -0.0033872, -0.0042456, -0.0034217, -0.0007409, 0.0007709
5: 0.0016571, 0.0046489, 0.0017935, 0.0046322, -0.0027132, 0.0025941
6: -0.0026072, -0.0022989, -0.0026020, -0.0023082, -0.0002989, 0.0003032
7: -0.0130988, -0.0081689, -0.0130963, -0.0084768, -0.0045847, 0.0048917
8: -0.0136991, -0.0042815, -0.0136285, -0.0046378, -0.0080469, 0.0083559
9: -0.0020797, 0.0026168, -0.0019118, 0.0025790, -0.0041413, 0.0039953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019658, upper bound: 0.0020114
time: 1.28 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019605, upper bound: 0.0020121
time: 1.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0030014, 0.0008778, -0.0030014, 0.0008778, -0.0034550, 0.0034550
1: -0.0046300, -0.0033371, -0.0046300, -0.0033371, -0.0011932, 0.0011932
2: 0.0107923, 0.0160271, 0.0107923, 0.0160271, -0.0045991, 0.0045991
3: 1.0067239, 1.0100356, 1.0067239, 1.0100356, -0.0033118, 0.0033118
4: -0.0042516, -0.0033872, -0.0042516, -0.0033872, -0.0007497, 0.0007497
5: 0.0016571, 0.0046489, 0.0016571, 0.0046489, -0.0026587, 0.0026587
6: -0.0026072, -0.0022989, -0.0026072, -0.0022989, -0.0003083, 0.0003083
7: -0.0130988, -0.0081689, -0.0130988, -0.0081689, -0.0048842, 0.0048842
8: -0.0136991, -0.0042815, -0.0136991, -0.0042815, -0.0081074, 0.0081074
9: -0.0020797, 0.0026168, -0.0020797, 0.0026168, -0.0040061, 0.0040061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019658, upper bound: 0.0020142
time: 1.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019605, upper bound: 0.0020150
time: 1.62 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.91 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019394
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020087, upper bound: 0.0019394
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019049
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0018876
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019505, upper bound: 0.0020087
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020087
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019506, upper bound: 0.0020102
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020102
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019600
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019600
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019261
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019083
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019500, upper bound: 0.0020264
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020264
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019500, upper bound: 0.0020285
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020285
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020263, upper bound: 0.0019371
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020263, upper bound: 0.0019371
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0018872
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019654, upper bound: 0.0020068
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019601, upper bound: 0.0020072
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019654, upper bound: 0.0020083
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019601, upper bound: 0.0020087
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019406
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0020280, upper bound: 0.0019406
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019658, upper bound: 0.0020114
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019605, upper bound: 0.0020121
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019658, upper bound: 0.0020142
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 3, lower bound: -0.0019605, upper bound: 0.0020150

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027416, 0.0007534, -0.0029512, 0.0007880, -0.0032110, 0.0033701
1: -0.0045644, -0.0033981, -0.0046177, -0.0033815, -0.0011375, 0.0011568
2: 0.0111187, 0.0158359, 0.0108536, 0.0158891, -0.0042903, 0.0044804
3: 1.0068822, 1.0098720, 1.0068362, 1.0100049, -0.0031227, 0.0030358
4: -0.0042160, -0.0034370, -0.0042259, -0.0033965, -0.0007293, 0.0007025
5: 0.0018555, 0.0045511, 0.0016953, 0.0045783, -0.0024726, 0.0025931
6: -0.0025932, -0.0023124, -0.0026002, -0.0023014, -0.0002918, 0.0002879
7: -0.0130842, -0.0086208, -0.0130883, -0.0082508, -0.0047966, 0.0044322
8: -0.0132848, -0.0047968, -0.0134001, -0.0043780, -0.0078789, 0.0076245
9: -0.0018377, 0.0023954, -0.0020344, 0.0024570, -0.0037876, 0.0038937

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019767, upper bound: 0.0019063
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029511, 0.0006903, -0.0029512, 0.0007880, -0.0034436, 0.0033331
1: -0.0046069, -0.0034274, -0.0046177, -0.0033815, -0.0011918, 0.0011506
2: 0.0108685, 0.0157389, 0.0108536, 0.0158891, -0.0045793, 0.0044235
3: 1.0069287, 1.0099781, 1.0068362, 1.0100049, -0.0030762, 0.0031419
4: -0.0041980, -0.0034012, -0.0042259, -0.0033965, -0.0007187, 0.0007454
5: 0.0016969, 0.0045014, 0.0016953, 0.0045783, -0.0026500, 0.0025640
6: -0.0025987, -0.0023036, -0.0026002, -0.0023014, -0.0002973, 0.0002966
7: -0.0130768, -0.0081643, -0.0130883, -0.0082508, -0.0047922, 0.0048931
8: -0.0130745, -0.0044425, -0.0134001, -0.0043780, -0.0077557, 0.0080617
9: -0.0019955, 0.0022830, -0.0020344, 0.0024570, -0.0039882, 0.0038279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019767, upper bound: 0.0019063
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027030, 0.0007636, -0.0031474, 0.0007203, -0.0031643, 0.0036002
1: -0.0045544, -0.0033937, -0.0046578, -0.0034123, -0.0011390, 0.0012143
2: 0.0111670, 0.0158515, 0.0106207, 0.0157849, -0.0042296, 0.0047628
3: 1.0068750, 1.0098473, 1.0068871, 1.0101050, -0.0032300, 0.0029602
4: -0.0042189, -0.0034444, -0.0042065, -0.0033632, -0.0007720, 0.0006927
5: 0.0018850, 0.0045591, 0.0015468, 0.0045250, -0.0024368, 0.0027678
6: -0.0025925, -0.0023144, -0.0026056, -0.0022931, -0.0002993, 0.0002912
7: -0.0130854, -0.0086887, -0.0130803, -0.0078177, -0.0052348, 0.0043637
8: -0.0133187, -0.0048739, -0.0131744, -0.0040476, -0.0083265, 0.0075179
9: -0.0018009, 0.0024135, -0.0021831, 0.0023364, -0.0037323, 0.0041057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0018876
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0018876
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026565, 0.0007920, -0.0031316, 0.0007178, -0.0031387, 0.0036571
1: -0.0045406, -0.0033787, -0.0046533, -0.0034135, -0.0011271, 0.0012465
2: 0.0112286, 0.0158951, 0.0106411, 0.0157811, -0.0041981, 0.0048551
3: 1.0068367, 1.0098130, 1.0068913, 1.0100938, -0.0032572, 0.0029217
4: -0.0042271, -0.0034543, -0.0042058, -0.0033664, -0.0007898, 0.0006882
5: 0.0019208, 0.0045814, 0.0015590, 0.0045230, -0.0024174, 0.0028130
6: -0.0025941, -0.0023172, -0.0026050, -0.0022941, -0.0003000, 0.0002878
7: -0.0130887, -0.0087552, -0.0130800, -0.0078432, -0.0052174, 0.0042996
8: -0.0134131, -0.0049791, -0.0131660, -0.0040817, -0.0085376, 0.0074732
9: -0.0017503, 0.0024640, -0.0021665, 0.0023319, -0.0037119, 0.0042199

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019003, upper bound: 0.0018876
time: 1.31 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019003, upper bound: 0.0018876
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0027447, 0.0007975, -0.0034125, 0.0032172
1: -0.0046147, -0.0033830, -0.0045649, -0.0033772, -0.0011823, 0.0011283
2: 0.0108720, 0.0158846, 0.0111151, 0.0159036, -0.0045508, 0.0043008
3: 1.0068407, 1.0099975, 1.0068339, 1.0098733, -0.0030326, 0.0031636
4: -0.0042251, -0.0033993, -0.0042286, -0.0034365, -0.0007049, 0.0007432
5: 0.0017064, 0.0045760, 0.0018531, 0.0045857, -0.0026269, 0.0024777
6: -0.0025995, -0.0023020, -0.0025957, -0.0023122, -0.0002873, 0.0002937
7: -0.0130880, -0.0082790, -0.0130894, -0.0086144, -0.0044383, 0.0047746
8: -0.0133904, -0.0044056, -0.0134315, -0.0047916, -0.0076518, 0.0080460
9: -0.0020231, 0.0024518, -0.0018397, 0.0024738, -0.0039845, 0.0037993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020087
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020087
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0027431, 0.0007771, -0.0035966, 0.0032031
1: -0.0046538, -0.0034138, -0.0045646, -0.0033870, -0.0012194, 0.0011371
2: 0.0106433, 0.0157805, 0.0111171, 0.0158723, -0.0047656, 0.0042795
3: 1.0068923, 1.0100950, 1.0068563, 1.0098726, -0.0029802, 0.0032387
4: -0.0042057, -0.0033664, -0.0042228, -0.0034368, -0.0007010, 0.0007741
5: 0.0015609, 0.0045227, 0.0018544, 0.0045697, -0.0027658, 0.0024665
6: -0.0026047, -0.0022940, -0.0025946, -0.0023123, -0.0002924, 0.0003007
7: -0.0130800, -0.0078570, -0.0130870, -0.0086177, -0.0044336, 0.0051965
8: -0.0131647, -0.0040777, -0.0133637, -0.0047945, -0.0076075, 0.0083539
9: -0.0021684, 0.0023312, -0.0018385, 0.0024375, -0.0041194, 0.0037761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019049, upper bound: 0.0019701
time: 1.52 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019701
time: 1.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0029399, 0.0008272, -0.0033344, 0.0033187
1: -0.0046147, -0.0033830, -0.0046153, -0.0033624, -0.0011541, 0.0011392
2: 0.0108720, 0.0158846, 0.0108683, 0.0159493, -0.0044247, 0.0044017
3: 1.0068407, 1.0099975, 1.0067922, 1.0099990, -0.0031583, 0.0032053
4: -0.0042251, -0.0033993, -0.0042371, -0.0033987, -0.0007152, 0.0007191
5: 0.0017064, 0.0045760, 0.0017041, 0.0046091, -0.0025653, 0.0025529
6: -0.0025995, -0.0023020, -0.0026019, -0.0023019, -0.0002976, 0.0002999
7: -0.0130880, -0.0082790, -0.0130929, -0.0082728, -0.0047701, 0.0047670
8: -0.0133904, -0.0044056, -0.0135305, -0.0044004, -0.0077230, 0.0077698
9: -0.0020231, 0.0024518, -0.0020252, 0.0025267, -0.0038400, 0.0038121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019519, upper bound: 0.0020102
time: 1.48 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019519, upper bound: 0.0020102
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0029387, 0.0008063, -0.0035354, 0.0032821
1: -0.0046538, -0.0034138, -0.0046150, -0.0033725, -0.0012001, 0.0011437
2: 0.0106433, 0.0157805, 0.0108698, 0.0159172, -0.0046727, 0.0043436
3: 1.0068923, 1.0100950, 1.0068153, 1.0099984, -0.0031061, 0.0032797
4: -0.0042057, -0.0033664, -0.0042312, -0.0033989, -0.0007039, 0.0007556
5: 0.0015609, 0.0045227, 0.0017050, 0.0045927, -0.0027179, 0.0025241
6: -0.0026047, -0.0022940, -0.0026008, -0.0023019, -0.0003028, 0.0003069
7: -0.0130800, -0.0078570, -0.0130904, -0.0082751, -0.0047645, 0.0051902
8: -0.0131647, -0.0040777, -0.0134609, -0.0044024, -0.0075920, 0.0081390
9: -0.0021684, 0.0023312, -0.0020244, 0.0024895, -0.0040077, 0.0037445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019235, upper bound: 0.0019736
time: 1.57 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019112, upper bound: 0.0019736
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027416, 0.0007534, -0.0030157, 0.0008368, -0.0032723, 0.0034712
1: -0.0045644, -0.0033981, -0.0046341, -0.0033571, -0.0011639, 0.0011889
2: 0.0111187, 0.0158359, 0.0107747, 0.0159640, -0.0043846, 0.0046217
3: 1.0068822, 1.0098720, 1.0067695, 1.0100461, -0.0031639, 0.0031024
4: -0.0042160, -0.0034370, -0.0042399, -0.0033844, -0.0007539, 0.0007201
5: 0.0018555, 0.0045511, 0.0016464, 0.0046166, -0.0025209, 0.0026711
6: -0.0025932, -0.0023124, -0.0026054, -0.0022980, -0.0002952, 0.0002930
7: -0.0130842, -0.0086208, -0.0130940, -0.0081456, -0.0049055, 0.0044394
8: -0.0132848, -0.0047968, -0.0135624, -0.0042504, -0.0081587, 0.0078288
9: -0.0018377, 0.0023954, -0.0020951, 0.0025437, -0.0038968, 0.0040331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019756, upper bound: 0.0019160
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019174
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0029511, 0.0006903, -0.0030157, 0.0008368, -0.0035049, 0.0034336
1: -0.0046069, -0.0034274, -0.0046341, -0.0033571, -0.0012182, 0.0011817
2: 0.0108685, 0.0157389, 0.0107747, 0.0159640, -0.0046736, 0.0045639
3: 1.0069287, 1.0099781, 1.0067695, 1.0100461, -0.0031174, 0.0032085
4: -0.0041980, -0.0034012, -0.0042399, -0.0033844, -0.0007431, 0.0007629
5: 0.0016969, 0.0045014, 0.0016464, 0.0046166, -0.0026983, 0.0026415
6: -0.0025987, -0.0023036, -0.0026054, -0.0022980, -0.0003007, 0.0003018
7: -0.0130768, -0.0081643, -0.0130940, -0.0081456, -0.0049011, 0.0049003
8: -0.0130745, -0.0044425, -0.0135624, -0.0042504, -0.0080334, 0.0082660
9: -0.0019955, 0.0022830, -0.0020951, 0.0025437, -0.0040974, 0.0039662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019756, upper bound: 0.0019160
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019174
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027030, 0.0007636, -0.0032347, 0.0007752, -0.0032280, 0.0037317
1: -0.0045544, -0.0033937, -0.0046763, -0.0033841, -0.0011648, 0.0012529
2: 0.0111670, 0.0158515, 0.0105146, 0.0158693, -0.0043274, 0.0049493
3: 1.0068750, 1.0098473, 1.0068102, 1.0101511, -0.0032761, 0.0030371
4: -0.0042189, -0.0034444, -0.0042223, -0.0033478, -0.0008042, 0.0007109
5: 0.0018850, 0.0045591, 0.0014805, 0.0045682, -0.0024869, 0.0028704
6: -0.0025925, -0.0023144, -0.0026117, -0.0022893, -0.0003031, 0.0002973
7: -0.0130854, -0.0086887, -0.0130868, -0.0076528, -0.0054020, 0.0043711
8: -0.0133187, -0.0048739, -0.0133573, -0.0038929, -0.0086855, 0.0077299
9: -0.0018009, 0.0024135, -0.0022518, 0.0024341, -0.0038456, 0.0042853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019079
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019083
time: 1.45 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0026565, 0.0007920, -0.0032169, 0.0007726, -0.0032021, 0.0037842
1: -0.0045406, -0.0033787, -0.0046716, -0.0033855, -0.0011551, 0.0012823
2: 0.0112286, 0.0158951, 0.0105372, 0.0158654, -0.0042957, 0.0050343
3: 1.0068367, 1.0098130, 1.0068147, 1.0101392, -0.0033026, 0.0029982
4: -0.0042271, -0.0034543, -0.0042215, -0.0033513, -0.0008206, 0.0007064
5: 0.0019208, 0.0045814, 0.0014941, 0.0045661, -0.0024673, 0.0029120
6: -0.0025941, -0.0023172, -0.0026110, -0.0022903, -0.0003037, 0.0002938
7: -0.0130887, -0.0087552, -0.0130865, -0.0076826, -0.0053802, 0.0043071
8: -0.0134131, -0.0049791, -0.0133487, -0.0039301, -0.0088777, 0.0076846
9: -0.0017503, 0.0024640, -0.0022343, 0.0024295, -0.0038249, 0.0043888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019079
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019083
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0028231, 0.0008566, -0.0034876, 0.0033207
1: -0.0046147, -0.0033830, -0.0045844, -0.0033479, -0.0012147, 0.0011588
2: 0.0108720, 0.0158846, 0.0110185, 0.0159945, -0.0046663, 0.0044427
3: 1.0068407, 1.0099975, 1.0067562, 1.0099220, -0.0030813, 0.0032413
4: -0.0042251, -0.0033993, -0.0042456, -0.0034217, -0.0007277, 0.0007647
5: 0.0017064, 0.0045760, 0.0017935, 0.0046322, -0.0026860, 0.0025573
6: -0.0025995, -0.0023020, -0.0026020, -0.0023082, -0.0002913, 0.0003000
7: -0.0130880, -0.0082790, -0.0130963, -0.0084768, -0.0045781, 0.0047834
8: -0.0133904, -0.0044056, -0.0136285, -0.0046378, -0.0078908, 0.0082963
9: -0.0020231, 0.0024518, -0.0019118, 0.0025790, -0.0041182, 0.0039118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020264
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020264
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0028216, 0.0008371, -0.0036743, 0.0033066
1: -0.0046538, -0.0034138, -0.0045841, -0.0033573, -0.0012524, 0.0011676
2: 0.0106433, 0.0157805, 0.0110204, 0.0159645, -0.0048850, 0.0044216
3: 1.0068923, 1.0100950, 1.0067778, 1.0099213, -0.0030290, 0.0033172
4: -0.0042057, -0.0033664, -0.0042400, -0.0034220, -0.0007239, 0.0007963
5: 0.0015609, 0.0045227, 0.0017947, 0.0046169, -0.0028270, 0.0025463
6: -0.0026047, -0.0022940, -0.0026009, -0.0023083, -0.0002964, 0.0003070
7: -0.0130800, -0.0078570, -0.0130940, -0.0084799, -0.0045735, 0.0052056
8: -0.0131647, -0.0040777, -0.0135635, -0.0046406, -0.0078468, 0.0086127
9: -0.0021684, 0.0023312, -0.0019107, 0.0025443, -0.0042577, 0.0038886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019032, upper bound: 0.0019879
time: 1.58 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0030014, 0.0008778, -0.0034093, 0.0034092
1: -0.0046147, -0.0033830, -0.0046300, -0.0033371, -0.0011845, 0.0011706
2: 0.0108720, 0.0158846, 0.0107923, 0.0160271, -0.0045399, 0.0045309
3: 1.0068407, 1.0099975, 1.0067239, 1.0100356, -0.0031949, 0.0032736
4: -0.0042251, -0.0033993, -0.0042516, -0.0033872, -0.0007372, 0.0007405
5: 0.0017064, 0.0045760, 0.0016571, 0.0046489, -0.0026242, 0.0026227
6: -0.0025995, -0.0023020, -0.0026072, -0.0022989, -0.0003006, 0.0003051
7: -0.0130880, -0.0082790, -0.0130988, -0.0081689, -0.0048764, 0.0047758
8: -0.0133904, -0.0044056, -0.0136991, -0.0042815, -0.0079603, 0.0080193
9: -0.0020231, 0.0024518, -0.0020797, 0.0026168, -0.0039734, 0.0039276

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019499, upper bound: 0.0020285
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019499, upper bound: 0.0020285
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0029997, 0.0008576, -0.0036122, 0.0033919
1: -0.0046538, -0.0034138, -0.0046296, -0.0033468, -0.0012310, 0.0011778
2: 0.0106433, 0.0157805, 0.0107944, 0.0159960, -0.0047908, 0.0045048
3: 1.0068923, 1.0100950, 1.0067458, 1.0100348, -0.0031425, 0.0033492
4: -0.0042057, -0.0033664, -0.0042459, -0.0033875, -0.0007325, 0.0007776
5: 0.0015609, 0.0045227, 0.0016585, 0.0046330, -0.0027783, 0.0026091
6: -0.0026047, -0.0022940, -0.0026061, -0.0022989, -0.0003058, 0.0003121
7: -0.0130800, -0.0078570, -0.0130965, -0.0081725, -0.0048711, 0.0051992
8: -0.0131647, -0.0040777, -0.0136318, -0.0042847, -0.0079058, 0.0083951
9: -0.0021684, 0.0023312, -0.0020784, 0.0025808, -0.0041445, 0.0038988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019212, upper bound: 0.0019922
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019099, upper bound: 0.0019921
time: 1.84 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028177, 0.0008106, -0.0029512, 0.0007880, -0.0033336, 0.0034388
1: -0.0045833, -0.0033700, -0.0046177, -0.0033815, -0.0011728, 0.0011858
2: 0.0110251, 0.0159238, 0.0108536, 0.0158891, -0.0044644, 0.0045859
3: 1.0068079, 1.0099194, 1.0068362, 1.0100049, -0.0031970, 0.0030832
4: -0.0042324, -0.0034227, -0.0042259, -0.0033965, -0.0007490, 0.0007321
5: 0.0017977, 0.0045961, 0.0016953, 0.0045783, -0.0025677, 0.0026472
6: -0.0025992, -0.0023085, -0.0026002, -0.0023014, -0.0002978, 0.0002918
7: -0.0130909, -0.0084878, -0.0130883, -0.0082508, -0.0048046, 0.0045692
8: -0.0134753, -0.0046476, -0.0134001, -0.0043780, -0.0081076, 0.0079437
9: -0.0019077, 0.0024972, -0.0020344, 0.0024570, -0.0039412, 0.0040159

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019937, upper bound: 0.0019062
time: 1.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019063
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0030486, 0.0007533, -0.0029512, 0.0007880, -0.0035882, 0.0034177
1: -0.0046284, -0.0033956, -0.0046177, -0.0033815, -0.0012360, 0.0011849
2: 0.0107505, 0.0158357, 0.0108536, 0.0158891, -0.0047785, 0.0045536
3: 1.0068442, 1.0100317, 1.0068362, 1.0100049, -0.0031607, 0.0031955
4: -0.0042160, -0.0033837, -0.0042259, -0.0033965, -0.0007430, 0.0007797
5: 0.0016228, 0.0045509, 0.0016953, 0.0045783, -0.0027615, 0.0026306
6: -0.0026057, -0.0022992, -0.0026002, -0.0023014, -0.0003043, 0.0003011
7: -0.0130842, -0.0079707, -0.0130883, -0.0082508, -0.0048021, 0.0050891
8: -0.0132843, -0.0042660, -0.0134001, -0.0043780, -0.0080375, 0.0084377
9: -0.0020752, 0.0023951, -0.0020344, 0.0024570, -0.0041742, 0.0039785

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019937, upper bound: 0.0019062
time: 1.41 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019063
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027651, 0.0008239, -0.0031474, 0.0007203, -0.0032642, 0.0036772
1: -0.0045701, -0.0033639, -0.0046578, -0.0034123, -0.0011579, 0.0012459
2: 0.0110910, 0.0159442, 0.0106207, 0.0157849, -0.0043679, 0.0048813
3: 1.0067976, 1.0098865, 1.0068871, 1.0101050, -0.0033075, 0.0029994
4: -0.0042362, -0.0034327, -0.0042065, -0.0033632, -0.0007941, 0.0007155
5: 0.0018377, 0.0046065, 0.0015468, 0.0045250, -0.0025141, 0.0028285
6: -0.0025983, -0.0023112, -0.0026056, -0.0022931, -0.0003052, 0.0002944
7: -0.0130925, -0.0085845, -0.0130803, -0.0078177, -0.0052438, 0.0044702
8: -0.0135195, -0.0047509, -0.0131744, -0.0040476, -0.0085834, 0.0077577
9: -0.0018590, 0.0025208, -0.0021831, 0.0023364, -0.0038446, 0.0042429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0018872
time: 1.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0018872
time: 1.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0027162, 0.0008501, -0.0031316, 0.0007178, -0.0032377, 0.0037259
1: -0.0045544, -0.0033499, -0.0046533, -0.0034135, -0.0011408, 0.0012737
2: 0.0111583, 0.0159846, 0.0106411, 0.0157811, -0.0043366, 0.0049609
3: 1.0067619, 1.0098473, 1.0068913, 1.0100938, -0.0033319, 0.0029560
4: -0.0042437, -0.0034438, -0.0042058, -0.0033664, -0.0008095, 0.0007112
5: 0.0018755, 0.0046272, 0.0015590, 0.0045230, -0.0024941, 0.0028672
6: -0.0025999, -0.0023144, -0.0026050, -0.0022941, -0.0003059, 0.0002906
7: -0.0130956, -0.0086468, -0.0130800, -0.0078432, -0.0052255, 0.0044103
8: -0.0136070, -0.0048698, -0.0131660, -0.0040817, -0.0087669, 0.0077154
9: -0.0018011, 0.0025676, -0.0021665, 0.0023319, -0.0038274, 0.0043424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 200

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0027447, 0.0007975, -0.0035000, 0.0032764
1: -0.0046287, -0.0033587, -0.0045649, -0.0033772, -0.0012072, 0.0011560
2: 0.0107999, 0.0159591, 0.0111151, 0.0159036, -0.0046743, 0.0043917
3: 1.0067751, 1.0100324, 1.0068339, 1.0098733, -0.0030981, 0.0031985
4: -0.0042390, -0.0033883, -0.0042286, -0.0034365, -0.0007218, 0.0007650
5: 0.0016620, 0.0046141, 0.0018531, 0.0045857, -0.0026945, 0.0025242
6: -0.0026045, -0.0022991, -0.0025957, -0.0023122, -0.0002923, 0.0002966
7: -0.0130936, -0.0081818, -0.0130894, -0.0086144, -0.0044453, 0.0048755
8: -0.0135519, -0.0042928, -0.0134315, -0.0047916, -0.0078488, 0.0082925
9: -0.0020749, 0.0025381, -0.0018397, 0.0024738, -0.0041075, 0.0039046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019600, upper bound: 0.0020068
time: 1.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019600, upper bound: 0.0020068
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0027431, 0.0007771, -0.0037179, 0.0032648
1: -0.0046698, -0.0033859, -0.0045646, -0.0033870, -0.0012546, 0.0011644
2: 0.0105463, 0.0158645, 0.0111171, 0.0158723, -0.0049361, 0.0043744
3: 1.0068165, 1.0101347, 1.0068563, 1.0098726, -0.0030560, 0.0032784
4: -0.0042213, -0.0033526, -0.0042228, -0.0034368, -0.0007187, 0.0008034
5: 0.0015005, 0.0045657, 0.0018544, 0.0045697, -0.0028600, 0.0025151
6: -0.0026105, -0.0022907, -0.0025946, -0.0023123, -0.0002982, 0.0003039
7: -0.0130864, -0.0077095, -0.0130870, -0.0086177, -0.0044408, 0.0053468
8: -0.0133467, -0.0039441, -0.0133637, -0.0047945, -0.0078133, 0.0086772
9: -0.0022276, 0.0024285, -0.0018385, 0.0024375, -0.0042813, 0.0038860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019261, upper bound: 0.0019688
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019688
time: 1.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0029399, 0.0008272, -0.0034497, 0.0033854
1: -0.0046287, -0.0033587, -0.0046153, -0.0033624, -0.0011904, 0.0011654
2: 0.0107999, 0.0159591, 0.0108683, 0.0159493, -0.0045953, 0.0045043
3: 1.0067751, 1.0100324, 1.0067922, 1.0099990, -0.0032239, 0.0032402
4: -0.0042390, -0.0033883, -0.0042371, -0.0033987, -0.0007343, 0.0007495
5: 0.0016620, 0.0046141, 0.0017041, 0.0046091, -0.0026548, 0.0026054
6: -0.0026045, -0.0022991, -0.0026019, -0.0023019, -0.0003026, 0.0003027
7: -0.0130936, -0.0081818, -0.0130929, -0.0082728, -0.0047779, 0.0048691
8: -0.0135519, -0.0042928, -0.0135305, -0.0044004, -0.0079453, 0.0081062
9: -0.0020749, 0.0025381, -0.0020252, 0.0025267, -0.0040068, 0.0039309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019737, upper bound: 0.0020082
time: 1.58 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019737, upper bound: 0.0020082
time: 1.58 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0029387, 0.0008063, -0.0036743, 0.0033526
1: -0.0046698, -0.0033859, -0.0046150, -0.0033725, -0.0012454, 0.0011707
2: 0.0105463, 0.0158645, 0.0108698, 0.0159172, -0.0048707, 0.0044520
3: 1.0068165, 1.0101347, 1.0068153, 1.0099984, -0.0031819, 0.0033194
4: -0.0042213, -0.0033526, -0.0042312, -0.0033989, -0.0007241, 0.0007909
5: 0.0015005, 0.0045657, 0.0017050, 0.0045927, -0.0028257, 0.0025795
6: -0.0026105, -0.0022907, -0.0026008, -0.0023019, -0.0003086, 0.0003101
7: -0.0130864, -0.0077095, -0.0130904, -0.0082751, -0.0047728, 0.0053427
8: -0.0133467, -0.0039441, -0.0134609, -0.0044024, -0.0078269, 0.0085338
9: -0.0022276, 0.0024285, -0.0020244, 0.0024895, -0.0042071, 0.0038700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019456, upper bound: 0.0019720
time: 1.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019320, upper bound: 0.0019720
time: 1.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028177, 0.0008106, -0.0030157, 0.0008368, -0.0033329, 0.0034781
1: -0.0045833, -0.0033700, -0.0046341, -0.0033571, -0.0011775, 0.0011971
2: 0.0110251, 0.0159238, 0.0107747, 0.0159640, -0.0044625, 0.0046316
3: 1.0068079, 1.0099194, 1.0067695, 1.0100461, -0.0032382, 0.0031499
4: -0.0042324, -0.0034227, -0.0042399, -0.0033844, -0.0007558, 0.0007317
5: 0.0017977, 0.0045961, 0.0016464, 0.0046166, -0.0025672, 0.0026765
6: -0.0025992, -0.0023085, -0.0026054, -0.0022980, -0.0003012, 0.0002969
7: -0.0130909, -0.0084878, -0.0130940, -0.0081456, -0.0049076, 0.0045703
8: -0.0134753, -0.0046476, -0.0135624, -0.0042504, -0.0081800, 0.0079425
9: -0.0019077, 0.0024972, -0.0020951, 0.0025437, -0.0039407, 0.0040457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019945, upper bound: 0.0019073
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
time: 1.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0030486, 0.0007533, -0.0030157, 0.0008368, -0.0035854, 0.0034470
1: -0.0046284, -0.0033956, -0.0046341, -0.0033571, -0.0012388, 0.0011916
2: 0.0107505, 0.0158357, 0.0107747, 0.0159640, -0.0047732, 0.0045838
3: 1.0068442, 1.0100317, 1.0067695, 1.0100461, -0.0032020, 0.0032622
4: -0.0042160, -0.0033837, -0.0042399, -0.0033844, -0.0007469, 0.0007783
5: 0.0016228, 0.0045509, 0.0016464, 0.0046166, -0.0027589, 0.0026520
6: -0.0026057, -0.0022992, -0.0026054, -0.0022980, -0.0003077, 0.0003062
7: -0.0130842, -0.0079707, -0.0130940, -0.0081456, -0.0049039, 0.0050901
8: -0.0132843, -0.0042660, -0.0135624, -0.0042504, -0.0080765, 0.0084230
9: -0.0020752, 0.0023951, -0.0020951, 0.0025437, -0.0041668, 0.0039903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019945, upper bound: 0.0019073
time: 1.53 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
time: 1.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027651, 0.0008239, -0.0032347, 0.0007752, -0.0032698, 0.0037394
1: -0.0045701, -0.0033639, -0.0046763, -0.0033841, -0.0011710, 0.0012563
2: 0.0110910, 0.0159442, 0.0105146, 0.0158693, -0.0043755, 0.0049577
3: 1.0067976, 1.0098865, 1.0068102, 1.0101511, -0.0033536, 0.0030763
4: -0.0042362, -0.0034327, -0.0042223, -0.0033478, -0.0008055, 0.0007169
5: 0.0018377, 0.0046065, 0.0014805, 0.0045682, -0.0025185, 0.0028761
6: -0.0025983, -0.0023112, -0.0026117, -0.0022893, -0.0003090, 0.0003005
7: -0.0130925, -0.0085845, -0.0130868, -0.0076528, -0.0054043, 0.0044722
8: -0.0135195, -0.0047509, -0.0133573, -0.0038929, -0.0086985, 0.0077753
9: -0.0018590, 0.0025208, -0.0022518, 0.0024341, -0.0038570, 0.0042899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0027162, 0.0008501, -0.0032169, 0.0007726, -0.0032440, 0.0037882
1: -0.0045544, -0.0033499, -0.0046716, -0.0033855, -0.0011652, 0.0012845
2: 0.0111583, 0.0159846, 0.0105372, 0.0158654, -0.0043454, 0.0050368
3: 1.0067619, 1.0098473, 1.0068147, 1.0101392, -0.0033773, 0.0030326
4: -0.0042437, -0.0034438, -0.0042215, -0.0033513, -0.0008208, 0.0007126
5: 0.0018755, 0.0046272, 0.0014941, 0.0045661, -0.0024990, 0.0029149
6: -0.0025999, -0.0023144, -0.0026110, -0.0022903, -0.0003096, 0.0002966
7: -0.0130956, -0.0086468, -0.0130865, -0.0076826, -0.0053820, 0.0044121
8: -0.0136070, -0.0048698, -0.0133487, -0.0039301, -0.0088788, 0.0077355
9: -0.0018011, 0.0025676, -0.0022343, 0.0024295, -0.0038404, 0.0043875

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.48 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.52 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0028231, 0.0008566, -0.0035177, 0.0033207
1: -0.0046287, -0.0033587, -0.0045844, -0.0033479, -0.0012190, 0.0011636
2: 0.0107999, 0.0159591, 0.0110185, 0.0159945, -0.0046993, 0.0044417
3: 1.0067751, 1.0100324, 1.0067562, 1.0099220, -0.0031469, 0.0032762
4: -0.0042390, -0.0033883, -0.0042456, -0.0034217, -0.0007275, 0.0007697
5: 0.0016620, 0.0046141, 0.0017935, 0.0046322, -0.0027082, 0.0025574
6: -0.0026045, -0.0022991, -0.0026020, -0.0023082, -0.0002963, 0.0003029
7: -0.0130936, -0.0081818, -0.0130963, -0.0084768, -0.0045792, 0.0048788
8: -0.0135519, -0.0042928, -0.0136285, -0.0046378, -0.0078918, 0.0083448
9: -0.0020749, 0.0025381, -0.0019118, 0.0025790, -0.0041369, 0.0039124

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019604, upper bound: 0.0020114
time: 1.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019604, upper bound: 0.0020114
time: 1.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0028216, 0.0008371, -0.0037300, 0.0033114
1: -0.0046698, -0.0033859, -0.0045841, -0.0033573, -0.0012620, 0.0011757
2: 0.0105463, 0.0158645, 0.0110204, 0.0159645, -0.0049508, 0.0044280
3: 1.0068165, 1.0101347, 1.0067778, 1.0099213, -0.0031048, 0.0033569
4: -0.0042213, -0.0033526, -0.0042400, -0.0034220, -0.0007251, 0.0008057
5: 0.0015005, 0.0045657, 0.0017947, 0.0046169, -0.0028692, 0.0025502
6: -0.0026105, -0.0022907, -0.0026009, -0.0023083, -0.0003022, 0.0003102
7: -0.0130864, -0.0077095, -0.0130940, -0.0084799, -0.0045752, 0.0053488
8: -0.0133467, -0.0039441, -0.0135635, -0.0046406, -0.0078639, 0.0087041
9: -0.0022276, 0.0024285, -0.0019107, 0.0025443, -0.0042957, 0.0038979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019261, upper bound: 0.0019714
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019083, upper bound: 0.0019714
time: 1.53 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0030014, 0.0008778, -0.0034488, 0.0034060
1: -0.0046287, -0.0033587, -0.0046300, -0.0033371, -0.0011922, 0.0011721
2: 0.0107999, 0.0159591, 0.0107923, 0.0160271, -0.0045918, 0.0045238
3: 1.0067751, 1.0100324, 1.0067239, 1.0100356, -0.0032605, 0.0033085
4: -0.0042390, -0.0033883, -0.0042516, -0.0033872, -0.0007357, 0.0007487
5: 0.0016620, 0.0046141, 0.0016571, 0.0046489, -0.0026540, 0.0026201
6: -0.0026045, -0.0022991, -0.0026072, -0.0022989, -0.0003057, 0.0003080
7: -0.0130936, -0.0081818, -0.0130988, -0.0081689, -0.0048785, 0.0048714
8: -0.0135519, -0.0042928, -0.0136991, -0.0042815, -0.0079441, 0.0080976
9: -0.0020749, 0.0025381, -0.0020797, 0.0026168, -0.0040022, 0.0039189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019757, upper bound: 0.0020142
time: 1.25 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019757, upper bound: 0.0020142
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0029997, 0.0008576, -0.0036724, 0.0033960
1: -0.0046698, -0.0033859, -0.0046296, -0.0033468, -0.0012453, 0.0011834
2: 0.0105463, 0.0158645, 0.0107944, 0.0159960, -0.0048649, 0.0045090
3: 1.0068165, 1.0101347, 1.0067458, 1.0100348, -0.0032183, 0.0033889
4: -0.0042213, -0.0033526, -0.0042459, -0.0033875, -0.0007330, 0.0007892
5: 0.0015005, 0.0045657, 0.0016585, 0.0046330, -0.0028240, 0.0026123
6: -0.0026105, -0.0022907, -0.0026061, -0.0022989, -0.0003116, 0.0003154
7: -0.0130864, -0.0077095, -0.0130965, -0.0081725, -0.0048740, 0.0053440
8: -0.0133467, -0.0039441, -0.0136318, -0.0042847, -0.0079140, 0.0085133
9: -0.0022276, 0.0024285, -0.0020784, 0.0025808, -0.0041973, 0.0039032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 195
type: B, layer: 1, pos: 242
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 90
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 71
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019212, upper bound: 0.0019771
time: 1.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019333, upper bound: 0.0019771
time: 1.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 5.28 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019767, upper bound: 0.0019063
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019767, upper bound: 0.0019063
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0018876
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0018876
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019003, upper bound: 0.0018876
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019003, upper bound: 0.0018876
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020087
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019394, upper bound: 0.0020087
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019049, upper bound: 0.0019701
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019701
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019519, upper bound: 0.0020102
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019519, upper bound: 0.0020102
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019235, upper bound: 0.0019736
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019112, upper bound: 0.0019736
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019756, upper bound: 0.0019160
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019174
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019756, upper bound: 0.0019160
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019174
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019079
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019083
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019079
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0019083
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020264
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019371, upper bound: 0.0020264
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019032, upper bound: 0.0019879
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019499, upper bound: 0.0020285
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019499, upper bound: 0.0020285
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019212, upper bound: 0.0019922
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019099, upper bound: 0.0019921
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019937, upper bound: 0.0019062
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019063
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019937, upper bound: 0.0019062
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019063
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0018872
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0018872
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019600, upper bound: 0.0020068
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019600, upper bound: 0.0020068
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019261, upper bound: 0.0019688
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019688
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019737, upper bound: 0.0020082
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019737, upper bound: 0.0020082
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019456, upper bound: 0.0019720
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019320, upper bound: 0.0019720
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019945, upper bound: 0.0019073
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019945, upper bound: 0.0019073
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019604, upper bound: 0.0020114
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019604, upper bound: 0.0020114
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019261, upper bound: 0.0019714
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019083, upper bound: 0.0019714
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019757, upper bound: 0.0020142
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019757, upper bound: 0.0020142
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019212, upper bound: 0.0019771
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 5.28
Output dim: 3, lower bound: -0.0019333, upper bound: 0.0019771

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027416, 0.0007534, -0.0029053, 0.0007750, -0.0032014, 0.0033192
1: -0.0045644, -0.0033981, -0.0046063, -0.0033881, -0.0011310, 0.0011417
2: 0.0111187, 0.0158359, 0.0109116, 0.0158690, -0.0042757, 0.0044138
3: 1.0068822, 1.0098720, 1.0068551, 1.0099767, -0.0030946, 0.0030168
4: -0.0042160, -0.0034370, -0.0042222, -0.0034053, -0.0007187, 0.0006998
5: 0.0018555, 0.0045511, 0.0017304, 0.0045680, -0.0024651, 0.0025540
6: -0.0025932, -0.0023124, -0.0025979, -0.0023037, -0.0002895, 0.0002856
7: -0.0130842, -0.0086208, -0.0130868, -0.0083347, -0.0047125, 0.0044311
8: -0.0132848, -0.0047968, -0.0133566, -0.0044688, -0.0077694, 0.0075927
9: -0.0018377, 0.0023954, -0.0019923, 0.0024337, -0.0037707, 0.0038383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019971, upper bound: 0.0019181
time: 1.56 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019971, upper bound: 0.0019181
time: 1.54 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027260, 0.0007509, -0.0028569, 0.0008070, -0.0032606, 0.0032912
1: -0.0045602, -0.0033993, -0.0045914, -0.0033710, -0.0011642, 0.0011351
2: 0.0111387, 0.0158321, 0.0109769, 0.0159182, -0.0043712, 0.0043809
3: 1.0068861, 1.0098618, 1.0068119, 1.0099394, -0.0030533, 0.0030500
4: -0.0042153, -0.0034402, -0.0042314, -0.0034159, -0.0007139, 0.0007182
5: 0.0018675, 0.0045491, 0.0017678, 0.0045932, -0.0025122, 0.0025328
6: -0.0025926, -0.0023132, -0.0025996, -0.0023068, -0.0002858, 0.0002864
7: -0.0130839, -0.0086474, -0.0130905, -0.0084029, -0.0046467, 0.0044131
8: -0.0132765, -0.0048291, -0.0134632, -0.0045816, -0.0077207, 0.0078093
9: -0.0018225, 0.0023909, -0.0019374, 0.0024907, -0.0038882, 0.0038160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019691, upper bound: 0.0018794
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019616, upper bound: 0.0018795
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029511, 0.0006903, -0.0029053, 0.0007750, -0.0034341, 0.0032823
1: -0.0046069, -0.0034274, -0.0046063, -0.0033881, -0.0011853, 0.0011354
2: 0.0108685, 0.0157389, 0.0109116, 0.0158690, -0.0045646, 0.0043569
3: 1.0069287, 1.0099781, 1.0068551, 1.0099767, -0.0030481, 0.0031229
4: -0.0041980, -0.0034012, -0.0042222, -0.0034053, -0.0007081, 0.0007426
5: 0.0016969, 0.0045014, 0.0017304, 0.0045680, -0.0026425, 0.0025249
6: -0.0025987, -0.0023036, -0.0025979, -0.0023037, -0.0002950, 0.0002943
7: -0.0130768, -0.0081643, -0.0130868, -0.0083347, -0.0047082, 0.0048920
8: -0.0130745, -0.0044425, -0.0133566, -0.0044688, -0.0076462, 0.0080299
9: -0.0019955, 0.0022830, -0.0019923, 0.0024337, -0.0039712, 0.0037725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
time: 1.32 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029346, 0.0006878, -0.0028569, 0.0008070, -0.0034925, 0.0032542
1: -0.0046023, -0.0034287, -0.0045914, -0.0033710, -0.0012182, 0.0011288
2: 0.0108899, 0.0157349, 0.0109769, 0.0159182, -0.0046589, 0.0043240
3: 1.0069327, 1.0099667, 1.0068119, 1.0099394, -0.0030067, 0.0031549
4: -0.0041972, -0.0034045, -0.0042314, -0.0034159, -0.0007033, 0.0007609
5: 0.0017096, 0.0044994, 0.0017678, 0.0045932, -0.0026888, 0.0025037
6: -0.0025981, -0.0023046, -0.0025996, -0.0023068, -0.0002913, 0.0002951
7: -0.0130765, -0.0081905, -0.0130905, -0.0084029, -0.0046424, 0.0048743
8: -0.0130660, -0.0044786, -0.0134632, -0.0045816, -0.0075974, 0.0082449
9: -0.0019785, 0.0022785, -0.0019374, 0.0024907, -0.0040875, 0.0037501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019331, upper bound: 0.0018549
time: 1.46 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019105, upper bound: 0.0018549
time: 1.90 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027030, 0.0007636, -0.0030985, 0.0007071, -0.0031548, 0.0035442
1: -0.0045544, -0.0033937, -0.0046461, -0.0034189, -0.0011324, 0.0011987
2: 0.0111670, 0.0158515, 0.0106809, 0.0157647, -0.0042150, 0.0046911
3: 1.0068750, 1.0098473, 1.0069065, 1.0100756, -0.0032005, 0.0029408
4: -0.0042189, -0.0034444, -0.0042028, -0.0033723, -0.0007605, 0.0006900
5: 0.0018850, 0.0045591, 0.0015841, 0.0045146, -0.0024293, 0.0027253
6: -0.0025925, -0.0023144, -0.0026031, -0.0022955, -0.0002969, 0.0002887
7: -0.0130854, -0.0086887, -0.0130788, -0.0079112, -0.0051414, 0.0043625
8: -0.0133187, -0.0048739, -0.0131304, -0.0041413, -0.0082071, 0.0074862
9: -0.0018009, 0.0024135, -0.0021397, 0.0023129, -0.0037154, 0.0040485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
time: 1.28 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
time: 1.19 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027030, 0.0007636, -0.0030553, 0.0007377, -0.0032278, 0.0035202
1: -0.0045544, -0.0033937, -0.0046310, -0.0034026, -0.0011518, 0.0011943
2: 0.0111670, 0.0158515, 0.0107404, 0.0158118, -0.0043272, 0.0046668
3: 1.0068750, 1.0098473, 1.0068649, 1.0100383, -0.0031632, 0.0029824
4: -0.0042189, -0.0034444, -0.0042115, -0.0033822, -0.0007573, 0.0007109
5: 0.0018850, 0.0045591, 0.0016176, 0.0045387, -0.0024868, 0.0027076
6: -0.0025925, -0.0023144, -0.0026048, -0.0022986, -0.0002938, 0.0002904
7: -0.0130854, -0.0086887, -0.0130824, -0.0079577, -0.0050945, 0.0043711
8: -0.0133187, -0.0048739, -0.0132325, -0.0042492, -0.0081751, 0.0077294
9: -0.0018009, 0.0024135, -0.0020847, 0.0023675, -0.0038453, 0.0040321

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0026565, 0.0007920, -0.0029346, 0.0006878, -0.0030768, 0.0034554
1: -0.0045406, -0.0033787, -0.0046023, -0.0034287, -0.0010945, 0.0011926
2: 0.0112286, 0.0158951, 0.0108899, 0.0157349, -0.0041031, 0.0046018
3: 1.0068367, 1.0098130, 1.0069327, 1.0099667, -0.0031301, 0.0028802
4: -0.0042271, -0.0034543, -0.0041972, -0.0034045, -0.0007502, 0.0006705
5: 0.0019208, 0.0045814, 0.0017096, 0.0044994, -0.0023687, 0.0026596
6: -0.0025941, -0.0023172, -0.0025981, -0.0023046, -0.0002895, 0.0002809
7: -0.0130887, -0.0087552, -0.0130765, -0.0081905, -0.0048700, 0.0042924
8: -0.0134131, -0.0049791, -0.0130660, -0.0044786, -0.0081212, 0.0072672
9: -0.0017503, 0.0024640, -0.0019785, 0.0022785, -0.0036018, 0.0040214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026565, 0.0007920, -0.0031129, 0.0007149, -0.0031184, 0.0036439
1: -0.0045406, -0.0033787, -0.0046495, -0.0034151, -0.0011172, 0.0012457
2: 0.0112286, 0.0158951, 0.0106638, 0.0157767, -0.0041669, 0.0048436
3: 1.0068367, 1.0098130, 1.0068964, 1.0100840, -0.0032474, 0.0029166
4: -0.0042271, -0.0034543, -0.0042050, -0.0033695, -0.0007892, 0.0006824
5: 0.0019208, 0.0045814, 0.0015733, 0.0045207, -0.0024014, 0.0028036
6: -0.0025941, -0.0023172, -0.0026041, -0.0022949, -0.0002992, 0.0002869
7: -0.0130887, -0.0087552, -0.0130797, -0.0078827, -0.0051782, 0.0042972
8: -0.0134131, -0.0049791, -0.0131564, -0.0041112, -0.0085333, 0.0074055
9: -0.0017503, 0.0024640, -0.0021524, 0.0023268, -0.0036758, 0.0042169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
time: 1.55 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
time: 1.55 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0027416, 0.0007534, -0.0033681, 0.0031910
1: -0.0046147, -0.0033830, -0.0045644, -0.0033981, -0.0011600, 0.0011222
2: 0.0108720, 0.0158846, 0.0111187, 0.0158359, -0.0044825, 0.0042596
3: 1.0068407, 1.0099975, 1.0068822, 1.0098720, -0.0030313, 0.0031153
4: -0.0042251, -0.0033993, -0.0042160, -0.0034370, -0.0006968, 0.0007305
5: 0.0017064, 0.0045760, 0.0018555, 0.0045511, -0.0025919, 0.0024569
6: -0.0025995, -0.0023020, -0.0025932, -0.0023124, -0.0002872, 0.0002912
7: -0.0130880, -0.0082790, -0.0130842, -0.0086208, -0.0044299, 0.0047693
8: -0.0133904, -0.0044056, -0.0132848, -0.0047968, -0.0075580, 0.0078979
9: -0.0020231, 0.0024518, -0.0018377, 0.0023954, -0.0039054, 0.0037521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019767
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019702
time: 1.41 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0029511, 0.0006903, -0.0033311, 0.0034237
1: -0.0046147, -0.0033830, -0.0046069, -0.0034274, -0.0011538, 0.0011765
2: 0.0108720, 0.0158846, 0.0108685, 0.0157389, -0.0044256, 0.0045486
3: 1.0068407, 1.0099975, 1.0069287, 1.0099781, -0.0031374, 0.0030688
4: -0.0042251, -0.0033993, -0.0041980, -0.0034012, -0.0007397, 0.0007199
5: 0.0017064, 0.0045760, 0.0016969, 0.0045014, -0.0025628, 0.0026343
6: -0.0025995, -0.0023020, -0.0025987, -0.0023036, -0.0002959, 0.0002967
7: -0.0130880, -0.0082790, -0.0130768, -0.0081643, -0.0048908, 0.0047650
8: -0.0133904, -0.0044056, -0.0130745, -0.0044425, -0.0079952, 0.0077747
9: -0.0020231, 0.0024518, -0.0019955, 0.0022830, -0.0038395, 0.0039526

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019767
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019702
time: 1.28 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0027030, 0.0007636, -0.0035871, 0.0031440
1: -0.0046538, -0.0034138, -0.0045544, -0.0033937, -0.0012134, 0.0011232
2: 0.0106433, 0.0157805, 0.0111670, 0.0158515, -0.0047508, 0.0041983
3: 1.0068923, 1.0100950, 1.0068750, 1.0098473, -0.0029550, 0.0032200
4: -0.0042057, -0.0033664, -0.0042189, -0.0034444, -0.0006869, 0.0007713
5: 0.0015609, 0.0045227, 0.0018850, 0.0045591, -0.0027583, 0.0024208
6: -0.0026047, -0.0022940, -0.0025925, -0.0023144, -0.0002903, 0.0002985
7: -0.0130800, -0.0078570, -0.0130854, -0.0086887, -0.0043613, 0.0051954
8: -0.0131647, -0.0040777, -0.0133187, -0.0048739, -0.0074500, 0.0083220
9: -0.0021684, 0.0023312, -0.0018009, 0.0024135, -0.0041024, 0.0036960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019701
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019702
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031129, 0.0007149, -0.0026565, 0.0007920, -0.0036439, 0.0031184
1: -0.0046495, -0.0034151, -0.0045406, -0.0033787, -0.0012457, 0.0011172
2: 0.0106638, 0.0157767, 0.0112286, 0.0158951, -0.0048436, 0.0041669
3: 1.0068964, 1.0100840, 1.0068367, 1.0098130, -0.0029166, 0.0032474
4: -0.0042050, -0.0033695, -0.0042271, -0.0034543, -0.0006824, 0.0007892
5: 0.0015733, 0.0045207, 0.0019208, 0.0045814, -0.0028036, 0.0024014
6: -0.0026041, -0.0022949, -0.0025941, -0.0023172, -0.0002869, 0.0002992
7: -0.0130797, -0.0078827, -0.0130887, -0.0087552, -0.0042972, 0.0051782
8: -0.0131564, -0.0041112, -0.0134131, -0.0049791, -0.0074055, 0.0085333
9: -0.0021524, 0.0023268, -0.0017503, 0.0024640, -0.0042169, 0.0036758

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019701
time: 1.44 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019702
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0029369, 0.0007851, -0.0032896, 0.0032896
1: -0.0046147, -0.0033830, -0.0046147, -0.0033830, -0.0011311, 0.0011311
2: 0.0108720, 0.0158846, 0.0108720, 0.0158846, -0.0043559, 0.0043559
3: 1.0068407, 1.0099975, 1.0068407, 1.0099975, -0.0031568, 0.0031568
4: -0.0042251, -0.0033993, -0.0042251, -0.0033993, -0.0007063, 0.0007063
5: 0.0017064, 0.0045760, 0.0017064, 0.0045760, -0.0025300, 0.0025300
6: -0.0025995, -0.0023020, -0.0025995, -0.0023020, -0.0002975, 0.0002975
7: -0.0130880, -0.0082790, -0.0130880, -0.0082790, -0.0047617, 0.0047617
8: -0.0133904, -0.0044056, -0.0133904, -0.0044056, -0.0076206, 0.0076206
9: -0.0020231, 0.0024518, -0.0020231, 0.0024518, -0.0037603, 0.0037603

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019233, upper bound: 0.0019798
time: 1.76 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019237, upper bound: 0.0019736
time: 1.40 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0031289, 0.0007174, -0.0032533, 0.0035083
1: -0.0046147, -0.0033830, -0.0046538, -0.0034138, -0.0011256, 0.0011851
2: 0.0108720, 0.0158846, 0.0106433, 0.0157805, -0.0043001, 0.0046310
3: 1.0068407, 1.0099975, 1.0068923, 1.0100950, -0.0032543, 0.0031052
4: -0.0042251, -0.0033993, -0.0042057, -0.0033664, -0.0007479, 0.0006959
5: 0.0017064, 0.0045760, 0.0015609, 0.0045227, -0.0025014, 0.0026965
6: -0.0025995, -0.0023020, -0.0026047, -0.0022940, -0.0003056, 0.0003027
7: -0.0130880, -0.0082790, -0.0130800, -0.0078570, -0.0051870, 0.0047575
8: -0.0133904, -0.0044056, -0.0131647, -0.0040777, -0.0080487, 0.0074996
9: -0.0020231, 0.0024518, -0.0021684, 0.0023312, -0.0036957, 0.0039594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019233, upper bound: 0.0019799
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019237, upper bound: 0.0019736
time: 1.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0028919, 0.0007932, -0.0035258, 0.0032303
1: -0.0046538, -0.0034138, -0.0046032, -0.0033791, -0.0011934, 0.0011288
2: 0.0106433, 0.0157805, 0.0109284, 0.0158970, -0.0046580, 0.0042775
3: 1.0068923, 1.0100950, 1.0068345, 1.0099689, -0.0030766, 0.0032605
4: -0.0042057, -0.0033664, -0.0042274, -0.0034079, -0.0006936, 0.0007529
5: 0.0015609, 0.0045227, 0.0017406, 0.0045823, -0.0027103, 0.0024843
6: -0.0026047, -0.0022940, -0.0025985, -0.0023044, -0.0003003, 0.0003045
7: -0.0130800, -0.0078570, -0.0130889, -0.0083606, -0.0046791, 0.0051891
8: -0.0131647, -0.0040777, -0.0134171, -0.0044944, -0.0074822, 0.0081073
9: -0.0021684, 0.0023312, -0.0019807, 0.0024661, -0.0039907, 0.0036900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
time: 1.77 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
time: 1.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031129, 0.0007149, -0.0028443, 0.0008248, -0.0035839, 0.0032021
1: -0.0046495, -0.0034151, -0.0045890, -0.0033622, -0.0012244, 0.0011218
2: 0.0106638, 0.0157767, 0.0109934, 0.0159456, -0.0047519, 0.0042427
3: 1.0068964, 1.0100840, 1.0067921, 1.0099336, -0.0030372, 0.0032920
4: -0.0042050, -0.0033695, -0.0042365, -0.0034185, -0.0006884, 0.0007710
5: 0.0015733, 0.0045207, 0.0017774, 0.0046072, -0.0027564, 0.0024629
6: -0.0026041, -0.0022949, -0.0026003, -0.0023073, -0.0002968, 0.0003054
7: -0.0130797, -0.0078827, -0.0130926, -0.0084257, -0.0046151, 0.0051719
8: -0.0131564, -0.0041112, -0.0135226, -0.0046067, -0.0074301, 0.0083212
9: -0.0021524, 0.0023268, -0.0019285, 0.0025225, -0.0041064, 0.0036659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
time: 1.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
time: 1.76 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027416, 0.0007534, -0.0029587, 0.0008240, -0.0032624, 0.0033925
1: -0.0045644, -0.0033981, -0.0046208, -0.0033636, -0.0011567, 0.0011670
2: 0.0111187, 0.0158359, 0.0108450, 0.0159443, -0.0043694, 0.0045129
3: 1.0068822, 1.0098720, 1.0067888, 1.0100127, -0.0031306, 0.0030831
4: -0.0042160, -0.0034370, -0.0042362, -0.0033949, -0.0007353, 0.0007172
5: 0.0018555, 0.0045511, 0.0016899, 0.0046066, -0.0025131, 0.0026107
6: -0.0025932, -0.0023124, -0.0026028, -0.0023007, -0.0002924, 0.0002904
7: -0.0130842, -0.0086208, -0.0130925, -0.0082499, -0.0047979, 0.0044383
8: -0.0132848, -0.0047968, -0.0135198, -0.0043573, -0.0079535, 0.0077960
9: -0.0018377, 0.0023954, -0.0020457, 0.0025209, -0.0038793, 0.0039312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019969, upper bound: 0.0019303
time: 1.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019969, upper bound: 0.0019303
time: 1.60 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027260, 0.0007509, -0.0029146, 0.0008569, -0.0033158, 0.0033651
1: -0.0045602, -0.0033993, -0.0046055, -0.0033467, -0.0011867, 0.0011598
2: 0.0111387, 0.0158321, 0.0109051, 0.0159950, -0.0044560, 0.0044792
3: 1.0068861, 1.0098618, 1.0067493, 1.0099745, -0.0030884, 0.0031126
4: -0.0042153, -0.0034402, -0.0042457, -0.0034052, -0.0007302, 0.0007339
5: 0.0018675, 0.0045491, 0.0017238, 0.0046325, -0.0025556, 0.0025897
6: -0.0025926, -0.0023132, -0.0026044, -0.0023039, -0.0002887, 0.0002912
7: -0.0130839, -0.0086474, -0.0130964, -0.0083032, -0.0047475, 0.0044196
8: -0.0132765, -0.0048291, -0.0136295, -0.0044711, -0.0079013, 0.0079929
9: -0.0018225, 0.0023909, -0.0019894, 0.0025796, -0.0039863, 0.0039066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0018942
time: 1.47 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019611, upper bound: 0.0018942
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0029511, 0.0006903, -0.0029587, 0.0008240, -0.0034951, 0.0033555
1: -0.0046069, -0.0034274, -0.0046208, -0.0033636, -0.0012110, 0.0011608
2: 0.0108685, 0.0157389, 0.0108450, 0.0159443, -0.0046584, 0.0044561
3: 1.0069287, 1.0099781, 1.0067888, 1.0100127, -0.0030841, 0.0031892
4: -0.0041980, -0.0034012, -0.0042362, -0.0033949, -0.0007247, 0.0007601
5: 0.0016969, 0.0045014, 0.0016899, 0.0046066, -0.0026905, 0.0025816
6: -0.0025987, -0.0023036, -0.0026028, -0.0023007, -0.0002980, 0.0002992
7: -0.0130768, -0.0081643, -0.0130925, -0.0082499, -0.0047935, 0.0048991
8: -0.0130745, -0.0044425, -0.0135198, -0.0043573, -0.0078303, 0.0082331
9: -0.0019955, 0.0022830, -0.0020457, 0.0025209, -0.0040798, 0.0038654

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019160
time: 1.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019160
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0029346, 0.0006878, -0.0029146, 0.0008569, -0.0035476, 0.0033281
1: -0.0046023, -0.0034287, -0.0046055, -0.0033467, -0.0012407, 0.0011535
2: 0.0108899, 0.0157349, 0.0109051, 0.0159950, -0.0047436, 0.0044224
3: 1.0069327, 1.0099667, 1.0067493, 1.0099745, -0.0030417, 0.0032175
4: -0.0041972, -0.0034045, -0.0042457, -0.0034052, -0.0007196, 0.0007767
5: 0.0017096, 0.0044994, 0.0017238, 0.0046325, -0.0027321, 0.0025606
6: -0.0025981, -0.0023046, -0.0026044, -0.0023039, -0.0002942, 0.0002999
7: -0.0130765, -0.0081905, -0.0130964, -0.0083032, -0.0047431, 0.0048808
8: -0.0130660, -0.0044786, -0.0136295, -0.0044711, -0.0077780, 0.0084286
9: -0.0019785, 0.0022785, -0.0019894, 0.0025796, -0.0041856, 0.0038407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019328, upper bound: 0.0018704
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019105, upper bound: 0.0018703
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027030, 0.0007636, -0.0031789, 0.0007623, -0.0032182, 0.0036497
1: -0.0045544, -0.0033937, -0.0046634, -0.0033907, -0.0011579, 0.0012348
2: 0.0111670, 0.0158515, 0.0105843, 0.0158495, -0.0043123, 0.0048343
3: 1.0068750, 1.0098473, 1.0068295, 1.0101190, -0.0032439, 0.0030178
4: -0.0042189, -0.0034444, -0.0042186, -0.0033579, -0.0007851, 0.0007081
5: 0.0018850, 0.0045591, 0.0015231, 0.0045580, -0.0024792, 0.0028065
6: -0.0025925, -0.0023144, -0.0026091, -0.0022920, -0.0003005, 0.0002947
7: -0.0130854, -0.0086887, -0.0130853, -0.0077544, -0.0052987, 0.0043700
8: -0.0133187, -0.0048739, -0.0133143, -0.0039967, -0.0084766, 0.0076972
9: -0.0018009, 0.0024135, -0.0022040, 0.0024112, -0.0038281, 0.0041807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
time: 1.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027030, 0.0007636, -0.0031356, 0.0007910, -0.0032848, 0.0036226
1: -0.0045544, -0.0033937, -0.0046491, -0.0033753, -0.0011791, 0.0012275
2: 0.0111670, 0.0158515, 0.0106404, 0.0158936, -0.0044148, 0.0048035
3: 1.0068750, 1.0098473, 1.0067912, 1.0100833, -0.0032083, 0.0030560
4: -0.0042189, -0.0034444, -0.0042268, -0.0033674, -0.0007798, 0.0007272
5: 0.0018850, 0.0045591, 0.0015564, 0.0045806, -0.0025316, 0.0027863
6: -0.0025925, -0.0023144, -0.0026104, -0.0022949, -0.0002975, 0.0002960
7: -0.0130854, -0.0086887, -0.0130886, -0.0078121, -0.0052407, 0.0043778
8: -0.0133187, -0.0048739, -0.0134099, -0.0041007, -0.0084189, 0.0079192
9: -0.0018009, 0.0024135, -0.0021517, 0.0024622, -0.0039468, 0.0041539

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
time: 1.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0026565, 0.0007920, -0.0031789, 0.0007623, -0.0031929, 0.0037209
1: -0.0045406, -0.0033787, -0.0046634, -0.0033907, -0.0011499, 0.0012703
2: 0.0112286, 0.0158951, 0.0105843, 0.0158495, -0.0042847, 0.0049438
3: 1.0068367, 1.0098130, 1.0068295, 1.0101190, -0.0032823, 0.0029835
4: -0.0042271, -0.0034543, -0.0042186, -0.0033579, -0.0008055, 0.0007046
5: 0.0019208, 0.0045814, 0.0015231, 0.0045580, -0.0024602, 0.0028625
6: -0.0025941, -0.0023172, -0.0026091, -0.0022920, -0.0003021, 0.0002919
7: -0.0130887, -0.0087552, -0.0130853, -0.0077544, -0.0053070, 0.0043049
8: -0.0134131, -0.0049791, -0.0133143, -0.0039967, -0.0087138, 0.0076633
9: -0.0017503, 0.0024640, -0.0022040, 0.0024112, -0.0038124, 0.0043075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
time: 1.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0026565, 0.0007920, -0.0031356, 0.0007910, -0.0032210, 0.0036514
1: -0.0045406, -0.0033787, -0.0046491, -0.0033753, -0.0011653, 0.0012422
2: 0.0112286, 0.0158951, 0.0106404, 0.0158936, -0.0043246, 0.0048453
3: 1.0068367, 1.0098130, 1.0067912, 1.0100833, -0.0032467, 0.0030217
4: -0.0042271, -0.0034543, -0.0042268, -0.0033674, -0.0007884, 0.0007118
5: 0.0019208, 0.0045814, 0.0015564, 0.0045806, -0.0024821, 0.0028089
6: -0.0025941, -0.0023172, -0.0026104, -0.0022949, -0.0002991, 0.0002932
7: -0.0130887, -0.0087552, -0.0130886, -0.0078121, -0.0052448, 0.0043093
8: -0.0134131, -0.0049791, -0.0134099, -0.0041007, -0.0085225, 0.0077473
9: -0.0017503, 0.0024640, -0.0021517, 0.0024622, -0.0038584, 0.0042092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
time: 1.50 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
time: 1.57 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0028177, 0.0008106, -0.0034367, 0.0033153
1: -0.0046147, -0.0033830, -0.0045833, -0.0033700, -0.0011890, 0.0011579
2: 0.0108720, 0.0158846, 0.0110251, 0.0159238, -0.0045880, 0.0044362
3: 1.0068407, 1.0099975, 1.0068079, 1.0099194, -0.0030787, 0.0031896
4: -0.0042251, -0.0033993, -0.0042324, -0.0034227, -0.0007268, 0.0007501
5: 0.0017064, 0.0045760, 0.0017977, 0.0045961, -0.0026459, 0.0025532
6: -0.0025995, -0.0023020, -0.0025992, -0.0023085, -0.0002911, 0.0002972
7: -0.0130880, -0.0082790, -0.0130909, -0.0084878, -0.0045670, 0.0047774
8: -0.0133904, -0.0044056, -0.0134753, -0.0046476, -0.0078825, 0.0081267
9: -0.0020231, 0.0024518, -0.0019077, 0.0024972, -0.0040276, 0.0039085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019937
time: 1.48 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019879
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0030486, 0.0007533, -0.0034157, 0.0035699
1: -0.0046147, -0.0033830, -0.0046284, -0.0033956, -0.0011881, 0.0012211
2: 0.0108720, 0.0158846, 0.0107505, 0.0158357, -0.0045557, 0.0047503
3: 1.0068407, 1.0099975, 1.0068442, 1.0100317, -0.0031910, 0.0031533
4: -0.0042251, -0.0033993, -0.0042160, -0.0033837, -0.0007744, 0.0007441
5: 0.0017064, 0.0045760, 0.0016228, 0.0045509, -0.0026294, 0.0027471
6: -0.0025995, -0.0023020, -0.0026057, -0.0022992, -0.0003003, 0.0003037
7: -0.0130880, -0.0082790, -0.0130842, -0.0079707, -0.0050870, 0.0047749
8: -0.0133904, -0.0044056, -0.0132843, -0.0042660, -0.0083765, 0.0080566
9: -0.0020231, 0.0024518, -0.0020752, 0.0023951, -0.0039901, 0.0041415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019937
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019879
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0027651, 0.0008239, -0.0036641, 0.0032458
1: -0.0046538, -0.0034138, -0.0045701, -0.0033639, -0.0012450, 0.0011465
2: 0.0106433, 0.0157805, 0.0110910, 0.0159442, -0.0048693, 0.0043395
3: 1.0068923, 1.0100950, 1.0067976, 1.0098865, -0.0029942, 0.0032974
4: -0.0042057, -0.0033664, -0.0042362, -0.0034327, -0.0007102, 0.0007934
5: 0.0015609, 0.0045227, 0.0018377, 0.0046065, -0.0028190, 0.0024996
6: -0.0026047, -0.0022940, -0.0025983, -0.0023112, -0.0002935, 0.0003044
7: -0.0130800, -0.0078570, -0.0130925, -0.0085845, -0.0044680, 0.0052044
8: -0.0131647, -0.0040777, -0.0135195, -0.0047509, -0.0076963, 0.0085788
9: -0.0021684, 0.0023312, -0.0018590, 0.0025208, -0.0042396, 0.0038117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019880
time: 1.34 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031129, 0.0007149, -0.0027162, 0.0008501, -0.0037128, 0.0032193
1: -0.0046495, -0.0034151, -0.0045544, -0.0033499, -0.0012729, 0.0011393
2: 0.0106638, 0.0157767, 0.0111583, 0.0159846, -0.0049494, 0.0043083
3: 1.0068964, 1.0100840, 1.0067619, 1.0098473, -0.0029509, 0.0033221
4: -0.0042050, -0.0033695, -0.0042437, -0.0034438, -0.0007059, 0.0008089
5: 0.0015733, 0.0045207, 0.0018755, 0.0046272, -0.0028577, 0.0024796
6: -0.0026041, -0.0022949, -0.0025999, -0.0023144, -0.0002897, 0.0003051
7: -0.0130797, -0.0078827, -0.0130956, -0.0086468, -0.0044081, 0.0051862
8: -0.0131564, -0.0041112, -0.0136070, -0.0048698, -0.0076541, 0.0087626
9: -0.0021524, 0.0023268, -0.0018011, 0.0025676, -0.0043395, 0.0037946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
time: 1.46 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0029951, 0.0008336, -0.0033581, 0.0034032
1: -0.0046147, -0.0033830, -0.0046287, -0.0033587, -0.0011586, 0.0011696
2: 0.0108720, 0.0158846, 0.0107999, 0.0159591, -0.0044611, 0.0045237
3: 1.0068407, 1.0099975, 1.0067751, 1.0100324, -0.0031917, 0.0032223
4: -0.0042251, -0.0033993, -0.0042390, -0.0033883, -0.0007362, 0.0007259
5: 0.0017064, 0.0045760, 0.0016620, 0.0046141, -0.0025839, 0.0026181
6: -0.0025995, -0.0023020, -0.0026045, -0.0022991, -0.0003004, 0.0003025
7: -0.0130880, -0.0082790, -0.0130936, -0.0081818, -0.0048636, 0.0047698
8: -0.0133904, -0.0044056, -0.0135519, -0.0042928, -0.0079511, 0.0078487
9: -0.0020231, 0.0024518, -0.0020749, 0.0025381, -0.0038822, 0.0039239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019978
time: 1.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019921
time: 3.19 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029369, 0.0007851, -0.0032084, 0.0007720, -0.0033387, 0.0036450
1: -0.0046147, -0.0033830, -0.0046698, -0.0033859, -0.0011578, 0.0012323
2: 0.0108720, 0.0158846, 0.0105463, 0.0158645, -0.0044312, 0.0048257
3: 1.0068407, 1.0099975, 1.0068165, 1.0101347, -0.0032940, 0.0031810
4: -0.0042251, -0.0033993, -0.0042213, -0.0033526, -0.0007825, 0.0007203
5: 0.0017064, 0.0045760, 0.0015005, 0.0045657, -0.0025686, 0.0028027
6: -0.0025995, -0.0023020, -0.0026105, -0.0022907, -0.0003088, 0.0003085
7: -0.0130880, -0.0082790, -0.0130864, -0.0077095, -0.0053393, 0.0047675
8: -0.0133904, -0.0044056, -0.0133467, -0.0039441, -0.0084361, 0.0077839
9: -0.0020231, 0.0024518, -0.0022276, 0.0024285, -0.0038476, 0.0041549

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019978
time: 1.72 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019922
time: 1.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0031289, 0.0007174, -0.0029408, 0.0008447, -0.0036021, 0.0033315
1: -0.0046538, -0.0034138, -0.0046157, -0.0033533, -0.0012235, 0.0011572
2: 0.0106433, 0.0157805, 0.0108674, 0.0159762, -0.0047753, 0.0044240
3: 1.0068923, 1.0100950, 1.0067655, 1.0100001, -0.0031078, 0.0033295
4: -0.0042057, -0.0033664, -0.0042422, -0.0033984, -0.0007191, 0.0007747
5: 0.0015609, 0.0045227, 0.0017034, 0.0046229, -0.0027704, 0.0025632
6: -0.0026047, -0.0022940, -0.0026034, -0.0023018, -0.0003029, 0.0003095
7: -0.0130800, -0.0078570, -0.0130949, -0.0082812, -0.0047622, 0.0051981
8: -0.0131647, -0.0040777, -0.0135889, -0.0043953, -0.0077586, 0.0083615
9: -0.0021684, 0.0023312, -0.0020270, 0.0025579, -0.0041265, 0.0038234

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
time: 1.66 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031129, 0.0007149, -0.0028984, 0.0008765, -0.0036515, 0.0033068
1: -0.0046495, -0.0034151, -0.0046011, -0.0033368, -0.0012509, 0.0011520
2: 0.0106638, 0.0157767, 0.0109251, 0.0160251, -0.0048559, 0.0043947
3: 1.0068964, 1.0100840, 1.0067252, 1.0099635, -0.0030671, 0.0033588
4: -0.0042050, -0.0033695, -0.0042513, -0.0034083, -0.0007150, 0.0007904
5: 0.0015733, 0.0045207, 0.0017360, 0.0046479, -0.0028096, 0.0025444
6: -0.0026041, -0.0022949, -0.0026052, -0.0023048, -0.0002993, 0.0003103
7: -0.0130797, -0.0078827, -0.0130987, -0.0083349, -0.0047098, 0.0051799
8: -0.0131564, -0.0041112, -0.0136947, -0.0045040, -0.0077198, 0.0085465
9: -0.0021524, 0.0023268, -0.0019733, 0.0026144, -0.0042268, 0.0038082

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
time: 1.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
time: 1.62 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028177, 0.0008106, -0.0029053, 0.0007750, -0.0033237, 0.0033879
1: -0.0045833, -0.0033700, -0.0046063, -0.0033881, -0.0011646, 0.0011707
2: 0.0110251, 0.0159238, 0.0109116, 0.0158690, -0.0044490, 0.0045193
3: 1.0068079, 1.0099194, 1.0068551, 1.0099767, -0.0031688, 0.0030643
4: -0.0042324, -0.0034227, -0.0042222, -0.0034053, -0.0007384, 0.0007292
5: 0.0017977, 0.0045961, 0.0017304, 0.0045680, -0.0025598, 0.0026081
6: -0.0025992, -0.0023085, -0.0025979, -0.0023037, -0.0002955, 0.0002895
7: -0.0130909, -0.0084878, -0.0130868, -0.0083347, -0.0047205, 0.0045680
8: -0.0134753, -0.0046476, -0.0133566, -0.0044688, -0.0079982, 0.0079104
9: -0.0019077, 0.0024972, -0.0019923, 0.0024337, -0.0039234, 0.0039606

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019179
time: 1.21 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019180
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027997, 0.0008080, -0.0028569, 0.0008070, -0.0033798, 0.0033597
1: -0.0045784, -0.0033713, -0.0045914, -0.0033710, -0.0011952, 0.0011639
2: 0.0110485, 0.0159198, 0.0109769, 0.0159182, -0.0045382, 0.0044862
3: 1.0068123, 1.0099072, 1.0068119, 1.0099394, -0.0031271, 0.0030954
4: -0.0042317, -0.0034264, -0.0042314, -0.0034159, -0.0007335, 0.0007464
5: 0.0018115, 0.0045940, 0.0017678, 0.0045932, -0.0026044, 0.0025867
6: -0.0025985, -0.0023095, -0.0025996, -0.0023068, -0.0002917, 0.0002902
7: -0.0130906, -0.0085186, -0.0130905, -0.0084029, -0.0046548, 0.0045464
8: -0.0134667, -0.0046857, -0.0134632, -0.0045816, -0.0079489, 0.0081141
9: -0.0018896, 0.0024926, -0.0019374, 0.0024907, -0.0040343, 0.0039380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019793, upper bound: 0.0018794
time: 1.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019700, upper bound: 0.0018794
time: 1.34 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0030486, 0.0007533, -0.0029053, 0.0007750, -0.0035783, 0.0033668
1: -0.0046284, -0.0033956, -0.0046063, -0.0033881, -0.0012278, 0.0011697
2: 0.0107505, 0.0158357, 0.0109116, 0.0158690, -0.0047632, 0.0044870
3: 1.0068442, 1.0100317, 1.0068551, 1.0099767, -0.0031326, 0.0031766
4: -0.0042160, -0.0033837, -0.0042222, -0.0034053, -0.0007323, 0.0007768
5: 0.0016228, 0.0045509, 0.0017304, 0.0045680, -0.0027537, 0.0025915
6: -0.0026057, -0.0022992, -0.0025979, -0.0023037, -0.0003020, 0.0002987
7: -0.0130842, -0.0079707, -0.0130868, -0.0083347, -0.0047181, 0.0050880
8: -0.0132843, -0.0042660, -0.0133566, -0.0044688, -0.0079281, 0.0084045
9: -0.0020752, 0.0023951, -0.0019923, 0.0024337, -0.0041564, 0.0039231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019880, upper bound: 0.0019062
time: 1.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019880, upper bound: 0.0019062
time: 1.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0030306, 0.0007507, -0.0028569, 0.0008070, -0.0036349, 0.0033387
1: -0.0046234, -0.0033969, -0.0045914, -0.0033710, -0.0012524, 0.0011629
2: 0.0107735, 0.0158317, 0.0109769, 0.0159182, -0.0048539, 0.0044539
3: 1.0068485, 1.0100193, 1.0068119, 1.0099394, -0.0030910, 0.0032074
4: -0.0042152, -0.0033873, -0.0042314, -0.0034159, -0.0007275, 0.0007941
5: 0.0016366, 0.0045489, 0.0017678, 0.0045932, -0.0027986, 0.0025701
6: -0.0026050, -0.0023002, -0.0025996, -0.0023068, -0.0002982, 0.0002994
7: -0.0130839, -0.0080011, -0.0130905, -0.0084029, -0.0046523, 0.0050664
8: -0.0132756, -0.0043037, -0.0134632, -0.0045816, -0.0078788, 0.0086092
9: -0.0020569, 0.0023905, -0.0019374, 0.0024907, -0.0042675, 0.0039005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019527, upper bound: 0.0018548
time: 1.55 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019360, upper bound: 0.0018548
time: 1.56 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027651, 0.0008239, -0.0030985, 0.0007071, -0.0032542, 0.0036213
1: -0.0045701, -0.0033639, -0.0046461, -0.0034189, -0.0011512, 0.0012303
2: 0.0110910, 0.0159442, 0.0106809, 0.0157647, -0.0043525, 0.0048096
3: 1.0067976, 1.0098865, 1.0069065, 1.0100756, -0.0032780, 0.0029800
4: -0.0042362, -0.0034327, -0.0042028, -0.0033723, -0.0007826, 0.0007127
5: 0.0018377, 0.0046065, 0.0015841, 0.0045146, -0.0025062, 0.0027860
6: -0.0025983, -0.0023112, -0.0026031, -0.0022955, -0.0003028, 0.0002920
7: -0.0130925, -0.0085845, -0.0130788, -0.0079112, -0.0051505, 0.0044690
8: -0.0135195, -0.0047509, -0.0131304, -0.0041413, -0.0084639, 0.0077244
9: -0.0018590, 0.0025208, -0.0021397, 0.0023129, -0.0038268, 0.0041857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
time: 1.43 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027651, 0.0008239, -0.0030553, 0.0007377, -0.0033253, 0.0035973
1: -0.0045701, -0.0033639, -0.0046310, -0.0034026, -0.0011675, 0.0012259
2: 0.0110910, 0.0159442, 0.0107404, 0.0158118, -0.0044619, 0.0047854
3: 1.0067976, 1.0098865, 1.0068649, 1.0100383, -0.0032407, 0.0030216
4: -0.0042362, -0.0034327, -0.0042115, -0.0033822, -0.0007794, 0.0007330
5: 0.0018377, 0.0046065, 0.0016176, 0.0045387, -0.0025622, 0.0027683
6: -0.0025983, -0.0023112, -0.0026048, -0.0022986, -0.0002997, 0.0002936
7: -0.0130925, -0.0085845, -0.0130824, -0.0079577, -0.0051036, 0.0044773
8: -0.0135195, -0.0047509, -0.0132325, -0.0042492, -0.0084319, 0.0079614
9: -0.0018590, 0.0025208, -0.0020847, 0.0023675, -0.0039534, 0.0041694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
time: 1.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027162, 0.0008501, -0.0029346, 0.0006878, -0.0031786, 0.0035242
1: -0.0045544, -0.0033499, -0.0046023, -0.0034287, -0.0011189, 0.0012198
2: 0.0111583, 0.0159846, 0.0108899, 0.0157349, -0.0042457, 0.0047076
3: 1.0067619, 1.0098473, 1.0069327, 1.0099667, -0.0032048, 0.0029145
4: -0.0042437, -0.0034438, -0.0041972, -0.0034045, -0.0007699, 0.0006942
5: 0.0018755, 0.0046272, 0.0017096, 0.0044994, -0.0024475, 0.0027137
6: -0.0025999, -0.0023144, -0.0025981, -0.0023046, -0.0002954, 0.0002837
7: -0.0130956, -0.0086468, -0.0130765, -0.0081905, -0.0048781, 0.0044033
8: -0.0136070, -0.0048698, -0.0130660, -0.0044786, -0.0083506, 0.0075185
9: -0.0018011, 0.0025676, -0.0019785, 0.0022785, -0.0037222, 0.0041439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
time: 1.50 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
time: 1.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027162, 0.0008501, -0.0031129, 0.0007149, -0.0032193, 0.0037128
1: -0.0045544, -0.0033499, -0.0046495, -0.0034151, -0.0011393, 0.0012729
2: 0.0111583, 0.0159846, 0.0106638, 0.0157767, -0.0043083, 0.0049494
3: 1.0067619, 1.0098473, 1.0068964, 1.0100840, -0.0033221, 0.0029509
4: -0.0042437, -0.0034438, -0.0042050, -0.0033695, -0.0008089, 0.0007059
5: 0.0018755, 0.0046272, 0.0015733, 0.0045207, -0.0024796, 0.0028577
6: -0.0025999, -0.0023144, -0.0026041, -0.0022949, -0.0003051, 0.0002897
7: -0.0130956, -0.0086468, -0.0130797, -0.0078827, -0.0051862, 0.0044081
8: -0.0136070, -0.0048698, -0.0131564, -0.0041112, -0.0087627, 0.0076541
9: -0.0018011, 0.0025676, -0.0021524, 0.0023268, -0.0037946, 0.0043395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
time: 1.77 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
time: 1.65 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0027416, 0.0007534, -0.0034537, 0.0032531
1: -0.0046287, -0.0033587, -0.0045644, -0.0033981, -0.0011868, 0.0011490
2: 0.0107999, 0.0159591, 0.0111187, 0.0158359, -0.0046033, 0.0043551
3: 1.0067751, 1.0100324, 1.0068822, 1.0098720, -0.0030968, 0.0031502
4: -0.0042390, -0.0033883, -0.0042160, -0.0034370, -0.0007146, 0.0007517
5: 0.0016620, 0.0046141, 0.0018555, 0.0045511, -0.0026581, 0.0025058
6: -0.0026045, -0.0022991, -0.0025932, -0.0023124, -0.0002922, 0.0002941
7: -0.0130936, -0.0081818, -0.0130842, -0.0086208, -0.0044372, 0.0048701
8: -0.0135519, -0.0042928, -0.0132848, -0.0047968, -0.0077649, 0.0081385
9: -0.0020749, 0.0025381, -0.0018377, 0.0023954, -0.0040252, 0.0038627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019160, upper bound: 0.0019756
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019173, upper bound: 0.0019685
time: 1.61 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0029511, 0.0006903, -0.0034161, 0.0034858
1: -0.0046287, -0.0033587, -0.0046069, -0.0034274, -0.0011796, 0.0012033
2: 0.0107999, 0.0159591, 0.0108685, 0.0157389, -0.0045454, 0.0046441
3: 1.0067751, 1.0100324, 1.0069287, 1.0099781, -0.0032029, 0.0031037
4: -0.0042390, -0.0033883, -0.0041980, -0.0034012, -0.0007574, 0.0007409
5: 0.0016620, 0.0046141, 0.0016969, 0.0045014, -0.0026285, 0.0026832
6: -0.0026045, -0.0022991, -0.0025987, -0.0023036, -0.0003009, 0.0002996
7: -0.0130936, -0.0081818, -0.0130768, -0.0081643, -0.0048980, 0.0048656
8: -0.0135519, -0.0042928, -0.0130745, -0.0044425, -0.0082021, 0.0080132
9: -0.0020749, 0.0025381, -0.0019955, 0.0022830, -0.0039583, 0.0040632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019160, upper bound: 0.0019756
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019173, upper bound: 0.0019685
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0027030, 0.0007636, -0.0037079, 0.0032084
1: -0.0046698, -0.0033859, -0.0045544, -0.0033937, -0.0012470, 0.0011493
2: 0.0105463, 0.0158645, 0.0111670, 0.0158515, -0.0049209, 0.0042973
3: 1.0068165, 1.0101347, 1.0068750, 1.0098473, -0.0030308, 0.0032597
4: -0.0042213, -0.0033526, -0.0042189, -0.0034444, -0.0007053, 0.0008005
5: 0.0015005, 0.0045657, 0.0018850, 0.0045591, -0.0028523, 0.0024715
6: -0.0026105, -0.0022907, -0.0025925, -0.0023144, -0.0002961, 0.0003018
7: -0.0130864, -0.0077095, -0.0130854, -0.0086887, -0.0043688, 0.0053456
8: -0.0133467, -0.0039441, -0.0133187, -0.0048739, -0.0076646, 0.0086442
9: -0.0022276, 0.0024285, -0.0018009, 0.0024135, -0.0042637, 0.0038107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019687
time: 1.74 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019688
time: 1.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031911, 0.0007694, -0.0026565, 0.0007920, -0.0037604, 0.0031826
1: -0.0046650, -0.0033872, -0.0045406, -0.0033787, -0.0012766, 0.0011430
2: 0.0105683, 0.0158605, 0.0112286, 0.0158951, -0.0050057, 0.0042657
3: 1.0068207, 1.0101230, 1.0068367, 1.0098130, -0.0029923, 0.0032864
4: -0.0042206, -0.0033560, -0.0042271, -0.0034543, -0.0007008, 0.0008169
5: 0.0015137, 0.0045637, 0.0019208, 0.0045814, -0.0028939, 0.0024520
6: -0.0026098, -0.0022916, -0.0025941, -0.0023172, -0.0002926, 0.0003024
7: -0.0130861, -0.0077392, -0.0130887, -0.0087552, -0.0043048, 0.0053236
8: -0.0133381, -0.0039803, -0.0134131, -0.0049791, -0.0076196, 0.0088371
9: -0.0022101, 0.0024239, -0.0017503, 0.0024640, -0.0043678, 0.0037902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019688
time: 1.36 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019688
time: 1.71 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0029369, 0.0007851, -0.0034032, 0.0033581
1: -0.0046287, -0.0033587, -0.0046147, -0.0033830, -0.0011696, 0.0011586
2: 0.0107999, 0.0159591, 0.0108720, 0.0158846, -0.0045237, 0.0044611
3: 1.0067751, 1.0100324, 1.0068407, 1.0099975, -0.0032223, 0.0031917
4: -0.0042390, -0.0033883, -0.0042251, -0.0033993, -0.0007259, 0.0007362
5: 0.0016620, 0.0046141, 0.0017064, 0.0045760, -0.0026181, 0.0025839
6: -0.0026045, -0.0022991, -0.0025995, -0.0023020, -0.0003025, 0.0003004
7: -0.0130936, -0.0081818, -0.0130880, -0.0082790, -0.0047698, 0.0048636
8: -0.0135519, -0.0042928, -0.0133904, -0.0044056, -0.0078487, 0.0079511
9: -0.0020749, 0.0025381, -0.0020231, 0.0024518, -0.0039239, 0.0038822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019357, upper bound: 0.0019786
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019377, upper bound: 0.0019717
time: 1.44 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0031289, 0.0007174, -0.0033671, 0.0035767
1: -0.0046287, -0.0033587, -0.0046538, -0.0034138, -0.0011625, 0.0012126
2: 0.0107999, 0.0159591, 0.0106433, 0.0157805, -0.0044682, 0.0047363
3: 1.0067751, 1.0100324, 1.0068923, 1.0100950, -0.0033199, 0.0031401
4: -0.0042390, -0.0033883, -0.0042057, -0.0033664, -0.0007675, 0.0007259
5: 0.0016620, 0.0046141, 0.0015609, 0.0045227, -0.0025897, 0.0027504
6: -0.0026045, -0.0022991, -0.0026047, -0.0022940, -0.0003106, 0.0003056
7: -0.0130936, -0.0081818, -0.0130800, -0.0078570, -0.0051951, 0.0048594
8: -0.0135519, -0.0042928, -0.0131647, -0.0040777, -0.0082768, 0.0078308
9: -0.0020749, 0.0025381, -0.0021684, 0.0023312, -0.0038596, 0.0040813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019357, upper bound: 0.0019786
time: 1.39 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019377, upper bound: 0.0019717
time: 1.43 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0028919, 0.0007932, -0.0036641, 0.0033008
1: -0.0046698, -0.0033859, -0.0046032, -0.0033791, -0.0012370, 0.0011559
2: 0.0105463, 0.0158645, 0.0109284, 0.0158970, -0.0048551, 0.0043859
3: 1.0068165, 1.0101347, 1.0068345, 1.0099689, -0.0031524, 0.0033002
4: -0.0042213, -0.0033526, -0.0042274, -0.0034079, -0.0007137, 0.0007880
5: 0.0015005, 0.0045657, 0.0017406, 0.0045823, -0.0028177, 0.0025398
6: -0.0026105, -0.0022907, -0.0025985, -0.0023044, -0.0003062, 0.0003078
7: -0.0130864, -0.0077095, -0.0130889, -0.0083606, -0.0046874, 0.0053415
8: -0.0133467, -0.0039441, -0.0134171, -0.0044944, -0.0077170, 0.0084999
9: -0.0022276, 0.0024285, -0.0019807, 0.0024661, -0.0041890, 0.0038155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019720
time: 1.74 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019720
time: 1.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031911, 0.0007694, -0.0028443, 0.0008248, -0.0037178, 0.0032725
1: -0.0046650, -0.0033872, -0.0045890, -0.0033622, -0.0012652, 0.0011486
2: 0.0105683, 0.0158605, 0.0109934, 0.0159456, -0.0049412, 0.0043508
3: 1.0068207, 1.0101230, 1.0067921, 1.0099336, -0.0031129, 0.0033309
4: -0.0042206, -0.0033560, -0.0042365, -0.0034185, -0.0007085, 0.0008044
5: 0.0015137, 0.0045637, 0.0017774, 0.0046072, -0.0028603, 0.0025183
6: -0.0026098, -0.0022916, -0.0026003, -0.0023073, -0.0003026, 0.0003086
7: -0.0130861, -0.0077392, -0.0130926, -0.0084257, -0.0046233, 0.0053196
8: -0.0133381, -0.0039803, -0.0135226, -0.0046067, -0.0076644, 0.0086940
9: -0.0022101, 0.0024239, -0.0019285, 0.0025225, -0.0042943, 0.0037911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019306, upper bound: 0.0019720
time: 1.61 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019306, upper bound: 0.0019720
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028177, 0.0008106, -0.0029587, 0.0008240, -0.0033227, 0.0033990
1: -0.0045833, -0.0033700, -0.0046208, -0.0033636, -0.0011686, 0.0011763
2: 0.0110251, 0.0159238, 0.0108450, 0.0159443, -0.0044468, 0.0045222
3: 1.0068079, 1.0099194, 1.0067888, 1.0100127, -0.0032048, 0.0031306
4: -0.0042324, -0.0034227, -0.0042362, -0.0033949, -0.0007372, 0.0007288
5: 0.0017977, 0.0045961, 0.0016899, 0.0046066, -0.0025592, 0.0026156
6: -0.0025992, -0.0023085, -0.0026028, -0.0023007, -0.0002984, 0.0002943
7: -0.0130909, -0.0084878, -0.0130925, -0.0082499, -0.0047997, 0.0045691
8: -0.0134753, -0.0046476, -0.0135198, -0.0043573, -0.0079732, 0.0079086
9: -0.0019077, 0.0024972, -0.0020457, 0.0025209, -0.0039225, 0.0039406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020072, upper bound: 0.0019186
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0020072, upper bound: 0.0019185
time: 1.57 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027997, 0.0008080, -0.0029146, 0.0008569, -0.0033752, 0.0033712
1: -0.0045784, -0.0033713, -0.0046055, -0.0033467, -0.0011977, 0.0011688
2: 0.0110485, 0.0159198, 0.0109051, 0.0159950, -0.0045307, 0.0044880
3: 1.0068123, 1.0099072, 1.0067493, 1.0099745, -0.0031621, 0.0031580
4: -0.0042317, -0.0034264, -0.0042457, -0.0034052, -0.0007320, 0.0007450
5: 0.0018115, 0.0045940, 0.0017238, 0.0046325, -0.0026009, 0.0025944
6: -0.0025985, -0.0023095, -0.0026044, -0.0023039, -0.0002946, 0.0002950
7: -0.0130906, -0.0085186, -0.0130964, -0.0083032, -0.0047491, 0.0045471
8: -0.0134667, -0.0046857, -0.0136295, -0.0044711, -0.0079198, 0.0081007
9: -0.0018896, 0.0024926, -0.0019894, 0.0025796, -0.0040270, 0.0039158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019796, upper bound: 0.0018799
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019703, upper bound: 0.0018799
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0030486, 0.0007533, -0.0029587, 0.0008240, -0.0035752, 0.0033640
1: -0.0046284, -0.0033956, -0.0046208, -0.0033636, -0.0012299, 0.0011719
2: 0.0107505, 0.0158357, 0.0108450, 0.0159443, -0.0047575, 0.0044684
3: 1.0068442, 1.0100317, 1.0067888, 1.0100127, -0.0031686, 0.0032429
4: -0.0042160, -0.0033837, -0.0042362, -0.0033949, -0.0007272, 0.0007754
5: 0.0016228, 0.0045509, 0.0016899, 0.0046066, -0.0027508, 0.0025881
6: -0.0026057, -0.0022992, -0.0026028, -0.0023007, -0.0003049, 0.0003036
7: -0.0130842, -0.0079707, -0.0130925, -0.0082499, -0.0047956, 0.0050889
8: -0.0132843, -0.0042660, -0.0135198, -0.0043573, -0.0078566, 0.0083891
9: -0.0020752, 0.0023951, -0.0020457, 0.0025209, -0.0041487, 0.0038783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
time: 1.50 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0030306, 0.0007507, -0.0029146, 0.0008569, -0.0036279, 0.0033362
1: -0.0046234, -0.0033969, -0.0046055, -0.0033467, -0.0012590, 0.0011644
2: 0.0107735, 0.0158317, 0.0109051, 0.0159950, -0.0048430, 0.0044342
3: 1.0068485, 1.0100193, 1.0067493, 1.0099745, -0.0031260, 0.0032700
4: -0.0042152, -0.0033873, -0.0042457, -0.0034052, -0.0007220, 0.0007918
5: 0.0016366, 0.0045489, 0.0017238, 0.0046325, -0.0027929, 0.0025669
6: -0.0026050, -0.0023002, -0.0026044, -0.0023039, -0.0003011, 0.0003042
7: -0.0130839, -0.0080011, -0.0130964, -0.0083032, -0.0047450, 0.0050670
8: -0.0132756, -0.0043037, -0.0136295, -0.0044711, -0.0078033, 0.0085840
9: -0.0020569, 0.0023905, -0.0019894, 0.0025796, -0.0042537, 0.0038535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 242

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019540, upper bound: 0.0018570
time: 1.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019379, upper bound: 0.0018570
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027651, 0.0008239, -0.0031789, 0.0007623, -0.0032596, 0.0036520
1: -0.0045701, -0.0033639, -0.0046634, -0.0033907, -0.0011621, 0.0012367
2: 0.0110910, 0.0159442, 0.0105843, 0.0158495, -0.0043598, 0.0048326
3: 1.0067976, 1.0098865, 1.0068295, 1.0101190, -0.0033214, 0.0030570
4: -0.0042362, -0.0034327, -0.0042186, -0.0033579, -0.0007837, 0.0007140
5: 0.0018377, 0.0046065, 0.0015231, 0.0045580, -0.0025104, 0.0028077
6: -0.0025983, -0.0023112, -0.0026091, -0.0022920, -0.0003064, 0.0002980
7: -0.0130925, -0.0085845, -0.0130853, -0.0077544, -0.0052999, 0.0044710
8: -0.0135195, -0.0047509, -0.0133143, -0.0039967, -0.0084545, 0.0077414
9: -0.0018590, 0.0025208, -0.0022040, 0.0024112, -0.0038388, 0.0041694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
time: 1.57 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027651, 0.0008239, -0.0031356, 0.0007910, -0.0033271, 0.0036230
1: -0.0045701, -0.0033639, -0.0046491, -0.0033753, -0.0011948, 0.0012307
2: 0.0110910, 0.0159442, 0.0106404, 0.0158936, -0.0044636, 0.0048023
3: 1.0067976, 1.0098865, 1.0067912, 1.0100833, -0.0032858, 0.0030953
4: -0.0042362, -0.0034327, -0.0042268, -0.0033674, -0.0007795, 0.0007333
5: 0.0018377, 0.0046065, 0.0015564, 0.0045806, -0.0025636, 0.0027863
6: -0.0025983, -0.0023112, -0.0026104, -0.0022949, -0.0003034, 0.0002992
7: -0.0130925, -0.0085845, -0.0130886, -0.0078121, -0.0052423, 0.0044789
8: -0.0135195, -0.0047509, -0.0134099, -0.0041007, -0.0084110, 0.0079662
9: -0.0018590, 0.0025208, -0.0021517, 0.0024622, -0.0039590, 0.0041476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
time: 1.50 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027162, 0.0008501, -0.0031789, 0.0007623, -0.0032257, 0.0037204
1: -0.0045544, -0.0033499, -0.0046634, -0.0033907, -0.0011547, 0.0012706
2: 0.0111583, 0.0159846, 0.0105843, 0.0158495, -0.0043170, 0.0049378
3: 1.0067619, 1.0098473, 1.0068295, 1.0101190, -0.0033571, 0.0030178
4: -0.0042437, -0.0034438, -0.0042186, -0.0033579, -0.0008033, 0.0007078
5: 0.0018755, 0.0046272, 0.0015231, 0.0045580, -0.0024847, 0.0028616
6: -0.0025999, -0.0023144, -0.0026091, -0.0022920, -0.0003079, 0.0002947
7: -0.0130956, -0.0086468, -0.0130853, -0.0077544, -0.0053080, 0.0044103
8: -0.0136070, -0.0048698, -0.0133143, -0.0039967, -0.0086825, 0.0076816
9: -0.0018011, 0.0025676, -0.0022040, 0.0024112, -0.0038115, 0.0042913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.46 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027162, 0.0008501, -0.0031356, 0.0007910, -0.0032523, 0.0036505
1: -0.0045544, -0.0033499, -0.0046491, -0.0033753, -0.0011658, 0.0012426
2: 0.0111583, 0.0159846, 0.0106404, 0.0158936, -0.0043581, 0.0048410
3: 1.0067619, 1.0098473, 1.0067912, 1.0100833, -0.0033214, 0.0030560
4: -0.0042437, -0.0034438, -0.0042268, -0.0033674, -0.0007863, 0.0007150
5: 0.0018755, 0.0046272, 0.0015564, 0.0045806, -0.0025055, 0.0028077
6: -0.0025999, -0.0023144, -0.0026104, -0.0022949, -0.0003050, 0.0002960
7: -0.0130956, -0.0086468, -0.0130886, -0.0078121, -0.0052460, 0.0044131
8: -0.0136070, -0.0048698, -0.0134099, -0.0041007, -0.0084918, 0.0077631
9: -0.0018011, 0.0025676, -0.0021517, 0.0024622, -0.0038551, 0.0041925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 195
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240
type: A, layer: 1, pos: 190

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 195

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.63 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
time: 1.62 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0028177, 0.0008106, -0.0034687, 0.0033150
1: -0.0046287, -0.0033587, -0.0045833, -0.0033700, -0.0011982, 0.0011627
2: 0.0107999, 0.0159591, 0.0110251, 0.0159238, -0.0046240, 0.0044350
3: 1.0067751, 1.0100324, 1.0068079, 1.0099194, -0.0031443, 0.0032245
4: -0.0042390, -0.0033883, -0.0042324, -0.0034227, -0.0007266, 0.0007557
5: 0.0016620, 0.0046141, 0.0017977, 0.0045961, -0.0026696, 0.0025532
6: -0.0026045, -0.0022991, -0.0025992, -0.0023085, -0.0002961, 0.0003001
7: -0.0130936, -0.0081818, -0.0130909, -0.0084878, -0.0045682, 0.0048731
8: -0.0135519, -0.0042928, -0.0134753, -0.0046476, -0.0078830, 0.0081816
9: -0.0020749, 0.0025381, -0.0019077, 0.0024972, -0.0040496, 0.0039089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019161, upper bound: 0.0019793
time: 1.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019174, upper bound: 0.0019710
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0030486, 0.0007533, -0.0034376, 0.0035676
1: -0.0046287, -0.0033587, -0.0046284, -0.0033956, -0.0011928, 0.0012240
2: 0.0107999, 0.0159591, 0.0107505, 0.0158357, -0.0045762, 0.0047457
3: 1.0067751, 1.0100324, 1.0068442, 1.0100317, -0.0032566, 0.0031883
4: -0.0042390, -0.0033883, -0.0042160, -0.0033837, -0.0007732, 0.0007468
5: 0.0016620, 0.0046141, 0.0016228, 0.0045509, -0.0026452, 0.0027448
6: -0.0026045, -0.0022991, -0.0026057, -0.0022992, -0.0003054, 0.0003066
7: -0.0130936, -0.0081818, -0.0130842, -0.0079707, -0.0050880, 0.0048694
8: -0.0135519, -0.0042928, -0.0132843, -0.0042660, -0.0083635, 0.0080780
9: -0.0020749, 0.0025381, -0.0020752, 0.0023951, -0.0039943, 0.0041350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019161, upper bound: 0.0019793
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019174, upper bound: 0.0019710
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0027651, 0.0008239, -0.0037199, 0.0032516
1: -0.0046698, -0.0033859, -0.0045701, -0.0033639, -0.0012531, 0.0011557
2: 0.0105463, 0.0158645, 0.0110910, 0.0159442, -0.0049353, 0.0043474
3: 1.0068165, 1.0101347, 1.0067976, 1.0098865, -0.0030700, 0.0033371
4: -0.0042213, -0.0033526, -0.0042362, -0.0034327, -0.0007117, 0.0008029
5: 0.0015005, 0.0045657, 0.0018377, 0.0046065, -0.0028613, 0.0025041
6: -0.0026105, -0.0022907, -0.0025983, -0.0023112, -0.0002994, 0.0003077
7: -0.0130864, -0.0077095, -0.0130925, -0.0085845, -0.0044701, 0.0053476
8: -0.0133467, -0.0039441, -0.0135195, -0.0047509, -0.0077145, 0.0086706
9: -0.0022276, 0.0024285, -0.0018590, 0.0025208, -0.0042778, 0.0038245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019713
time: 1.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019713
time: 1.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031911, 0.0007694, -0.0027162, 0.0008501, -0.0037685, 0.0032258
1: -0.0046650, -0.0033872, -0.0045544, -0.0033499, -0.0012813, 0.0011499
2: 0.0105683, 0.0158605, 0.0111583, 0.0159846, -0.0050142, 0.0043174
3: 1.0068207, 1.0101230, 1.0067619, 1.0098473, -0.0030266, 0.0033611
4: -0.0042206, -0.0033560, -0.0042437, -0.0034438, -0.0007074, 0.0008181
5: 0.0015137, 0.0045637, 0.0018755, 0.0046272, -0.0029000, 0.0024847
6: -0.0026098, -0.0022916, -0.0025999, -0.0023144, -0.0002954, 0.0003083
7: -0.0130861, -0.0077392, -0.0130956, -0.0086468, -0.0044100, 0.0053253
8: -0.0133381, -0.0039803, -0.0136070, -0.0048698, -0.0076750, 0.0088511
9: -0.0022101, 0.0024239, -0.0018011, 0.0025676, -0.0043754, 0.0038080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019714
time: 1.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019713
time: 1.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0029951, 0.0008336, -0.0033998, 0.0033998
1: -0.0046287, -0.0033587, -0.0046287, -0.0033587, -0.0011710, 0.0011710
2: 0.0107999, 0.0159591, 0.0107999, 0.0159591, -0.0045164, 0.0045164
3: 1.0067751, 1.0100324, 1.0067751, 1.0100324, -0.0032573, 0.0032573
4: -0.0042390, -0.0033883, -0.0042390, -0.0033883, -0.0007346, 0.0007346
5: 0.0016620, 0.0046141, 0.0016620, 0.0046141, -0.0026154, 0.0026154
6: -0.0026045, -0.0022991, -0.0026045, -0.0022991, -0.0003054, 0.0003054
7: -0.0130936, -0.0081818, -0.0130936, -0.0081818, -0.0048656, 0.0048656
8: -0.0135519, -0.0042928, -0.0135519, -0.0042928, -0.0079344, 0.0079344
9: -0.0020749, 0.0025381, -0.0020749, 0.0025381, -0.0039149, 0.0039149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019369, upper bound: 0.0019844
time: 1.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019392, upper bound: 0.0019767
time: 1.98 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0029951, 0.0008336, -0.0032084, 0.0007720, -0.0033696, 0.0036399
1: -0.0046287, -0.0033587, -0.0046698, -0.0033859, -0.0011653, 0.0012319
2: 0.0107999, 0.0159591, 0.0105463, 0.0158645, -0.0044701, 0.0048150
3: 1.0067751, 1.0100324, 1.0068165, 1.0101347, -0.0033596, 0.0032159
4: -0.0042390, -0.0033883, -0.0042213, -0.0033526, -0.0007799, 0.0007260
5: 0.0016620, 0.0046141, 0.0015005, 0.0045657, -0.0025917, 0.0027984
6: -0.0026045, -0.0022991, -0.0026105, -0.0022907, -0.0003139, 0.0003114
7: -0.0130936, -0.0081818, -0.0130864, -0.0077095, -0.0053402, 0.0048621
8: -0.0135519, -0.0042928, -0.0133467, -0.0039441, -0.0084051, 0.0078339
9: -0.0020749, 0.0025381, -0.0022276, 0.0024285, -0.0038612, 0.0041394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019369, upper bound: 0.0019844
time: 1.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019392, upper bound: 0.0019766
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0032084, 0.0007720, -0.0029408, 0.0008447, -0.0036621, 0.0033363
1: -0.0046698, -0.0033859, -0.0046157, -0.0033533, -0.0012360, 0.0011636
2: 0.0105463, 0.0158645, 0.0108674, 0.0159762, -0.0048491, 0.0044292
3: 1.0068165, 1.0101347, 1.0067655, 1.0100001, -0.0031836, 0.0033692
4: -0.0042213, -0.0033526, -0.0042422, -0.0033984, -0.0007198, 0.0007862
5: 0.0015005, 0.0045657, 0.0017034, 0.0046229, -0.0028159, 0.0025667
6: -0.0026105, -0.0022907, -0.0026034, -0.0023018, -0.0003087, 0.0003128
7: -0.0130864, -0.0077095, -0.0130949, -0.0082812, -0.0047652, 0.0053428
8: -0.0133467, -0.0039441, -0.0135889, -0.0043953, -0.0077675, 0.0084790
9: -0.0022276, 0.0024285, -0.0020270, 0.0025579, -0.0041789, 0.0038310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771
time: 1.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771
time: 1.87 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0031911, 0.0007694, -0.0028984, 0.0008765, -0.0037112, 0.0033150
1: -0.0046650, -0.0033872, -0.0046011, -0.0033368, -0.0012628, 0.0011597
2: 0.0105683, 0.0158605, 0.0109251, 0.0160251, -0.0049287, 0.0044056
3: 1.0068207, 1.0101230, 1.0067252, 1.0099635, -0.0031428, 0.0033978
4: -0.0042206, -0.0033560, -0.0042513, -0.0034083, -0.0007167, 0.0008016
5: 0.0015137, 0.0045637, 0.0017360, 0.0046479, -0.0028550, 0.0025507
6: -0.0026098, -0.0022916, -0.0026052, -0.0023048, -0.0003050, 0.0003136
7: -0.0130861, -0.0077392, -0.0130987, -0.0083349, -0.0047130, 0.0053205
8: -0.0133381, -0.0039803, -0.0136947, -0.0045040, -0.0077403, 0.0086611
9: -0.0022101, 0.0024239, -0.0019733, 0.0026144, -0.0042768, 0.0038209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 242
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 90
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 71
type: A, layer: 1, pos: 240

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771
time: 1.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771
time: 1.86 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.25 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019971, upper bound: 0.0019181
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019971, upper bound: 0.0019181
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019691, upper bound: 0.0018794
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019616, upper bound: 0.0018795
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019702, upper bound: 0.0019063
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019331, upper bound: 0.0018549
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019105, upper bound: 0.0018549
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019701, upper bound: 0.0019049
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019002, upper bound: 0.0018876
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019767
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019702
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019767
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019702
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019701
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019702
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019701
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019702
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019233, upper bound: 0.0019798
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019237, upper bound: 0.0019736
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019233, upper bound: 0.0019799
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019237, upper bound: 0.0019736
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019106, upper bound: 0.0019736
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019969, upper bound: 0.0019303
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019969, upper bound: 0.0019303
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019688, upper bound: 0.0018942
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019611, upper bound: 0.0018942
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019160
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019160
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019328, upper bound: 0.0018704
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019105, upper bound: 0.0018703
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019261
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019685, upper bound: 0.0019079
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019937
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019879
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019937
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019879
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019880
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018872, upper bound: 0.0019879
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019978
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019921
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019062, upper bound: 0.0019978
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019063, upper bound: 0.0019922
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019097, upper bound: 0.0019922
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019179
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0020068, upper bound: 0.0019180
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019793, upper bound: 0.0018794
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019700, upper bound: 0.0018794
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019880, upper bound: 0.0019062
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019880, upper bound: 0.0019062
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019527, upper bound: 0.0018548
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019360, upper bound: 0.0018548
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019879, upper bound: 0.0019032
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019197, upper bound: 0.0018872
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019160, upper bound: 0.0019756
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019173, upper bound: 0.0019685
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019160, upper bound: 0.0019756
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019173, upper bound: 0.0019685
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019687
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0018876, upper bound: 0.0019688
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019688
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019688
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019357, upper bound: 0.0019786
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019377, upper bound: 0.0019717
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019357, upper bound: 0.0019786
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019377, upper bound: 0.0019717
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019720
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019720
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019306, upper bound: 0.0019720
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019306, upper bound: 0.0019720
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0020072, upper bound: 0.0019186
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0020072, upper bound: 0.0019185
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019796, upper bound: 0.0018799
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019703, upper bound: 0.0018799
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019073
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019540, upper bound: 0.0018570
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019379, upper bound: 0.0018570
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0019051
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019886, upper bound: 0.0018896
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019161, upper bound: 0.0019793
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019174, upper bound: 0.0019710
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019161, upper bound: 0.0019793
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019174, upper bound: 0.0019710
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019713
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019713
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019714
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019079, upper bound: 0.0019713
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019369, upper bound: 0.0019844
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019392, upper bound: 0.0019767
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019369, upper bound: 0.0019844
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019392, upper bound: 0.0019766
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.25
Output dim: 3, lower bound: -0.0019317, upper bound: 0.0019771

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027017, 0.0007400, -0.0029053, 0.0007750, -0.0031565, 0.0033096
1: -0.0045542, -0.0034047, -0.0046063, -0.0033881, -0.0011173, 0.0011357
2: 0.0111685, 0.0158152, 0.0109116, 0.0158690, -0.0042181, 0.0043990
3: 1.0069011, 1.0098467, 1.0068551, 1.0099767, -0.0030756, 0.0029916
4: -0.0042122, -0.0034446, -0.0042222, -0.0034053, -0.0007159, 0.0006906
5: 0.0018860, 0.0045405, 0.0017304, 0.0045680, -0.0024307, 0.0025465
6: -0.0025910, -0.0023145, -0.0025979, -0.0023037, -0.0002873, 0.0002835
7: -0.0130826, -0.0086915, -0.0130868, -0.0083347, -0.0047114, 0.0043600
8: -0.0132399, -0.0048760, -0.0133566, -0.0044688, -0.0077374, 0.0074948
9: -0.0018002, 0.0023714, -0.0019923, 0.0024337, -0.0037205, 0.0038212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.65 + 597.32 = 600.97 seconds
