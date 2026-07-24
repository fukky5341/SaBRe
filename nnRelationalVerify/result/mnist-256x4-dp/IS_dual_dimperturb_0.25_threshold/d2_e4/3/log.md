## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0071008


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0065955, 0.0049056, -0.0065955, 0.0049056, -0.0115011, 0.0115011)
1: (0.9973307, 1.0101113, 0.9973307, 1.0101113, -0.0110931, 0.0110931)
2: (-0.0058871, 0.0056926, -0.0058871, 0.0056926, -0.0115797, 0.0115797)
3: (-0.0001320, 0.0024509, -0.0001320, 0.0024509, -0.0020441, 0.0020441)
4: (-0.0068269, 0.0015425, -0.0068269, 0.0015425, -0.0083693, 0.0083693)
5: (-0.0022242, 0.0081031, -0.0022242, 0.0081031, -0.0103273, 0.0103273)
6: (-0.0089111, 0.0019475, -0.0089111, 0.0019475, -0.0108586, 0.0108586)
7: (-0.0056616, 0.0000307, -0.0056616, 0.0000307, -0.0056923, 0.0056923)
8: (-0.0134321, -0.0021454, -0.0134321, -0.0021454, -0.0112867, 0.0112867)
9: (-0.0044693, 0.0075429, -0.0044693, 0.0075429, -0.0120123, 0.0120123)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.65 + 2.06 = 3.71 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0101766, upper bound: 0.0101766

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 118

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098573, upper bound: 0.0099282
time: 1.20 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0098673, upper bound: 0.0098673
time: 0.96 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.33 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.33
Output dim: 1, lower bound: -0.0098573, upper bound: 0.0099282
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.33
Output dim: 1, lower bound: -0.0098673, upper bound: 0.0098673

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0065516, 0.0048481, -0.0065845, 0.0048911, -0.0114428, 0.0114326
1: 0.9974643, 1.0100242, 0.9973658, 1.0100894, -0.0109689, 0.0109779
2: -0.0058300, 0.0056472, -0.0058727, 0.0056812, -0.0115112, 0.0115199
3: -0.0001198, 0.0023955, -0.0001289, 0.0024363, -0.0020201, 0.0020000
4: -0.0067692, 0.0015276, -0.0068124, 0.0015387, -0.0083079, 0.0083399
5: -0.0022042, 0.0080308, -0.0022192, 0.0080849, -0.0102891, 0.0102500
6: -0.0088219, 0.0019405, -0.0088887, 0.0019458, -0.0107677, 0.0108292
7: -0.0055682, -0.0000011, -0.0056370, 0.0000228, -0.0055909, 0.0056360
8: -0.0133876, -0.0024343, -0.0134209, -0.0022213, -0.0111663, 0.0109867
9: -0.0043898, 0.0075168, -0.0044494, 0.0075364, -0.0119261, 0.0119662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096173, upper bound: 0.0096789
time: 0.90 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096703, upper bound: 0.0097391
time: 0.93 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0069683, 0.0053936, -0.0065713, 0.0048738, -0.0118422, 0.0119649
1: 0.9974788, 1.0108516, 0.9973950, 1.0100631, -0.0109839, 0.0117740
2: -0.0063718, 0.0060787, -0.0058555, 0.0056675, -0.0120393, 0.0119343
3: -0.0002364, 0.0023894, -0.0001252, 0.0024242, -0.0021440, 0.0020226
4: -0.0073176, 0.0016689, -0.0067950, 0.0015342, -0.0088518, 0.0084639
5: -0.0023940, 0.0087171, -0.0022131, 0.0080631, -0.0104571, 0.0109302
6: -0.0096691, 0.0020075, -0.0088618, 0.0019436, -0.0116127, 0.0108693
7: -0.0055580, 0.0003012, -0.0056166, 0.0000132, -0.0055711, 0.0059178
8: -0.0138101, -0.0024659, -0.0134075, -0.0022844, -0.0115257, 0.0109417
9: -0.0051455, 0.0077647, -0.0044254, 0.0075285, -0.0126740, 0.0121901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 190

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096237, upper bound: 0.0096100
time: 1.10 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0096775, upper bound: 0.0096775
time: 1.24 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.88 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 1, lower bound: -0.0096173, upper bound: 0.0096789
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 1, lower bound: -0.0096703, upper bound: 0.0097391
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 1, lower bound: -0.0096237, upper bound: 0.0096100
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 3.88
Output dim: 1, lower bound: -0.0096775, upper bound: 0.0096775

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.0062362, 0.0044352, -0.0064622, 0.0047311, -0.0109673, 0.0108974
1: 0.9974356, 1.0093980, 0.9973681, 1.0098467, -0.0106532, 0.0103208
2: -0.0054199, 0.0053205, -0.0057138, 0.0055546, -0.0109745, 0.0110343
3: -0.0000314, 0.0024073, -0.0000947, 0.0024354, -0.0019104, 0.0019301
4: -0.0063540, 0.0014206, -0.0066515, 0.0014972, -0.0078512, 0.0080720
5: -0.0020606, 0.0075114, -0.0021635, 0.0078836, -0.0099442, 0.0096749
6: -0.0081806, 0.0018897, -0.0086402, 0.0019261, -0.0101067, 0.0105299
7: -0.0055882, -0.0002299, -0.0056355, -0.0000659, -0.0055222, 0.0054056
8: -0.0130679, -0.0023724, -0.0132970, -0.0022261, -0.0108418, 0.0109246
9: -0.0038178, 0.0073292, -0.0042277, 0.0074637, -0.0112814, 0.0115569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A1

### Relational analysis result of IS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094553, upper bound: 0.0095118
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A2

### Relational analysis result of IS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094596, upper bound: 0.0095028
time: 0.95 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.0063260, 0.0045527, -0.0065170, 0.0048028, -0.0111287, 0.0110697
1: 0.9974683, 1.0095761, 0.9973670, 1.0099554, -0.0108304, 0.0104216
2: -0.0055365, 0.0054135, -0.0057850, 0.0056113, -0.0111478, 0.0111984
3: -0.0000566, 0.0023938, -0.0001100, 0.0024358, -0.0019034, 0.0019745
4: -0.0064721, 0.0014510, -0.0067236, 0.0015158, -0.0079879, 0.0081746
5: -0.0021014, 0.0076592, -0.0021884, 0.0079738, -0.0100752, 0.0098476
6: -0.0083631, 0.0019042, -0.0087515, 0.0019349, -0.0102980, 0.0106556
7: -0.0055653, -0.0001648, -0.0056362, -0.0000262, -0.0055391, 0.0054714
8: -0.0131588, -0.0024430, -0.0133525, -0.0022239, -0.0109349, 0.0109095
9: -0.0039805, 0.0073826, -0.0043270, 0.0074962, -0.0114767, 0.0117096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094900, upper bound: 0.0095588
time: 1.38 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095104, upper bound: 0.0095728
time: 0.97 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -0.0066426, 0.0049672, -0.0064487, 0.0047133, -0.0113559, 0.0114159
1: 0.9974488, 1.0102048, 0.9973972, 1.0098199, -0.0106674, 0.0111008
2: -0.0059483, 0.0057414, -0.0056961, 0.0055406, -0.0114888, 0.0114375
3: -0.0001452, 0.0024019, -0.0000909, 0.0024233, -0.0020337, 0.0019529
4: -0.0068889, 0.0015584, -0.0066336, 0.0014926, -0.0083815, 0.0081920
5: -0.0022456, 0.0081806, -0.0021573, 0.0078612, -0.0101069, 0.0103379
6: -0.0090068, 0.0019551, -0.0086125, 0.0019239, -0.0109307, 0.0105676
7: -0.0055790, 0.0000649, -0.0056151, -0.0000758, -0.0055032, 0.0056800
8: -0.0134798, -0.0024007, -0.0132832, -0.0022892, -0.0111906, 0.0108825
9: -0.0045547, 0.0075709, -0.0042030, 0.0074556, -0.0120103, 0.0117740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A2_A1_A1

### Relational analysis result of IS_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094575, upper bound: 0.0094289
time: 0.93 seconds

## Relational analysis of IS_A2_A1_A2

### Relational analysis result of IS_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094640, upper bound: 0.0094439
time: 0.90 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -0.0067414, 0.0050966, -0.0065037, 0.0047854, -0.0115268, 0.0116002
1: 0.9974832, 1.0104010, 0.9973961, 1.0099288, -0.0108453, 0.0112170
2: -0.0060767, 0.0058437, -0.0057676, 0.0055975, -0.0116743, 0.0116114
3: -0.0001729, 0.0023876, -0.0001063, 0.0024237, -0.0020273, 0.0019969
4: -0.0070189, 0.0015919, -0.0067060, 0.0015113, -0.0085302, 0.0082980
5: -0.0022906, 0.0083433, -0.0021824, 0.0079518, -0.0102425, 0.0105257
6: -0.0092077, 0.0019710, -0.0087244, 0.0019328, -0.0111405, 0.0106954
7: -0.0055549, 0.0001366, -0.0056158, -0.0000359, -0.0055190, 0.0057524
8: -0.0135800, -0.0024753, -0.0133390, -0.0022870, -0.0112930, 0.0108637
9: -0.0047339, 0.0076297, -0.0043028, 0.0074883, -0.0122222, 0.0119325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=8, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 173

## Relational analysis of IS_A2_A2_A1

### Relational analysis result of IS_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094989, upper bound: 0.0094929
time: 0.96 seconds

## Relational analysis of IS_A2_A2_A2

### Relational analysis result of IS_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095158, upper bound: 0.0095158
time: 0.98 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.51 seconds
IS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0094553, upper bound: 0.0095118
IS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0094596, upper bound: 0.0095028
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0094900, upper bound: 0.0095588
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0095104, upper bound: 0.0095728
IS_A2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0094575, upper bound: 0.0094289
IS_A2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0094640, upper bound: 0.0094439
IS_A2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0094989, upper bound: 0.0094929
IS_A2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 3.51
Output dim: 1, lower bound: -0.0095158, upper bound: 0.0095158

## BFS IS instance: IS_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0059783, 0.0040975, -0.0063753, 0.0046173, -0.0105956, 0.0104728
1: 0.9974320, 1.0088857, 0.9973699, 1.0096741, -0.0104255, 0.0097954
2: -0.0050845, 0.0050534, -0.0056007, 0.0054646, -0.0105490, 0.0106541
3: 0.0000408, 0.0024088, -0.0000704, 0.0024346, -0.0018374, 0.0018792
4: -0.0060145, 0.0013331, -0.0065371, 0.0014677, -0.0074822, 0.0078701
5: -0.0019431, 0.0070866, -0.0021239, 0.0077404, -0.0096835, 0.0092105
6: -0.0076562, 0.0018482, -0.0084634, 0.0019121, -0.0095683, 0.0103116
7: -0.0055907, -0.0004170, -0.0056342, -0.0001290, -0.0054617, 0.0052172
8: -0.0128063, -0.0023647, -0.0132089, -0.0022301, -0.0105762, 0.0108442
9: -0.0033500, 0.0071758, -0.0040700, 0.0074120, -0.0107619, 0.0112458

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094553, upper bound: 0.0095118
time: 0.87 seconds

## Relational analysis of IS_A1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094553, upper bound: 0.0095118
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0059963, 0.0041211, -0.0063798, 0.0046231, -0.0106194, 0.0105009
1: 0.9974394, 1.0089213, 0.9973693, 1.0096828, -0.0104905, 0.0097854
2: -0.0051079, 0.0050721, -0.0056065, 0.0054692, -0.0105771, 0.0106786
3: 0.0000357, 0.0024057, -0.0000716, 0.0024348, -0.0018231, 0.0019016
4: -0.0060382, 0.0013392, -0.0065429, 0.0014692, -0.0075075, 0.0078821
5: -0.0019513, 0.0071163, -0.0021259, 0.0077478, -0.0096990, 0.0092422
6: -0.0076929, 0.0018511, -0.0084724, 0.0019128, -0.0096057, 0.0103236
7: -0.0055854, -0.0004039, -0.0056346, -0.0001258, -0.0054596, 0.0052307
8: -0.0128246, -0.0023810, -0.0132134, -0.0022290, -0.0105957, 0.0108324
9: -0.0033827, 0.0071865, -0.0040781, 0.0074146, -0.0107973, 0.0112646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094596, upper bound: 0.0095028
time: 1.17 seconds

## Relational analysis of IS_A1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094596, upper bound: 0.0095028
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0062316, 0.0044292, -0.0062548, 0.0044596, -0.0106912, 0.0106840
1: 0.9974701, 1.0093888, 0.9973719, 1.0094347, -0.0102976, 0.0101944
2: -0.0054139, 0.0053158, -0.0054441, 0.0053398, -0.0107537, 0.0107598
3: -0.0000302, 0.0023930, -0.0000367, 0.0024338, -0.0018571, 0.0019007
4: -0.0063479, 0.0014190, -0.0063785, 0.0014269, -0.0077748, 0.0077975
5: -0.0020584, 0.0075038, -0.0020690, 0.0075421, -0.0096005, 0.0095728
6: -0.0081713, 0.0018890, -0.0082185, 0.0018927, -0.0100640, 0.0101075
7: -0.0055640, -0.0002332, -0.0056329, -0.0002164, -0.0053477, 0.0053996
8: -0.0130632, -0.0024470, -0.0130867, -0.0022343, -0.0108289, 0.0106397
9: -0.0038094, 0.0073265, -0.0038515, 0.0073403, -0.0111497, 0.0111780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094900, upper bound: 0.0095588
time: 1.03 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0094900, upper bound: 0.0095588
time: 1.52 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0062480, 0.0044507, -0.0062897, 0.0045052, -0.0107532, 0.0107404
1: 0.9974697, 1.0094213, 0.9973708, 1.0095042, -0.0102899, 0.0102604
2: -0.0054352, 0.0053327, -0.0054894, 0.0053759, -0.0108111, 0.0108222
3: -0.0000347, 0.0023932, -0.0000464, 0.0024342, -0.0018744, 0.0018787
4: -0.0063695, 0.0014246, -0.0064244, 0.0014387, -0.0078082, 0.0078490
5: -0.0020659, 0.0075308, -0.0020849, 0.0075995, -0.0096654, 0.0096157
6: -0.0082046, 0.0018916, -0.0082894, 0.0018983, -0.0101030, 0.0101810
7: -0.0055644, -0.0002213, -0.0056335, -0.0001911, -0.0053733, 0.0054122
8: -0.0130798, -0.0024459, -0.0131221, -0.0022322, -0.0108476, 0.0106761
9: -0.0038392, 0.0073363, -0.0039148, 0.0073610, -0.0112002, 0.0112510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095104, upper bound: 0.0095728
time: 0.93 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0095104, upper bound: 0.0095728
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0063998, 0.0046493, -0.0063618, 0.0045995, -0.0109993, 0.0110111
1: 0.9974471, 1.0097226, 0.9973992, 1.0096471, -0.0104425, 0.0106038
2: -0.0056325, 0.0054899, -0.0055831, 0.0054506, -0.0110831, 0.0110730
3: -0.0000772, 0.0024025, -0.0000666, 0.0024225, -0.0019659, 0.0019020
4: -0.0065693, 0.0014760, -0.0065192, 0.0014632, -0.0080324, 0.0079953
5: -0.0021350, 0.0077807, -0.0021177, 0.0077181, -0.0098532, 0.0098985
6: -0.0085131, 0.0019160, -0.0084359, 0.0019099, -0.0104231, 0.0103519
7: -0.0055801, -0.0001113, -0.0056138, -0.0001388, -0.0054412, 0.0055026
8: -0.0132337, -0.0023975, -0.0131951, -0.0022932, -0.0109405, 0.0107976
9: -0.0041144, 0.0074265, -0.0040454, 0.0074039, -0.0115183, 0.0114719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_A1_A1

### Relational analysis result of IS_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090220, upper bound: 0.0089907
time: 0.83 seconds

## Relational analysis of IS_A2_A1_A1_A2

### Relational analysis result of IS_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092828, upper bound: 0.0092506
time: 1.08 seconds

## BFS IS instance: IS_A2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0063916, 0.0046387, -0.0063663, 0.0046055, -0.0109972, 0.0110050
1: 0.9974532, 1.0097065, 0.9973986, 1.0096561, -0.0105049, 0.0105747
2: -0.0056219, 0.0054815, -0.0055891, 0.0054553, -0.0110772, 0.0110705
3: -0.0000750, 0.0024000, -0.0000679, 0.0024227, -0.0019552, 0.0019242
4: -0.0065585, 0.0014733, -0.0065253, 0.0014647, -0.0080232, 0.0079985
5: -0.0021313, 0.0077673, -0.0021198, 0.0077256, -0.0098570, 0.0098871
6: -0.0084966, 0.0019147, -0.0084452, 0.0019107, -0.0104072, 0.0103599
7: -0.0055759, -0.0001172, -0.0056141, -0.0001355, -0.0054404, 0.0054969
8: -0.0132254, -0.0024104, -0.0131998, -0.0022922, -0.0109332, 0.0107894
9: -0.0040996, 0.0074217, -0.0040537, 0.0074066, -0.0115062, 0.0114754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090298, upper bound: 0.0090201
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092877, upper bound: 0.0092641
time: 0.86 seconds

## BFS IS instance: IS_A2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0064863, 0.0047626, -0.0064105, 0.0046633, -0.0111496, 0.0111730
1: 0.9974868, 1.0098944, 0.9973980, 1.0097438, -0.0105910, 0.0107137
2: -0.0057450, 0.0055795, -0.0056464, 0.0055010, -0.0112460, 0.0112259
3: -0.0001014, 0.0023862, -0.0000802, 0.0024229, -0.0019594, 0.0019402
4: -0.0066831, 0.0015054, -0.0065833, 0.0014797, -0.0081628, 0.0080887
5: -0.0021744, 0.0079232, -0.0021399, 0.0077983, -0.0099728, 0.0100631
6: -0.0086890, 0.0019300, -0.0085349, 0.0019178, -0.0106068, 0.0104648
7: -0.0055524, -0.0000485, -0.0056145, -0.0001035, -0.0054489, 0.0055661
8: -0.0133214, -0.0024828, -0.0132445, -0.0022909, -0.0110304, 0.0107617
9: -0.0042713, 0.0074780, -0.0041337, 0.0074329, -0.0117041, 0.0116117

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090491, upper bound: 0.0090434
time: 1.27 seconds

## Relational analysis of IS_A2_A2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093250, upper bound: 0.0093187
time: 1.17 seconds

## BFS IS instance: IS_A2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0065043, 0.0047861, -0.0064250, 0.0046823, -0.0111866, 0.0112111
1: 0.9974876, 1.0099300, 0.9973975, 1.0097728, -0.0106868, 0.0107162
2: -0.0057684, 0.0055982, -0.0056653, 0.0055160, -0.0112845, 0.0112635
3: -0.0001065, 0.0023858, -0.0000843, 0.0024231, -0.0019545, 0.0019683
4: -0.0067068, 0.0015115, -0.0066024, 0.0014846, -0.0081914, 0.0081139
5: -0.0021826, 0.0079528, -0.0021465, 0.0078222, -0.0100049, 0.0100994
6: -0.0087256, 0.0019329, -0.0085644, 0.0019201, -0.0106457, 0.0104972
7: -0.0055518, -0.0000354, -0.0056149, -0.0000929, -0.0054588, 0.0055794
8: -0.0133396, -0.0024849, -0.0132592, -0.0022900, -0.0110497, 0.0107743
9: -0.0043039, 0.0074887, -0.0041601, 0.0074415, -0.0117454, 0.0116488

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090696, upper bound: 0.0090803
time: 0.81 seconds

## Relational analysis of IS_A2_A2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093513, upper bound: 0.0093513
time: 1.00 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.32 seconds
IS_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0094553, upper bound: 0.0095118
IS_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0094553, upper bound: 0.0095118
IS_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0094596, upper bound: 0.0095028
IS_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0094596, upper bound: 0.0095028
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0094900, upper bound: 0.0095588
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0094900, upper bound: 0.0095588
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0095104, upper bound: 0.0095728
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0095104, upper bound: 0.0095728
IS_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0090220, upper bound: 0.0089907
IS_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0092828, upper bound: 0.0092506
IS_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0090298, upper bound: 0.0090201
IS_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0092877, upper bound: 0.0092641
IS_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0090491, upper bound: 0.0090434
IS_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0093250, upper bound: 0.0093187
IS_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0090696, upper bound: 0.0090803
IS_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.32
Output dim: 1, lower bound: -0.0093513, upper bound: 0.0093513

## BFS IS instance: IS_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0059783, 0.0040975, -0.0063427, 0.0045746, -0.0105529, 0.0104402
1: 0.9974320, 1.0088857, 0.9974685, 1.0096095, -0.0103614, 0.0097216
2: -0.0050845, 0.0050534, -0.0055583, 0.0054308, -0.0105153, 0.0106117
3: 0.0000408, 0.0024088, -0.0000613, 0.0023938, -0.0018077, 0.0018697
4: -0.0060145, 0.0013331, -0.0064942, 0.0014567, -0.0074712, 0.0078272
5: -0.0019431, 0.0070866, -0.0021090, 0.0076868, -0.0096298, 0.0091956
6: -0.0076562, 0.0018482, -0.0083971, 0.0019069, -0.0095630, 0.0102453
7: -0.0055907, -0.0004170, -0.0055653, -0.0001526, -0.0054380, 0.0051483
8: -0.0128063, -0.0023647, -0.0131758, -0.0024431, -0.0103632, 0.0108112
9: -0.0033500, 0.0071758, -0.0040109, 0.0073926, -0.0107425, 0.0111867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090164, upper bound: 0.0090833
time: 0.86 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092797, upper bound: 0.0093352
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0059783, 0.0040975, -0.0067570, 0.0051170, -0.0110953, 0.0108546
1: 0.9974320, 1.0088857, 0.9974833, 1.0104321, -0.0112140, 0.0096502
2: -0.0050845, 0.0050534, -0.0060971, 0.0058599, -0.0109444, 0.0111505
3: 0.0000408, 0.0024088, -0.0001773, 0.0023876, -0.0017831, 0.0020165
4: -0.0060145, 0.0013331, -0.0070395, 0.0015972, -0.0076117, 0.0083726
5: -0.0019431, 0.0070866, -0.0022978, 0.0083691, -0.0103122, 0.0093843
6: -0.0076562, 0.0018482, -0.0092395, 0.0019735, -0.0096297, 0.0110878
7: -0.0055907, -0.0004170, -0.0055549, 0.0001479, -0.0057386, 0.0051379
8: -0.0128063, -0.0023647, -0.0135959, -0.0024752, -0.0103311, 0.0112312
9: -0.0033500, 0.0071758, -0.0047623, 0.0076390, -0.0109890, 0.0119381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090164, upper bound: 0.0090833
time: 1.02 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092797, upper bound: 0.0093352
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0059963, 0.0041211, -0.0063469, 0.0045801, -0.0105764, 0.0104681
1: 0.9974394, 1.0089213, 0.9974680, 1.0096176, -0.0104260, 0.0096936
2: -0.0051079, 0.0050721, -0.0055638, 0.0054352, -0.0105431, 0.0106359
3: 0.0000357, 0.0024057, -0.0000624, 0.0023940, -0.0017860, 0.0018920
4: -0.0060382, 0.0013392, -0.0064997, 0.0014581, -0.0074963, 0.0078389
5: -0.0019513, 0.0071163, -0.0021110, 0.0076937, -0.0096450, 0.0092272
6: -0.0076929, 0.0018511, -0.0084057, 0.0019075, -0.0096004, 0.0102568
7: -0.0055854, -0.0004039, -0.0055657, -0.0001496, -0.0054358, 0.0051618
8: -0.0128246, -0.0023810, -0.0131801, -0.0024419, -0.0103827, 0.0107991
9: -0.0033827, 0.0071865, -0.0040185, 0.0073951, -0.0107778, 0.0112051

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090251, upper bound: 0.0090846
time: 1.12 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092841, upper bound: 0.0093259
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0059963, 0.0041211, -0.0067586, 0.0051190, -0.0111153, 0.0108797
1: 0.9974394, 1.0089213, 0.9974827, 1.0104350, -0.0112610, 0.0096778
2: -0.0051079, 0.0050721, -0.0060991, 0.0058615, -0.0109694, 0.0111711
3: 0.0000357, 0.0024057, -0.0001777, 0.0023878, -0.0017809, 0.0020335
4: -0.0060382, 0.0013392, -0.0070415, 0.0015978, -0.0076360, 0.0083807
5: -0.0019513, 0.0071163, -0.0022985, 0.0083716, -0.0103229, 0.0094147
6: -0.0076929, 0.0018511, -0.0092426, 0.0019738, -0.0096666, 0.0110937
7: -0.0055854, -0.0004039, -0.0055552, 0.0001490, -0.0057344, 0.0051513
8: -0.0128246, -0.0023810, -0.0135974, -0.0024742, -0.0103504, 0.0112165
9: -0.0033827, 0.0071865, -0.0047650, 0.0076399, -0.0110226, 0.0119516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090251, upper bound: 0.0090846
time: 0.93 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092841, upper bound: 0.0093259
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0062316, 0.0044292, -0.0062212, 0.0044156, -0.0106472, 0.0106504
1: 0.9974701, 1.0093888, 0.9974751, 1.0093682, -0.0102310, 0.0101173
2: -0.0054139, 0.0053158, -0.0054004, 0.0053050, -0.0107189, 0.0107161
3: -0.0000302, 0.0023930, -0.0000273, 0.0023910, -0.0018249, 0.0018908
4: -0.0063479, 0.0014190, -0.0063343, 0.0014155, -0.0077634, 0.0077533
5: -0.0020584, 0.0075038, -0.0020537, 0.0074867, -0.0095452, 0.0095575
6: -0.0081713, 0.0018890, -0.0081502, 0.0018873, -0.0100586, 0.0100391
7: -0.0055640, -0.0002332, -0.0055606, -0.0002407, -0.0053233, 0.0053274
8: -0.0130632, -0.0024470, -0.0130527, -0.0024577, -0.0106055, 0.0106057
9: -0.0038094, 0.0073265, -0.0037906, 0.0073203, -0.0111297, 0.0111171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090399, upper bound: 0.0091393
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093138, upper bound: 0.0093854
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0062316, 0.0044292, -0.0066503, 0.0049773, -0.0112089, 0.0110794
1: 0.9974701, 1.0093888, 0.9974836, 1.0102201, -0.0111111, 0.0100825
2: -0.0054139, 0.0053158, -0.0059583, 0.0057493, -0.0111632, 0.0112740
3: -0.0000302, 0.0023930, -0.0001474, 0.0023875, -0.0018140, 0.0020432
4: -0.0063479, 0.0014190, -0.0068990, 0.0015610, -0.0079090, 0.0083180
5: -0.0020584, 0.0075038, -0.0022491, 0.0081933, -0.0102517, 0.0097529
6: -0.0081713, 0.0018890, -0.0090225, 0.0019563, -0.0101276, 0.0109114
7: -0.0055640, -0.0002332, -0.0055547, 0.0000705, -0.0056345, 0.0053214
8: -0.0130632, -0.0024470, -0.0134876, -0.0024760, -0.0105872, 0.0110406
9: -0.0038094, 0.0073265, -0.0045687, 0.0075755, -0.0113849, 0.0118952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090399, upper bound: 0.0091393
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093138, upper bound: 0.0093854
time: 1.61 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0062480, 0.0044507, -0.0062567, 0.0044620, -0.0107100, 0.0107073
1: 0.9974697, 1.0094213, 0.9974692, 1.0094385, -0.0102249, 0.0101903
2: -0.0054352, 0.0053327, -0.0054465, 0.0053417, -0.0107769, 0.0107792
3: -0.0000347, 0.0023932, -0.0000372, 0.0023934, -0.0018458, 0.0018692
4: -0.0063695, 0.0014246, -0.0063809, 0.0014275, -0.0077970, 0.0078055
5: -0.0020659, 0.0075308, -0.0020699, 0.0075450, -0.0096110, 0.0096007
6: -0.0082046, 0.0018916, -0.0082222, 0.0018930, -0.0100976, 0.0101138
7: -0.0055644, -0.0002213, -0.0055647, -0.0002150, -0.0053494, 0.0053434
8: -0.0130798, -0.0024459, -0.0130886, -0.0024449, -0.0106349, 0.0106427
9: -0.0038392, 0.0073363, -0.0038549, 0.0073414, -0.0111806, 0.0111911

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090644, upper bound: 0.0091465
time: 0.82 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093459, upper bound: 0.0094070
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0062480, 0.0044507, -0.0066624, 0.0049931, -0.0112412, 0.0111131
1: 0.9974697, 1.0094213, 0.9974846, 1.0102441, -0.0110585, 0.0101094
2: -0.0054352, 0.0053327, -0.0059740, 0.0057619, -0.0111971, 0.0113068
3: -0.0000347, 0.0023932, -0.0001508, 0.0023870, -0.0018195, 0.0020188
4: -0.0063695, 0.0014246, -0.0069149, 0.0015651, -0.0079347, 0.0083395
5: -0.0020659, 0.0075308, -0.0022547, 0.0082133, -0.0102792, 0.0097855
6: -0.0082046, 0.0018916, -0.0090471, 0.0019583, -0.0101629, 0.0109387
7: -0.0055644, -0.0002213, -0.0055539, 0.0000793, -0.0056437, 0.0053326
8: -0.0130798, -0.0024459, -0.0134999, -0.0024782, -0.0106016, 0.0110540
9: -0.0038392, 0.0073363, -0.0045907, 0.0075827, -0.0114219, 0.0119269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090644, upper bound: 0.0091465
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093459, upper bound: 0.0094070
time: 0.96 seconds

## BFS IS instance: IS_A2_A1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0059520, 0.0040631, -0.0061755, 0.0043557, -0.0103078, 0.0102387
1: 0.9972658, 1.0088334, 0.9974042, 1.0092773, -0.0101306, 0.0097051
2: -0.0050503, 0.0050262, -0.0053409, 0.0052577, -0.0103080, 0.0103671
3: 0.0000481, 0.0024777, -0.0000144, 0.0024204, -0.0018135, 0.0018606
4: -0.0059799, 0.0013242, -0.0062741, 0.0014000, -0.0073799, 0.0075983
5: -0.0019311, 0.0070433, -0.0020329, 0.0074114, -0.0093425, 0.0090762
6: -0.0076028, 0.0018440, -0.0080572, 0.0018800, -0.0094828, 0.0099012
7: -0.0057069, -0.0004360, -0.0056102, -0.0002739, -0.0054330, 0.0051742
8: -0.0127797, -0.0020052, -0.0130063, -0.0023042, -0.0104755, 0.0110011
9: -0.0033024, 0.0071602, -0.0037077, 0.0072931, -0.0105955, 0.0108679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A1_A1_A1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090220, upper bound: 0.0089867
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090220, upper bound: 0.0089867
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0061964, 0.0043830, -0.0063576, 0.0045941, -0.0107905, 0.0107407
1: 0.9974524, 1.0093187, 0.9973992, 1.0096391, -0.0104303, 0.0101730
2: -0.0053681, 0.0052793, -0.0055777, 0.0054463, -0.0108143, 0.0108570
3: -0.0000203, 0.0024004, -0.0000654, 0.0024225, -0.0018904, 0.0018970
4: -0.0063015, 0.0014071, -0.0065138, 0.0014618, -0.0077633, 0.0079208
5: -0.0020424, 0.0074458, -0.0021158, 0.0077113, -0.0097537, 0.0095616
6: -0.0080996, 0.0018833, -0.0084275, 0.0019093, -0.0100089, 0.0103108
7: -0.0055765, -0.0002588, -0.0056137, -0.0001418, -0.0054346, 0.0053550
8: -0.0130274, -0.0024086, -0.0131909, -0.0022934, -0.0107341, 0.0107823
9: -0.0037455, 0.0073055, -0.0040379, 0.0074014, -0.0111469, 0.0113435

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A1_A1_A2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092828, upper bound: 0.0092452
time: 0.97 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092828, upper bound: 0.0092452
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0062103, 0.0044013, -0.0058976, 0.0039919, -0.0102022, 0.0102989
1: 0.9974586, 1.0093466, 0.9972403, 1.0087256, -0.0095611, 0.0102700
2: -0.0053862, 0.0052937, -0.0049796, 0.0049698, -0.0103561, 0.0102733
3: -0.0000242, 0.0023978, 0.0000634, 0.0024884, -0.0019109, 0.0017743
4: -0.0063200, 0.0014118, -0.0059083, 0.0013057, -0.0076257, 0.0073201
5: -0.0020488, 0.0074688, -0.0019063, 0.0069537, -0.0090025, 0.0093751
6: -0.0081280, 0.0018856, -0.0074921, 0.0018352, -0.0099633, 0.0093777
7: -0.0055721, -0.0002486, -0.0057249, -0.0004755, -0.0050966, 0.0054763
8: -0.0130416, -0.0024220, -0.0127245, -0.0019496, -0.0110920, 0.0103025
9: -0.0037709, 0.0073139, -0.0032036, 0.0071278, -0.0108987, 0.0105175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090298, upper bound: 0.0090124
time: 0.79 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090298, upper bound: 0.0090124
time: 1.15 seconds

## BFS IS instance: IS_A2_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0063875, 0.0046333, -0.0061767, 0.0043573, -0.0107448, 0.0108100
1: 0.9974533, 1.0096984, 0.9974035, 1.0092797, -0.0100822, 0.0105621
2: -0.0056166, 0.0054772, -0.0053425, 0.0052589, -0.0108755, 0.0108198
3: -0.0000738, 0.0024000, -0.0000148, 0.0024207, -0.0019502, 0.0018473
4: -0.0065531, 0.0014719, -0.0062757, 0.0014004, -0.0079535, 0.0077476
5: -0.0021295, 0.0077605, -0.0020335, 0.0074134, -0.0095429, 0.0097940
6: -0.0084882, 0.0019141, -0.0080597, 0.0018802, -0.0103684, 0.0099738
7: -0.0055758, -0.0001201, -0.0056107, -0.0002730, -0.0053028, 0.0054906
8: -0.0132212, -0.0024106, -0.0130075, -0.0023026, -0.0109186, 0.0105969
9: -0.0040921, 0.0074192, -0.0037099, 0.0072939, -0.0113860, 0.0111291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092877, upper bound: 0.0092579
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092877, upper bound: 0.0092579
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0063027, 0.0045223, -0.0059434, 0.0040519, -0.0103546, 0.0104657
1: 0.9974922, 1.0095299, 0.9972396, 1.0088165, -0.0096465, 0.0104244
2: -0.0055063, 0.0053894, -0.0050392, 0.0050173, -0.0105236, 0.0104286
3: -0.0000501, 0.0023838, 0.0000505, 0.0024886, -0.0019200, 0.0017882
4: -0.0064415, 0.0014431, -0.0059686, 0.0013213, -0.0077628, 0.0074118
5: -0.0020908, 0.0076209, -0.0019272, 0.0070292, -0.0091200, 0.0095481
6: -0.0083158, 0.0019004, -0.0075854, 0.0018426, -0.0101584, 0.0094858
7: -0.0055485, -0.0001817, -0.0057253, -0.0004423, -0.0051062, 0.0055437
8: -0.0131353, -0.0024951, -0.0127710, -0.0019484, -0.0111869, 0.0102760
9: -0.0039383, 0.0073688, -0.0032868, 0.0071551, -0.0110934, 0.0106556

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A2_A1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090491, upper bound: 0.0090402
time: 0.92 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090491, upper bound: 0.0090402
time: 1.06 seconds

## BFS IS instance: IS_A2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0064819, 0.0047568, -0.0062178, 0.0044110, -0.0108929, 0.0109746
1: 0.9974869, 1.0098858, 0.9974028, 1.0093614, -0.0101561, 0.0107012
2: -0.0057393, 0.0055750, -0.0053959, 0.0053014, -0.0110407, 0.0109708
3: -0.0001002, 0.0023861, -0.0000263, 0.0024210, -0.0019545, 0.0018615
4: -0.0066774, 0.0015039, -0.0063297, 0.0014143, -0.0080917, 0.0078336
5: -0.0021725, 0.0079160, -0.0020521, 0.0074810, -0.0096534, 0.0099681
6: -0.0086801, 0.0019293, -0.0081431, 0.0018868, -0.0105669, 0.0100723
7: -0.0055524, -0.0000517, -0.0056112, -0.0002433, -0.0053091, 0.0055595
8: -0.0133169, -0.0024831, -0.0130491, -0.0023012, -0.0110157, 0.0105661
9: -0.0042633, 0.0074754, -0.0037843, 0.0073183, -0.0115816, 0.0112597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A2_A1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093250, upper bound: 0.0093131
time: 0.89 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093250, upper bound: 0.0093131
time: 1.20 seconds

## BFS IS instance: IS_A2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0063101, 0.0045319, -0.0059512, 0.0040620, -0.0103721, 0.0104831
1: 0.9974931, 1.0095445, 0.9972391, 1.0088320, -0.0097272, 0.0104000
2: -0.0055159, 0.0053970, -0.0050492, 0.0050253, -0.0105412, 0.0104462
3: -0.0000521, 0.0023835, 0.0000484, 0.0024888, -0.0019099, 0.0018178
4: -0.0064512, 0.0014456, -0.0059788, 0.0013239, -0.0077751, 0.0074244
5: -0.0020942, 0.0076330, -0.0019307, 0.0070419, -0.0091361, 0.0095637
6: -0.0083308, 0.0019016, -0.0076011, 0.0018439, -0.0101746, 0.0095027
7: -0.0055480, -0.0001763, -0.0057257, -0.0004366, -0.0051113, 0.0055494
8: -0.0131427, -0.0024966, -0.0127788, -0.0019472, -0.0111955, 0.0102822
9: -0.0039517, 0.0073732, -0.0033008, 0.0071597, -0.0111114, 0.0106740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090696, upper bound: 0.0090737
time: 0.91 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090696, upper bound: 0.0090737
time: 0.84 seconds

## BFS IS instance: IS_A2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0065001, 0.0047806, -0.0062298, 0.0044269, -0.0109269, 0.0110105
1: 0.9974878, 1.0099219, 0.9974024, 1.0093851, -0.0102658, 0.0107037
2: -0.0057630, 0.0055938, -0.0054116, 0.0053139, -0.0110769, 0.0110054
3: -0.0001053, 0.0023857, -0.0000297, 0.0024212, -0.0019495, 0.0018976
4: -0.0067013, 0.0015101, -0.0063456, 0.0014184, -0.0081197, 0.0078557
5: -0.0021807, 0.0079459, -0.0020576, 0.0075009, -0.0096816, 0.0100036
6: -0.0087171, 0.0019322, -0.0081677, 0.0018887, -0.0106058, 0.0100998
7: -0.0055517, -0.0000385, -0.0056115, -0.0002345, -0.0053172, 0.0055730
8: -0.0133354, -0.0024851, -0.0130614, -0.0023003, -0.0110351, 0.0105762
9: -0.0042963, 0.0074862, -0.0038062, 0.0073254, -0.0116217, 0.0112924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 118

## Relational analysis of IS_A2_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093513, upper bound: 0.0093457
time: 0.88 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093513, upper bound: 0.0093457
time: 0.95 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.36 seconds
IS_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090164, upper bound: 0.0090833
IS_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092797, upper bound: 0.0093352
IS_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090164, upper bound: 0.0090833
IS_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092797, upper bound: 0.0093352
IS_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090251, upper bound: 0.0090846
IS_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092841, upper bound: 0.0093259
IS_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090251, upper bound: 0.0090846
IS_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092841, upper bound: 0.0093259
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090399, upper bound: 0.0091393
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093138, upper bound: 0.0093854
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090399, upper bound: 0.0091393
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093138, upper bound: 0.0093854
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090644, upper bound: 0.0091465
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093459, upper bound: 0.0094070
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090644, upper bound: 0.0091465
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093459, upper bound: 0.0094070
IS_A2_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090220, upper bound: 0.0089867
IS_A2_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090220, upper bound: 0.0089867
IS_A2_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092828, upper bound: 0.0092452
IS_A2_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092828, upper bound: 0.0092452
IS_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090298, upper bound: 0.0090124
IS_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090298, upper bound: 0.0090124
IS_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092877, upper bound: 0.0092579
IS_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0092877, upper bound: 0.0092579
IS_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090491, upper bound: 0.0090402
IS_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090491, upper bound: 0.0090402
IS_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093250, upper bound: 0.0093131
IS_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093250, upper bound: 0.0093131
IS_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090696, upper bound: 0.0090737
IS_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0090696, upper bound: 0.0090737
IS_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093513, upper bound: 0.0093457
IS_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.36
Output dim: 1, lower bound: -0.0093513, upper bound: 0.0093457

## BFS IS instance: IS_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0057989, 0.0038626, -0.0058758, 0.0039634, -0.0097622, 0.0097384
1: 0.9974375, 1.0085293, 0.9973211, 1.0086824, -0.0094305, 0.0094601
2: -0.0048512, 0.0048676, -0.0049512, 0.0049472, -0.0097984, 0.0098188
3: 0.0000910, 0.0024066, 0.0000695, 0.0024548, -0.0017812, 0.0017208
4: -0.0057783, 0.0012722, -0.0058796, 0.0012983, -0.0070766, 0.0071518
5: -0.0018613, 0.0067911, -0.0018964, 0.0069178, -0.0087791, 0.0086875
6: -0.0072914, 0.0018194, -0.0074478, 0.0018317, -0.0091231, 0.0092672
7: -0.0055870, -0.0005471, -0.0056684, -0.0004913, -0.0050957, 0.0051212
8: -0.0126244, -0.0023761, -0.0127024, -0.0021245, -0.0104999, 0.0103264
9: -0.0030246, 0.0070691, -0.0031641, 0.0071149, -0.0101395, 0.0102332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090419, upper bound: 0.0088572
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089237, upper bound: 0.0088602
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0059741, 0.0040920, -0.0061491, 0.0043212, -0.0102952, 0.0102411
1: 0.9974322, 1.0088774, 0.9974732, 1.0092248, -0.0099093, 0.0097096
2: -0.0050790, 0.0050490, -0.0053066, 0.0052303, -0.0103093, 0.0103556
3: 0.0000419, 0.0024088, -0.0000071, 0.0023918, -0.0018029, 0.0017853
4: -0.0060090, 0.0013316, -0.0062393, 0.0013910, -0.0074000, 0.0075710
5: -0.0019411, 0.0070797, -0.0020209, 0.0073679, -0.0093091, 0.0091005
6: -0.0076476, 0.0018475, -0.0080035, 0.0018757, -0.0095233, 0.0098511
7: -0.0055906, -0.0004200, -0.0055619, -0.0002931, -0.0052975, 0.0051419
8: -0.0128021, -0.0023649, -0.0129795, -0.0024535, -0.0103486, 0.0106147
9: -0.0033424, 0.0071733, -0.0036598, 0.0072774, -0.0106198, 0.0108331

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0092774
time: 0.77 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0094809
time: 1.28 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0057989, 0.0038626, -0.0062989, 0.0045173, -0.0103162, 0.0101616
1: 0.9974375, 1.0085293, 0.9973267, 1.0095223, -0.0102854, 0.0093806
2: -0.0048512, 0.0048676, -0.0055014, 0.0053855, -0.0102367, 0.0103690
3: 0.0000910, 0.0024066, -0.0000490, 0.0024525, -0.0017512, 0.0018578
4: -0.0057783, 0.0012722, -0.0064365, 0.0014419, -0.0072202, 0.0077087
5: -0.0018613, 0.0067911, -0.0020891, 0.0076147, -0.0094760, 0.0088802
6: -0.0072914, 0.0018194, -0.0083082, 0.0018998, -0.0091912, 0.0101275
7: -0.0055870, -0.0005471, -0.0056644, -0.0001844, -0.0054026, 0.0051172
8: -0.0126244, -0.0023761, -0.0131314, -0.0021368, -0.0104876, 0.0107554
9: -0.0030246, 0.0070691, -0.0039315, 0.0073665, -0.0103911, 0.0110006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087965, upper bound: 0.0089164
time: 1.19 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087892, upper bound: 0.0087632
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0059741, 0.0040920, -0.0065558, 0.0048536, -0.0108277, 0.0106478
1: 0.9974322, 1.0088774, 0.9974881, 1.0100325, -0.0107567, 0.0096374
2: -0.0050790, 0.0050490, -0.0058354, 0.0056515, -0.0107305, 0.0108845
3: 0.0000419, 0.0024088, -0.0001209, 0.0023855, -0.0017782, 0.0019320
4: -0.0060090, 0.0013316, -0.0067747, 0.0015290, -0.0075379, 0.0081063
5: -0.0019411, 0.0070797, -0.0022061, 0.0080377, -0.0099789, 0.0092858
6: -0.0076476, 0.0018475, -0.0088304, 0.0019412, -0.0095888, 0.0106780
7: -0.0055906, -0.0004200, -0.0055514, 0.0000020, -0.0055925, 0.0051313
8: -0.0128021, -0.0023649, -0.0133919, -0.0024862, -0.0103159, 0.0110270
9: -0.0033424, 0.0071733, -0.0043974, 0.0075193, -0.0108617, 0.0115707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090169, upper bound: 0.0091018
time: 0.72 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090169, upper bound: 0.0093352
time: 0.93 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0058088, 0.0038756, -0.0058797, 0.0039685, -0.0097773, 0.0097554
1: 0.9974449, 1.0085493, 0.9973204, 1.0086900, -0.0094845, 0.0093924
2: -0.0048641, 0.0048779, -0.0049563, 0.0049513, -0.0098154, 0.0098342
3: 0.0000882, 0.0024035, 0.0000684, 0.0024551, -0.0017490, 0.0017426
4: -0.0057914, 0.0012756, -0.0058848, 0.0012996, -0.0070911, 0.0071604
5: -0.0018659, 0.0068075, -0.0018982, 0.0069243, -0.0087902, 0.0087057
6: -0.0073116, 0.0018210, -0.0074559, 0.0018324, -0.0091440, 0.0092768
7: -0.0055817, -0.0005399, -0.0056688, -0.0004885, -0.0050933, 0.0051289
8: -0.0126345, -0.0023923, -0.0127064, -0.0021231, -0.0105114, 0.0103142
9: -0.0030426, 0.0070750, -0.0031713, 0.0071172, -0.0101598, 0.0102463

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089296, upper bound: 0.0090086
time: 1.09 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089257, upper bound: 0.0088578
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0059922, 0.0041158, -0.0061573, 0.0043319, -0.0103241, 0.0102731
1: 0.9974397, 1.0089134, 0.9974726, 1.0092413, -0.0100023, 0.0096814
2: -0.0051026, 0.0050678, -0.0053173, 0.0052388, -0.0103414, 0.0103851
3: 0.0000369, 0.0024056, -0.0000094, 0.0023920, -0.0017812, 0.0018150
4: -0.0060328, 0.0013378, -0.0062502, 0.0013938, -0.0074266, 0.0075880
5: -0.0019494, 0.0071095, -0.0020246, 0.0073814, -0.0093309, 0.0091341
6: -0.0076845, 0.0018505, -0.0080202, 0.0018770, -0.0095616, 0.0098707
7: -0.0055853, -0.0004069, -0.0055623, -0.0002871, -0.0052982, 0.0051554
8: -0.0128205, -0.0023812, -0.0129879, -0.0024523, -0.0103682, 0.0106067
9: -0.0033752, 0.0071841, -0.0036747, 0.0072823, -0.0106576, 0.0108588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0092341
time: 1.26 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0094617
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0058088, 0.0038756, -0.0062971, 0.0045148, -0.0103236, 0.0101727
1: 0.9974449, 1.0085493, 0.9973263, 1.0095187, -0.0103202, 0.0093668
2: -0.0048641, 0.0048779, -0.0054990, 0.0053835, -0.0102476, 0.0103768
3: 0.0000882, 0.0024035, -0.0000485, 0.0024527, -0.0017397, 0.0018755
4: -0.0057914, 0.0012756, -0.0064341, 0.0014412, -0.0072326, 0.0077096
5: -0.0018659, 0.0068075, -0.0020883, 0.0076116, -0.0094774, 0.0088957
6: -0.0073116, 0.0018210, -0.0083043, 0.0018995, -0.0092111, 0.0101253
7: -0.0055817, -0.0005399, -0.0056648, -0.0001858, -0.0053960, 0.0051249
8: -0.0126345, -0.0023923, -0.0131295, -0.0021356, -0.0104989, 0.0107372
9: -0.0030426, 0.0070750, -0.0039281, 0.0073654, -0.0104080, 0.0110031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088011, upper bound: 0.0089196
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087956, upper bound: 0.0087608
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0059922, 0.0041158, -0.0065623, 0.0048620, -0.0108542, 0.0106780
1: 0.9974397, 1.0089134, 0.9974879, 1.0100453, -0.0108339, 0.0096651
2: -0.0051026, 0.0050678, -0.0058438, 0.0056582, -0.0107608, 0.0109116
3: 0.0000369, 0.0024056, -0.0001227, 0.0023857, -0.0017760, 0.0019576
4: -0.0060328, 0.0013378, -0.0067831, 0.0015312, -0.0075640, 0.0081209
5: -0.0019494, 0.0071095, -0.0022090, 0.0080483, -0.0099977, 0.0093186
6: -0.0076845, 0.0018505, -0.0088435, 0.0019422, -0.0096267, 0.0106939
7: -0.0055853, -0.0004069, -0.0055517, 0.0000066, -0.0055920, 0.0051448
8: -0.0128205, -0.0023812, -0.0133984, -0.0024852, -0.0103353, 0.0110172
9: -0.0033752, 0.0071841, -0.0044090, 0.0075231, -0.0108984, 0.0115931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090290, upper bound: 0.0090892
time: 0.93 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090290, upper bound: 0.0093259
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0060470, 0.0041875, -0.0057546, 0.0038047, -0.0098517, 0.0099422
1: 0.9974751, 1.0090221, 0.9973109, 1.0084417, -0.0092949, 0.0098076
2: -0.0051739, 0.0051246, -0.0047936, 0.0048217, -0.0099956, 0.0099182
3: 0.0000215, 0.0023909, 0.0001034, 0.0024590, -0.0017834, 0.0017369
4: -0.0061050, 0.0013564, -0.0057201, 0.0012572, -0.0073622, 0.0070765
5: -0.0019744, 0.0071998, -0.0018412, 0.0067182, -0.0086926, 0.0090410
6: -0.0077960, 0.0018593, -0.0072014, 0.0018122, -0.0096082, 0.0090607
7: -0.0055605, -0.0003671, -0.0056754, -0.0005792, -0.0049813, 0.0053083
8: -0.0128761, -0.0024580, -0.0125796, -0.0021028, -0.0107733, 0.0101216
9: -0.0034747, 0.0072167, -0.0029443, 0.0070428, -0.0105175, 0.0101611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089771, upper bound: 0.0090837
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089772, upper bound: 0.0089530
time: 1.09 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0062274, 0.0044237, -0.0060239, 0.0041572, -0.0103847, 0.0104476
1: 0.9974702, 1.0093803, 0.9974802, 1.0089765, -0.0098062, 0.0101052
2: -0.0054085, 0.0053114, -0.0051438, 0.0051006, -0.0105091, 0.0104552
3: -0.0000290, 0.0023930, 0.0000280, 0.0023888, -0.0018200, 0.0018181
4: -0.0063424, 0.0014176, -0.0060745, 0.0013485, -0.0076910, 0.0074921
5: -0.0020566, 0.0074969, -0.0019638, 0.0071617, -0.0092183, 0.0094608
6: -0.0081628, 0.0018883, -0.0077490, 0.0018556, -0.0100184, 0.0096373
7: -0.0055640, -0.0002362, -0.0055570, -0.0003839, -0.0051801, 0.0053207
8: -0.0130590, -0.0024472, -0.0128526, -0.0024689, -0.0105901, 0.0104054
9: -0.0038019, 0.0073240, -0.0034327, 0.0072030, -0.0110048, 0.0107567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093034, upper bound: 0.0092976
time: 0.95 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093034, upper bound: 0.0092976
time: 0.85 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0060470, 0.0041875, -0.0061818, 0.0043640, -0.0104111, 0.0103694
1: 0.9974751, 1.0090221, 0.9973080, 1.0092900, -0.0101594, 0.0097835
2: -0.0051739, 0.0051246, -0.0053492, 0.0052642, -0.0104381, 0.0104738
3: 0.0000215, 0.0023909, -0.0000162, 0.0024603, -0.0017753, 0.0018767
4: -0.0061050, 0.0013564, -0.0062824, 0.0014021, -0.0075071, 0.0076388
5: -0.0019744, 0.0071998, -0.0020358, 0.0074219, -0.0093962, 0.0092356
6: -0.0077960, 0.0018593, -0.0080701, 0.0018810, -0.0096770, 0.0099294
7: -0.0055605, -0.0003671, -0.0056775, -0.0002693, -0.0052912, 0.0053104
8: -0.0128761, -0.0024580, -0.0130127, -0.0020963, -0.0107798, 0.0105548
9: -0.0034747, 0.0072167, -0.0037192, 0.0072969, -0.0107716, 0.0109359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088328, upper bound: 0.0089878
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088284, upper bound: 0.0088489
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0062274, 0.0044237, -0.0064444, 0.0047077, -0.0109352, 0.0108681
1: 0.9974702, 1.0093803, 0.9974887, 1.0098112, -0.0106786, 0.0100699
2: -0.0054085, 0.0053114, -0.0056905, 0.0055361, -0.0109446, 0.0110020
3: -0.0000290, 0.0023930, -0.0000897, 0.0023853, -0.0018089, 0.0019691
4: -0.0063424, 0.0014176, -0.0066280, 0.0014912, -0.0078336, 0.0080456
5: -0.0020566, 0.0074969, -0.0021554, 0.0078542, -0.0099107, 0.0096523
6: -0.0081628, 0.0018883, -0.0086038, 0.0019232, -0.0100860, 0.0104922
7: -0.0055640, -0.0002362, -0.0055510, -0.0000789, -0.0054851, 0.0053148
8: -0.0130590, -0.0024472, -0.0132789, -0.0024873, -0.0105716, 0.0108317
9: -0.0038019, 0.0073240, -0.0041953, 0.0074530, -0.0112549, 0.0115193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090427, upper bound: 0.0091370
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090427, upper bound: 0.0093854
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0060577, 0.0042015, -0.0057726, 0.0038283, -0.0098860, 0.0099741
1: 0.9974747, 1.0090435, 0.9973218, 1.0084774, -0.0092587, 0.0099078
2: -0.0051877, 0.0051356, -0.0048171, 0.0048404, -0.0100281, 0.0099527
3: 0.0000185, 0.0023912, 0.0000983, 0.0024545, -0.0018115, 0.0017148
4: -0.0061190, 0.0013600, -0.0057438, 0.0012633, -0.0073823, 0.0071038
5: -0.0019792, 0.0072174, -0.0018494, 0.0067479, -0.0087271, 0.0090668
6: -0.0078177, 0.0018610, -0.0072381, 0.0018151, -0.0096328, 0.0090991
7: -0.0055609, -0.0003594, -0.0056678, -0.0005662, -0.0049947, 0.0053084
8: -0.0128869, -0.0024568, -0.0125978, -0.0021262, -0.0107606, 0.0101410
9: -0.0034940, 0.0072231, -0.0029770, 0.0070535, -0.0105475, 0.0102001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090835, upper bound: 0.0089577
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089775, upper bound: 0.0089581
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0062439, 0.0044453, -0.0060593, 0.0042036, -0.0104475, 0.0105046
1: 0.9974697, 1.0094131, 0.9974740, 1.0090467, -0.0097892, 0.0101780
2: -0.0054299, 0.0053285, -0.0051898, 0.0051373, -0.0105672, 0.0105183
3: -0.0000336, 0.0023932, 0.0000181, 0.0023914, -0.0018409, 0.0017901
4: -0.0063641, 0.0014232, -0.0061212, 0.0013606, -0.0077247, 0.0075443
5: -0.0020641, 0.0075241, -0.0019800, 0.0072201, -0.0092841, 0.0095040
6: -0.0081963, 0.0018910, -0.0078210, 0.0018613, -0.0100575, 0.0097119
7: -0.0055643, -0.0002243, -0.0055613, -0.0003582, -0.0052061, 0.0053370
8: -0.0130756, -0.0024461, -0.0128885, -0.0024554, -0.0106203, 0.0104424
9: -0.0038317, 0.0073338, -0.0034970, 0.0072240, -0.0110557, 0.0108308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093016, upper bound: 0.0092976
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0093016, upper bound: 0.0095444
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0060577, 0.0042015, -0.0061756, 0.0043559, -0.0104136, 0.0103771
1: 0.9974747, 1.0090435, 0.9973282, 1.0092777, -0.0100818, 0.0098268
2: -0.0051877, 0.0051356, -0.0053411, 0.0052578, -0.0104455, 0.0104767
3: 0.0000185, 0.0023912, -0.0000145, 0.0024519, -0.0017825, 0.0018518
4: -0.0061190, 0.0013600, -0.0062743, 0.0014000, -0.0075190, 0.0076343
5: -0.0019792, 0.0072174, -0.0020330, 0.0074116, -0.0093909, 0.0092503
6: -0.0078177, 0.0018610, -0.0080575, 0.0018800, -0.0096977, 0.0099185
7: -0.0055609, -0.0003594, -0.0056634, -0.0002738, -0.0052871, 0.0053041
8: -0.0128869, -0.0024568, -0.0130064, -0.0021397, -0.0107471, 0.0105496
9: -0.0034940, 0.0072231, -0.0037079, 0.0072932, -0.0107872, 0.0109310

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088620, upper bound: 0.0089979
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088577, upper bound: 0.0088628
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0062439, 0.0044453, -0.0064631, 0.0047322, -0.0109761, 0.0109083
1: 0.9974697, 1.0094131, 0.9974896, 1.0098482, -0.0106216, 0.0100970
2: -0.0054299, 0.0053285, -0.0057148, 0.0055555, -0.0109853, 0.0110433
3: -0.0000336, 0.0023932, -0.0000950, 0.0023849, -0.0018144, 0.0019386
4: -0.0063641, 0.0014232, -0.0066526, 0.0014975, -0.0078616, 0.0080758
5: -0.0020641, 0.0075241, -0.0021639, 0.0078849, -0.0099490, 0.0096879
6: -0.0081963, 0.0018910, -0.0086418, 0.0019262, -0.0101225, 0.0105328
7: -0.0055643, -0.0002243, -0.0055504, -0.0000653, -0.0054990, 0.0053261
8: -0.0130756, -0.0024461, -0.0132978, -0.0024892, -0.0105864, 0.0108517
9: -0.0038317, 0.0073338, -0.0042291, 0.0074642, -0.0112959, 0.0115630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090737, upper bound: 0.0091436
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090737, upper bound: 0.0091436
time: 1.14 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0059520, 0.0040631, -0.0061565, 0.0043308, -0.0102828, 0.0102196
1: 0.9972658, 1.0088334, 0.9974734, 1.0092394, -0.0100282, 0.0096830
2: -0.0050503, 0.0050262, -0.0053162, 0.0052379, -0.0102882, 0.0103423
3: 0.0000481, 0.0024777, -0.0000091, 0.0023917, -0.0018046, 0.0018214
4: -0.0059799, 0.0013242, -0.0062490, 0.0013935, -0.0073734, 0.0075732
5: -0.0019311, 0.0070433, -0.0020242, 0.0073800, -0.0093111, 0.0090676
6: -0.0076028, 0.0018440, -0.0080185, 0.0018769, -0.0094797, 0.0098625
7: -0.0057069, -0.0004360, -0.0055617, -0.0002877, -0.0054192, 0.0051257
8: -0.0127797, -0.0020052, -0.0129870, -0.0024541, -0.0103256, 0.0109817
9: -0.0033024, 0.0071602, -0.0036731, 0.0072818, -0.0105842, 0.0108333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_A1_B1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088814, upper bound: 0.0087691
time: 0.87 seconds

## Relational analysis of IS_A2_A1_A1_A1_B1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087680, upper bound: 0.0087660
time: 0.92 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0059520, 0.0040631, -0.0065770, 0.0048814, -0.0108334, 0.0106402
1: 0.9972658, 1.0088334, 0.9974884, 1.0100745, -0.0108032, 0.0095736
2: -0.0050503, 0.0050262, -0.0058630, 0.0056735, -0.0107239, 0.0108892
3: 0.0000481, 0.0024777, -0.0001269, 0.0023854, -0.0017601, 0.0019229
4: -0.0059799, 0.0013242, -0.0068026, 0.0015362, -0.0075161, 0.0081268
5: -0.0019311, 0.0070433, -0.0022158, 0.0080727, -0.0100038, 0.0092591
6: -0.0076028, 0.0018440, -0.0088736, 0.0019446, -0.0095474, 0.0107176
7: -0.0057069, -0.0004360, -0.0055512, 0.0000174, -0.0057243, 0.0051152
8: -0.0127797, -0.0020052, -0.0134134, -0.0024867, -0.0102930, 0.0114081
9: -0.0033024, 0.0071602, -0.0044359, 0.0075320, -0.0108343, 0.0115961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_A1_B2_A1

### Relational analysis result of IS_A2_A1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087695, upper bound: 0.0088467
time: 0.89 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2_A2

### Relational analysis result of IS_A2_A1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087680, upper bound: 0.0087660
time: 0.85 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0061964, 0.0043830, -0.0063386, 0.0045692, -0.0107656, 0.0107216
1: 0.9974524, 1.0093187, 0.9974685, 1.0096011, -0.0103154, 0.0101518
2: -0.0053681, 0.0052793, -0.0055530, 0.0054265, -0.0107946, 0.0108322
3: -0.0000203, 0.0024004, -0.0000601, 0.0023937, -0.0018846, 0.0018550
4: -0.0063015, 0.0014071, -0.0064887, 0.0014553, -0.0077568, 0.0078958
5: -0.0020424, 0.0074458, -0.0021072, 0.0076799, -0.0097223, 0.0095529
6: -0.0080996, 0.0018833, -0.0083887, 0.0019062, -0.0100058, 0.0102721
7: -0.0055765, -0.0002588, -0.0055652, -0.0001556, -0.0054208, 0.0053065
8: -0.0130274, -0.0024086, -0.0131716, -0.0024433, -0.0105841, 0.0107630
9: -0.0037455, 0.0073055, -0.0040034, 0.0073901, -0.0111356, 0.0113089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_A1_A2_B1_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0089824
time: 0.91 seconds

## Relational analysis of IS_A2_A1_A1_A2_B1_B2

### Relational analysis result of IS_A2_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0092401
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0061964, 0.0043830, -0.0067527, 0.0051113, -0.0113076, 0.0111357
1: 0.9974524, 1.0093187, 0.9974833, 1.0104233, -0.0111001, 0.0100396
2: -0.0053681, 0.0052793, -0.0060914, 0.0058554, -0.0112234, 0.0113706
3: -0.0000203, 0.0024004, -0.0001760, 0.0023876, -0.0018342, 0.0019581
4: -0.0063015, 0.0014071, -0.0070337, 0.0015957, -0.0078973, 0.0084408
5: -0.0020424, 0.0074458, -0.0022958, 0.0083619, -0.0104043, 0.0097415
6: -0.0080996, 0.0018833, -0.0092306, 0.0019728, -0.0100724, 0.0111139
7: -0.0055765, -0.0002588, -0.0055548, 0.0001447, -0.0057212, 0.0052960
8: -0.0130274, -0.0024086, -0.0135914, -0.0024755, -0.0105520, 0.0111828
9: -0.0037455, 0.0073055, -0.0047543, 0.0076364, -0.0113819, 0.0120598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_A1_A2_B2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0089824
time: 0.94 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0092401
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0062103, 0.0044013, -0.0058797, 0.0039685, -0.0101789, 0.0102811
1: 0.9974586, 1.0093466, 0.9973204, 1.0086900, -0.0094042, 0.0102204
2: -0.0053862, 0.0052937, -0.0049563, 0.0049513, -0.0103376, 0.0102501
3: -0.0000242, 0.0023978, 0.0000684, 0.0024551, -0.0018940, 0.0017166
4: -0.0063200, 0.0014118, -0.0058848, 0.0012996, -0.0076196, 0.0072966
5: -0.0020488, 0.0074688, -0.0018982, 0.0069243, -0.0089731, 0.0093670
6: -0.0081280, 0.0018856, -0.0074559, 0.0018324, -0.0099604, 0.0093414
7: -0.0055721, -0.0002486, -0.0056688, -0.0004885, -0.0050836, 0.0054202
8: -0.0130416, -0.0024220, -0.0127064, -0.0021231, -0.0109185, 0.0102844
9: -0.0037709, 0.0073139, -0.0031713, 0.0071172, -0.0108881, 0.0104851

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089064, upper bound: 0.0087327
time: 1.05 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087957, upper bound: 0.0087330
time: 0.89 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0062103, 0.0044013, -0.0062971, 0.0045148, -0.0107252, 0.0106984
1: 0.9974586, 1.0093466, 0.9973263, 1.0095187, -0.0102202, 0.0101159
2: -0.0053862, 0.0052937, -0.0054990, 0.0053835, -0.0107698, 0.0107927
3: -0.0000242, 0.0023978, -0.0000485, 0.0024527, -0.0018433, 0.0018324
4: -0.0063200, 0.0014118, -0.0064341, 0.0014412, -0.0077612, 0.0078459
5: -0.0020488, 0.0074688, -0.0020883, 0.0076116, -0.0096603, 0.0095570
6: -0.0081280, 0.0018856, -0.0083043, 0.0018995, -0.0100276, 0.0101899
7: -0.0055721, -0.0002486, -0.0056648, -0.0001858, -0.0053864, 0.0054161
8: -0.0130416, -0.0024220, -0.0131295, -0.0021356, -0.0109060, 0.0107075
9: -0.0037709, 0.0073139, -0.0039281, 0.0073654, -0.0111363, 0.0112419

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087981, upper bound: 0.0088609
time: 1.07 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_A2

### Relational analysis result of IS_A2_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087957, upper bound: 0.0087330
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0063875, 0.0046333, -0.0061573, 0.0043319, -0.0107194, 0.0107906
1: 0.9974533, 1.0096984, 0.9974726, 1.0092413, -0.0099177, 0.0105007
2: -0.0056166, 0.0054772, -0.0053173, 0.0052388, -0.0108554, 0.0107945
3: -0.0000738, 0.0024000, -0.0000094, 0.0023920, -0.0019275, 0.0017848
4: -0.0065531, 0.0014719, -0.0062502, 0.0013938, -0.0079469, 0.0077220
5: -0.0021295, 0.0077605, -0.0020246, 0.0073814, -0.0095109, 0.0097851
6: -0.0084882, 0.0019141, -0.0080202, 0.0018770, -0.0103653, 0.0099343
7: -0.0055758, -0.0001201, -0.0055623, -0.0002871, -0.0052887, 0.0054422
8: -0.0132212, -0.0024106, -0.0129879, -0.0024523, -0.0107689, 0.0105772
9: -0.0040921, 0.0074192, -0.0036747, 0.0072823, -0.0113744, 0.0110939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_A2_B2_B1_A1

### Relational analysis result of IS_A2_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0090097
time: 0.91 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_A2

### Relational analysis result of IS_A2_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0089593
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0063875, 0.0046333, -0.0065623, 0.0048620, -0.0112495, 0.0111955
1: 0.9974533, 1.0096984, 0.9974879, 1.0100453, -0.0107390, 0.0104020
2: -0.0056166, 0.0054772, -0.0058438, 0.0056582, -0.0112748, 0.0113210
3: -0.0000738, 0.0024000, -0.0001227, 0.0023857, -0.0018743, 0.0019077
4: -0.0065531, 0.0014719, -0.0067831, 0.0015312, -0.0080843, 0.0082550
5: -0.0021295, 0.0077605, -0.0022090, 0.0080483, -0.0101777, 0.0099696
6: -0.0084882, 0.0019141, -0.0088435, 0.0019422, -0.0104304, 0.0107575
7: -0.0055758, -0.0001201, -0.0055517, 0.0000066, -0.0055824, 0.0054315
8: -0.0132212, -0.0024106, -0.0133984, -0.0024852, -0.0107361, 0.0109877
9: -0.0040921, 0.0074192, -0.0044090, 0.0075231, -0.0116153, 0.0118282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0090097
time: 0.89 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0092556
time: 0.93 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0063027, 0.0045223, -0.0059255, 0.0040284, -0.0103311, 0.0104478
1: 0.9974922, 1.0095299, 0.9973201, 1.0087810, -0.0095499, 0.0104190
2: -0.0055063, 0.0053894, -0.0050158, 0.0049987, -0.0105050, 0.0104052
3: -0.0000501, 0.0023838, 0.0000555, 0.0024553, -0.0019174, 0.0017517
4: -0.0064415, 0.0014431, -0.0059450, 0.0013152, -0.0077567, 0.0073881
5: -0.0020908, 0.0076209, -0.0019190, 0.0069997, -0.0090905, 0.0095399
6: -0.0083158, 0.0019004, -0.0075489, 0.0018397, -0.0101555, 0.0094493
7: -0.0055485, -0.0001817, -0.0056691, -0.0004553, -0.0050932, 0.0054875
8: -0.0131353, -0.0024951, -0.0127528, -0.0021221, -0.0110131, 0.0102578
9: -0.0039383, 0.0073688, -0.0032543, 0.0071444, -0.0110828, 0.0106230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089305, upper bound: 0.0087867
time: 0.89 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088337, upper bound: 0.0087883
time: 0.95 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0063027, 0.0045223, -0.0063442, 0.0045766, -0.0108793, 0.0108665
1: 0.9974922, 1.0095299, 0.9973258, 1.0096123, -0.0103198, 0.0103069
2: -0.0055063, 0.0053894, -0.0055603, 0.0054324, -0.0109387, 0.0109497
3: -0.0000501, 0.0023838, -0.0000617, 0.0024529, -0.0018641, 0.0018478
4: -0.0064415, 0.0014431, -0.0064961, 0.0014572, -0.0078987, 0.0079393
5: -0.0020908, 0.0076209, -0.0021097, 0.0076893, -0.0097801, 0.0097306
6: -0.0083158, 0.0019004, -0.0084002, 0.0019071, -0.0102229, 0.0103006
7: -0.0055485, -0.0001817, -0.0056651, -0.0001515, -0.0053970, 0.0054835
8: -0.0131353, -0.0024951, -0.0131774, -0.0021346, -0.0110007, 0.0106823
9: -0.0039383, 0.0073688, -0.0040136, 0.0073935, -0.0113318, 0.0113824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088412, upper bound: 0.0089119
time: 0.99 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088337, upper bound: 0.0087883
time: 0.91 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.0064819, 0.0047568, -0.0061981, 0.0043853, -0.0108672, 0.0109549
1: 0.9974869, 1.0098858, 0.9974722, 1.0093222, -0.0100483, 0.0106826
2: -0.0057393, 0.0055750, -0.0053703, 0.0052810, -0.0110204, 0.0109453
3: -0.0001002, 0.0023861, -0.0000208, 0.0023923, -0.0019469, 0.0018213
4: -0.0066774, 0.0015039, -0.0063038, 0.0014076, -0.0080850, 0.0078077
5: -0.0021725, 0.0079160, -0.0020432, 0.0074486, -0.0096210, 0.0099592
6: -0.0086801, 0.0019293, -0.0081031, 0.0018836, -0.0105637, 0.0100323
7: -0.0055524, -0.0000517, -0.0055627, -0.0002575, -0.0052948, 0.0055111
8: -0.0133169, -0.0024831, -0.0130292, -0.0024511, -0.0108659, 0.0105461
9: -0.0042633, 0.0074754, -0.0037486, 0.0073066, -0.0115699, 0.0112240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0090396
time: 1.15 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0089939
time: 1.07 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.0064819, 0.0047568, -0.0066141, 0.0049299, -0.0114118, 0.0113709
1: 0.9974869, 1.0098858, 0.9974872, 1.0101483, -0.0108299, 0.0105723
2: -0.0057393, 0.0055750, -0.0059112, 0.0057119, -0.0114512, 0.0114862
3: -0.0001002, 0.0023861, -0.0001372, 0.0023860, -0.0018921, 0.0019205
4: -0.0066774, 0.0015039, -0.0068514, 0.0015488, -0.0082261, 0.0083553
5: -0.0021725, 0.0079160, -0.0022327, 0.0081337, -0.0103062, 0.0101486
6: -0.0086801, 0.0019293, -0.0089489, 0.0019505, -0.0106307, 0.0108781
7: -0.0055524, -0.0000517, -0.0055521, 0.0000442, -0.0055966, 0.0055005
8: -0.0133169, -0.0024831, -0.0134510, -0.0024839, -0.0108331, 0.0109679
9: -0.0042633, 0.0074754, -0.0045031, 0.0075540, -0.0118173, 0.0119784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0090396
time: 1.35 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0093117
time: 0.94 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0063101, 0.0045319, -0.0059329, 0.0040381, -0.0103481, 0.0104648
1: 0.9974931, 1.0095445, 0.9973193, 1.0087957, -0.0095771, 0.0103556
2: -0.0055159, 0.0053970, -0.0050254, 0.0050064, -0.0105222, 0.0104224
3: -0.0000521, 0.0023835, 0.0000535, 0.0024556, -0.0018933, 0.0017621
4: -0.0064512, 0.0014456, -0.0059547, 0.0013177, -0.0077689, 0.0074003
5: -0.0020942, 0.0076330, -0.0019224, 0.0070118, -0.0091060, 0.0095554
6: -0.0083308, 0.0019016, -0.0075639, 0.0018409, -0.0101717, 0.0094655
7: -0.0055480, -0.0001763, -0.0056696, -0.0004499, -0.0050981, 0.0054933
8: -0.0131427, -0.0024966, -0.0127603, -0.0021207, -0.0110220, 0.0102637
9: -0.0039517, 0.0073732, -0.0032676, 0.0071488, -0.0111005, 0.0106408

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A2_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089499, upper bound: 0.0088415
time: 1.18 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088619, upper bound: 0.0088425
time: 1.12 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0063101, 0.0045319, -0.0063439, 0.0045761, -0.0108862, 0.0108757
1: 0.9974931, 1.0095445, 0.9973252, 1.0096116, -0.0103830, 0.0102543
2: -0.0055159, 0.0053970, -0.0055598, 0.0054320, -0.0109479, 0.0109569
3: -0.0000521, 0.0023835, -0.0000616, 0.0024531, -0.0018403, 0.0018753
4: -0.0064512, 0.0014456, -0.0064957, 0.0014571, -0.0079083, 0.0079413
5: -0.0020942, 0.0076330, -0.0021096, 0.0076887, -0.0097828, 0.0097426
6: -0.0083308, 0.0019016, -0.0083995, 0.0019070, -0.0102378, 0.0103011
7: -0.0055480, -0.0001763, -0.0056655, -0.0001518, -0.0053962, 0.0054892
8: -0.0131427, -0.0024966, -0.0131770, -0.0021334, -0.0110093, 0.0106803
9: -0.0039517, 0.0073732, -0.0040130, 0.0073933, -0.0113449, 0.0113861

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088636, upper bound: 0.0089474
time: 1.04 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088619, upper bound: 0.0088425
time: 0.95 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0065001, 0.0047806, -0.0062102, 0.0044011, -0.0109012, 0.0109908
1: 0.9974878, 1.0099219, 0.9974716, 1.0093462, -0.0101096, 0.0106625
2: -0.0057630, 0.0055938, -0.0053860, 0.0052935, -0.0110565, 0.0109798
3: -0.0001053, 0.0023857, -0.0000242, 0.0023925, -0.0019320, 0.0018355
4: -0.0067013, 0.0015101, -0.0063197, 0.0014117, -0.0081130, 0.0078298
5: -0.0021807, 0.0079459, -0.0020487, 0.0074685, -0.0096492, 0.0099946
6: -0.0087171, 0.0019322, -0.0081276, 0.0018855, -0.0106026, 0.0100598
7: -0.0055517, -0.0000385, -0.0055631, -0.0002488, -0.0053029, 0.0055246
8: -0.0133354, -0.0024851, -0.0130414, -0.0024499, -0.0108854, 0.0105563
9: -0.0042963, 0.0074862, -0.0037705, 0.0073137, -0.0116100, 0.0112567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0090644
time: 1.19 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0093428
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0065001, 0.0047806, -0.0066213, 0.0049394, -0.0114395, 0.0114020
1: 0.9974878, 1.0099219, 0.9974867, 1.0101626, -0.0109211, 0.0105651
2: -0.0057630, 0.0055938, -0.0059206, 0.0057194, -0.0114824, 0.0115144
3: -0.0001053, 0.0023857, -0.0001393, 0.0023862, -0.0018772, 0.0019551
4: -0.0067013, 0.0015101, -0.0068609, 0.0015512, -0.0082525, 0.0083710
5: -0.0021807, 0.0079459, -0.0022359, 0.0081456, -0.0103263, 0.0101819
6: -0.0087171, 0.0019322, -0.0089636, 0.0019517, -0.0106688, 0.0108958
7: -0.0055517, -0.0000385, -0.0055525, 0.0000495, -0.0056012, 0.0055140
8: -0.0133354, -0.0024851, -0.0134583, -0.0024829, -0.0108525, 0.0109731
9: -0.0042963, 0.0074862, -0.0045162, 0.0075583, -0.0118546, 0.0120023

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 194

## Relational analysis of IS_A2_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0090644
time: 1.16 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0093428
time: 0.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.71 seconds
IS_A1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090419, upper bound: 0.0088572
IS_A1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089237, upper bound: 0.0088602
IS_A1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0092774
IS_A1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0094809
IS_A1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087965, upper bound: 0.0089164
IS_A1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087892, upper bound: 0.0087632
IS_A1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090169, upper bound: 0.0091018
IS_A1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090169, upper bound: 0.0093352
IS_A1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089296, upper bound: 0.0090086
IS_A1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089257, upper bound: 0.0088578
IS_A1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0092341
IS_A1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0092556, upper bound: 0.0094617
IS_A1_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088011, upper bound: 0.0089196
IS_A1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087956, upper bound: 0.0087608
IS_A1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090290, upper bound: 0.0090892
IS_A1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090290, upper bound: 0.0093259
IS_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089771, upper bound: 0.0090837
IS_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089772, upper bound: 0.0089530
IS_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0093034, upper bound: 0.0092976
IS_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0093034, upper bound: 0.0092976
IS_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088328, upper bound: 0.0089878
IS_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088284, upper bound: 0.0088489
IS_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090427, upper bound: 0.0091370
IS_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090427, upper bound: 0.0093854
IS_A1_A2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090835, upper bound: 0.0089577
IS_A1_A2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089775, upper bound: 0.0089581
IS_A1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0093016, upper bound: 0.0092976
IS_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0093016, upper bound: 0.0095444
IS_A1_A2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088620, upper bound: 0.0089979
IS_A1_A2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088577, upper bound: 0.0088628
IS_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090737, upper bound: 0.0091436
IS_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090737, upper bound: 0.0091436
IS_A2_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088814, upper bound: 0.0087691
IS_A2_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087680, upper bound: 0.0087660
IS_A2_A1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087695, upper bound: 0.0088467
IS_A2_A1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087680, upper bound: 0.0087660
IS_A2_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0089824
IS_A2_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0092401
IS_A2_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0089824
IS_A2_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090181, upper bound: 0.0092401
IS_A2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089064, upper bound: 0.0087327
IS_A2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087957, upper bound: 0.0087330
IS_A2_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087981, upper bound: 0.0088609
IS_A2_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0087957, upper bound: 0.0087330
IS_A2_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0090097
IS_A2_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0089593
IS_A2_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0090097
IS_A2_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089786, upper bound: 0.0092556
IS_A2_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089305, upper bound: 0.0087867
IS_A2_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088337, upper bound: 0.0087883
IS_A2_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088412, upper bound: 0.0089119
IS_A2_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088337, upper bound: 0.0087883
IS_A2_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0090396
IS_A2_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0089939
IS_A2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0090396
IS_A2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090074, upper bound: 0.0093117
IS_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0089499, upper bound: 0.0088415
IS_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088619, upper bound: 0.0088425
IS_A2_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088636, upper bound: 0.0089474
IS_A2_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0088619, upper bound: 0.0088425
IS_A2_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0090644
IS_A2_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0093428
IS_A2_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0090644
IS_A2_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.71
Output dim: 1, lower bound: -0.0090191, upper bound: 0.0093428

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0057954, 0.0038581, -0.0058593, 0.0039417, -0.0097372, 0.0097174
1: 0.9974493, 1.0085227, 0.9973778, 1.0086496, -0.0093900, 0.0094123
2: -0.0048467, 0.0048640, -0.0049298, 0.0049302, -0.0097769, 0.0097938
3: 0.0000920, 0.0024017, 0.0000741, 0.0024313, -0.0017640, 0.0017128
4: -0.0057738, 0.0012710, -0.0058579, 0.0012927, -0.0070665, 0.0071289
5: -0.0018598, 0.0067854, -0.0018889, 0.0068906, -0.0087504, 0.0086743
6: -0.0072844, 0.0018188, -0.0074143, 0.0018291, -0.0091135, 0.0092331
7: -0.0055786, -0.0005496, -0.0056286, -0.0005033, -0.0050753, 0.0050790
8: -0.0126209, -0.0024019, -0.0126857, -0.0022473, -0.0103736, 0.0102838
9: -0.0030184, 0.0070671, -0.0031342, 0.0071051, -0.0101234, 0.0102013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0084307, upper bound: 0.0083740
time: 0.73 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083949, upper bound: 0.0082160
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0057870, 0.0038471, -0.0060719, 0.0042201, -0.0100070, 0.0099190
1: 0.9974669, 1.0085058, 0.9973916, 1.0090716, -0.0097551, 0.0094301
2: -0.0048357, 0.0048553, -0.0052062, 0.0051503, -0.0099861, 0.0100614
3: 0.0000943, 0.0023944, 0.0000146, 0.0024256, -0.0017807, 0.0017410
4: -0.0057627, 0.0012682, -0.0061377, 0.0013648, -0.0071275, 0.0074059
5: -0.0018559, 0.0067715, -0.0019857, 0.0072407, -0.0090967, 0.0087572
6: -0.0072672, 0.0018174, -0.0078465, 0.0018633, -0.0091305, 0.0096640
7: -0.0055663, -0.0005558, -0.0056190, -0.0003491, -0.0052173, 0.0050632
8: -0.0126124, -0.0024398, -0.0129012, -0.0022772, -0.0103352, 0.0104614
9: -0.0030030, 0.0070620, -0.0035197, 0.0072315, -0.0102345, 0.0105818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083240, upper bound: 0.0083781
time: 0.96 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082855, upper bound: 0.0082167
time: 0.93 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0055296, 0.0035101, -0.0061491, 0.0043212, -0.0098507, 0.0096592
1: 0.9972677, 1.0079948, 0.9974732, 1.0092248, -0.0100420, 0.0088293
2: -0.0045010, 0.0045887, -0.0053066, 0.0052303, -0.0097313, 0.0098953
3: 0.0001664, 0.0024770, -0.0000071, 0.0023918, -0.0016658, 0.0018462
4: -0.0054239, 0.0012567, -0.0062393, 0.0013910, -0.0068149, 0.0074961
5: -0.0017387, 0.0063476, -0.0020209, 0.0073679, -0.0091066, 0.0083685
6: -0.0067439, 0.0017760, -0.0080035, 0.0018757, -0.0086196, 0.0097795
7: -0.0057058, -0.0007425, -0.0055619, -0.0002931, -0.0054127, 0.0048195
8: -0.0123514, -0.0020088, -0.0129795, -0.0024535, -0.0098980, 0.0109707
9: -0.0025363, 0.0069089, -0.0036598, 0.0072774, -0.0098137, 0.0105687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088848, upper bound: 0.0090387
time: 0.98 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088814, upper bound: 0.0089104
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0057825, 0.0038412, -0.0061491, 0.0043212, -0.0101037, 0.0099903
1: 0.9974372, 1.0084968, 0.9974732, 1.0092248, -0.0099051, 0.0092957
2: -0.0048299, 0.0048506, -0.0053066, 0.0052303, -0.0100602, 0.0101572
3: 0.0000956, 0.0024067, -0.0000071, 0.0023918, -0.0017330, 0.0017812
4: -0.0057568, 0.0012667, -0.0062393, 0.0013910, -0.0071478, 0.0075060
5: -0.0018539, 0.0067642, -0.0020209, 0.0073679, -0.0092218, 0.0087850
6: -0.0072582, 0.0018167, -0.0080035, 0.0018757, -0.0091339, 0.0098202
7: -0.0055871, -0.0005590, -0.0055619, -0.0002931, -0.0052941, 0.0050029
8: -0.0126079, -0.0023756, -0.0129795, -0.0024535, -0.0101544, 0.0106040
9: -0.0029949, 0.0070594, -0.0036598, 0.0072774, -0.0102724, 0.0107192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088848, upper bound: 0.0091255
time: 1.08 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_A2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088814, upper bound: 0.0089793
time: 1.20 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0057810, 0.0038393, -0.0062957, 0.0045131, -0.0102941, 0.0101350
1: 0.9974979, 1.0084941, 0.9973382, 1.0095158, -0.0102313, 0.0093353
2: -0.0048280, 0.0048491, -0.0054972, 0.0053821, -0.0102101, 0.0103463
3: 0.0000960, 0.0023815, -0.0000481, 0.0024477, -0.0017423, 0.0018369
4: -0.0057549, 0.0012662, -0.0064323, 0.0014407, -0.0071956, 0.0076985
5: -0.0018532, 0.0067617, -0.0020876, 0.0076093, -0.0094626, 0.0088494
6: -0.0072552, 0.0018165, -0.0083015, 0.0018993, -0.0091545, 0.0101180
7: -0.0055446, -0.0005601, -0.0056562, -0.0001867, -0.0053578, 0.0050962
8: -0.0126064, -0.0025071, -0.0131282, -0.0021620, -0.0104444, 0.0106211
9: -0.0029923, 0.0070585, -0.0039256, 0.0073646, -0.0103569, 0.0109841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083060, upper bound: 0.0082619
time: 0.80 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081803, upper bound: 0.0082329
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0059988, 0.0041244, -0.0062870, 0.0045017, -0.0105005, 0.0104115
1: 0.9975029, 1.0089265, 0.9973561, 1.0094987, -0.0102454, 0.0097391
2: -0.0051112, 0.0050747, -0.0054859, 0.0053732, -0.0104844, 0.0105606
3: 0.0000350, 0.0023795, -0.0000457, 0.0024404, -0.0017803, 0.0018516
4: -0.0060415, 0.0013400, -0.0064209, 0.0014378, -0.0074793, 0.0077609
5: -0.0019524, 0.0071204, -0.0020837, 0.0075951, -0.0095475, 0.0092041
6: -0.0076980, 0.0018515, -0.0082839, 0.0018979, -0.0095959, 0.0101355
7: -0.0055412, -0.0004021, -0.0056439, -0.0001930, -0.0053481, 0.0052418
8: -0.0128272, -0.0025177, -0.0131194, -0.0022001, -0.0106270, 0.0106017
9: -0.0033873, 0.0071880, -0.0039099, 0.0073595, -0.0107467, 0.0110980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082971, upper bound: 0.0081452
time: 1.32 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081727, upper bound: 0.0081158
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0055296, 0.0035101, -0.0065558, 0.0048536, -0.0103832, 0.0100659
1: 0.9972677, 1.0079948, 0.9974881, 1.0100325, -0.0108779, 0.0087571
2: -0.0045010, 0.0045887, -0.0058354, 0.0056515, -0.0101525, 0.0104241
3: 0.0001664, 0.0024770, -0.0001209, 0.0023855, -0.0016411, 0.0019915
4: -0.0054239, 0.0012567, -0.0067747, 0.0015290, -0.0069529, 0.0080314
5: -0.0017387, 0.0063476, -0.0022061, 0.0080377, -0.0097764, 0.0085537
6: -0.0067439, 0.0017760, -0.0088304, 0.0019412, -0.0086851, 0.0106064
7: -0.0057058, -0.0007425, -0.0055514, 0.0000020, -0.0057077, 0.0048089
8: -0.0123514, -0.0020088, -0.0133919, -0.0024862, -0.0098653, 0.0113831
9: -0.0025363, 0.0069089, -0.0043974, 0.0075193, -0.0100556, 0.0113063

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087415, upper bound: 0.0089354
time: 0.98 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087375, upper bound: 0.0088238
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0057825, 0.0038412, -0.0065558, 0.0048536, -0.0106361, 0.0103970
1: 0.9974372, 1.0084968, 0.9974881, 1.0100325, -0.0107526, 0.0092204
2: -0.0048299, 0.0048506, -0.0058354, 0.0056515, -0.0104814, 0.0106861
3: 0.0000956, 0.0024067, -0.0001209, 0.0023855, -0.0017030, 0.0019279
4: -0.0057568, 0.0012667, -0.0067747, 0.0015290, -0.0072858, 0.0080413
5: -0.0018539, 0.0067642, -0.0022061, 0.0080377, -0.0098916, 0.0089703
6: -0.0072582, 0.0018167, -0.0088304, 0.0019412, -0.0091993, 0.0106471
7: -0.0055871, -0.0005590, -0.0055514, 0.0000020, -0.0055891, 0.0049924
8: -0.0126079, -0.0023756, -0.0133919, -0.0024862, -0.0101217, 0.0110163
9: -0.0029949, 0.0070594, -0.0043974, 0.0075193, -0.0105143, 0.0114568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087415, upper bound: 0.0090764
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087375, upper bound: 0.0089324
time: 1.04 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0057905, 0.0038518, -0.0058765, 0.0039643, -0.0097549, 0.0097283
1: 0.9975044, 1.0085130, 0.9973317, 1.0086838, -0.0094331, 0.0093484
2: -0.0048404, 0.0048590, -0.0049522, 0.0049480, -0.0097884, 0.0098112
3: 0.0000933, 0.0023788, 0.0000692, 0.0024505, -0.0017411, 0.0017222
4: -0.0057674, 0.0012694, -0.0058806, 0.0012986, -0.0070660, 0.0071500
5: -0.0018576, 0.0067774, -0.0018967, 0.0069191, -0.0087766, 0.0086742
6: -0.0072745, 0.0018180, -0.0074494, 0.0018319, -0.0091064, 0.0092674
7: -0.0055401, -0.0005532, -0.0056610, -0.0004908, -0.0050493, 0.0051078
8: -0.0126160, -0.0025211, -0.0127032, -0.0021473, -0.0104687, 0.0101821
9: -0.0030095, 0.0070642, -0.0031655, 0.0071153, -0.0101248, 0.0102296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083271, upper bound: 0.0085013
time: 0.73 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082900, upper bound: 0.0083112
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0060101, 0.0041392, -0.0058682, 0.0039534, -0.0099635, 0.0100074
1: 0.9975104, 1.0089489, 0.9973515, 1.0086672, -0.0094489, 0.0097433
2: -0.0051259, 0.0050864, -0.0049413, 0.0049394, -0.0100653, 0.0100277
3: 0.0000319, 0.0023764, 0.0000716, 0.0024422, -0.0017733, 0.0017380
4: -0.0060564, 0.0013439, -0.0058696, 0.0012957, -0.0073521, 0.0072135
5: -0.0019576, 0.0071390, -0.0018929, 0.0069053, -0.0088629, 0.0090319
6: -0.0077209, 0.0018533, -0.0074324, 0.0018305, -0.0095514, 0.0092857
7: -0.0055359, -0.0003939, -0.0056469, -0.0004968, -0.0050391, 0.0052530
8: -0.0128386, -0.0025339, -0.0126947, -0.0021907, -0.0106479, 0.0101608
9: -0.0034077, 0.0071948, -0.0031503, 0.0071104, -0.0105181, 0.0103451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083233, upper bound: 0.0083537
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082847, upper bound: 0.0081731
time: 0.84 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0055270, 0.0035067, -0.0061573, 0.0043319, -0.0098589, 0.0096640
1: 0.9972810, 1.0079896, 0.9974726, 1.0092413, -0.0101432, 0.0087585
2: -0.0044977, 0.0045860, -0.0053173, 0.0052388, -0.0097365, 0.0099033
3: 0.0001671, 0.0024715, -0.0000094, 0.0023920, -0.0016416, 0.0018773
4: -0.0054205, 0.0012506, -0.0062502, 0.0013938, -0.0068143, 0.0075007
5: -0.0017375, 0.0063434, -0.0020246, 0.0073814, -0.0091190, 0.0083680
6: -0.0067387, 0.0017756, -0.0080202, 0.0018770, -0.0086157, 0.0097958
7: -0.0056964, -0.0007444, -0.0055623, -0.0002871, -0.0054093, 0.0048180
8: -0.0123488, -0.0020377, -0.0129879, -0.0024523, -0.0098965, 0.0109502
9: -0.0025315, 0.0069074, -0.0036747, 0.0072823, -0.0098139, 0.0105821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088854, upper bound: 0.0090190
time: 1.31 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088824, upper bound: 0.0089021
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058037, 0.0038690, -0.0061573, 0.0043319, -0.0101356, 0.0100263
1: 0.9974446, 1.0085392, 0.9974726, 1.0092413, -0.0099980, 0.0092593
2: -0.0048575, 0.0048726, -0.0053173, 0.0052388, -0.0100963, 0.0101899
3: 0.0000896, 0.0024036, -0.0000094, 0.0023920, -0.0017041, 0.0018112
4: -0.0057847, 0.0012739, -0.0062502, 0.0013938, -0.0071785, 0.0075240
5: -0.0018636, 0.0067991, -0.0020246, 0.0073814, -0.0092450, 0.0088237
6: -0.0073013, 0.0018201, -0.0080202, 0.0018770, -0.0091783, 0.0098404
7: -0.0055819, -0.0005436, -0.0055623, -0.0002871, -0.0052947, 0.0050187
8: -0.0126294, -0.0023919, -0.0129879, -0.0024523, -0.0101771, 0.0105959
9: -0.0030334, 0.0070720, -0.0036747, 0.0072823, -0.0103157, 0.0107467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088854, upper bound: 0.0089912
time: 0.97 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088824, upper bound: 0.0089801
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0057905, 0.0038518, -0.0062938, 0.0045106, -0.0103012, 0.0101456
1: 0.9975044, 1.0085130, 0.9973379, 1.0095123, -0.0102688, 0.0093206
2: -0.0048404, 0.0048590, -0.0054948, 0.0053802, -0.0102206, 0.0103537
3: 0.0000933, 0.0023788, -0.0000476, 0.0024479, -0.0017311, 0.0018551
4: -0.0057674, 0.0012694, -0.0064298, 0.0014401, -0.0072075, 0.0076992
5: -0.0018576, 0.0067774, -0.0020868, 0.0076063, -0.0094638, 0.0088642
6: -0.0072745, 0.0018180, -0.0082978, 0.0018990, -0.0091735, 0.0101158
7: -0.0055401, -0.0005532, -0.0056566, -0.0001881, -0.0053520, 0.0051035
8: -0.0126160, -0.0025211, -0.0131263, -0.0021608, -0.0104552, 0.0106051
9: -0.0030095, 0.0070642, -0.0039222, 0.0073635, -0.0103730, 0.0109864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087850, upper bound: 0.0089112
time: 1.04 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087850, upper bound: 0.0089189
time: 1.16 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0060101, 0.0041392, -0.0062851, 0.0044992, -0.0105094, 0.0104243
1: 0.9975104, 1.0089489, 0.9973556, 1.0094950, -0.0102839, 0.0097239
2: -0.0051259, 0.0050864, -0.0054835, 0.0053712, -0.0104971, 0.0105698
3: 0.0000319, 0.0023764, -0.0000451, 0.0024406, -0.0017675, 0.0018709
4: -0.0060564, 0.0013439, -0.0064184, 0.0014372, -0.0074936, 0.0077623
5: -0.0019576, 0.0071390, -0.0020828, 0.0075919, -0.0095495, 0.0092218
6: -0.0077209, 0.0018533, -0.0082801, 0.0018976, -0.0096185, 0.0101334
7: -0.0055359, -0.0003939, -0.0056443, -0.0001944, -0.0053416, 0.0052504
8: -0.0128386, -0.0025339, -0.0131175, -0.0021989, -0.0106397, 0.0105835
9: -0.0034077, 0.0071948, -0.0039065, 0.0073583, -0.0107661, 0.0111012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082983, upper bound: 0.0081123
time: 0.91 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081727, upper bound: 0.0080839
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0055270, 0.0035067, -0.0065623, 0.0048620, -0.0103890, 0.0100690
1: 0.9972810, 1.0079896, 0.9974879, 1.0100453, -0.0109681, 0.0087422
2: -0.0044977, 0.0045860, -0.0058438, 0.0056582, -0.0101558, 0.0104298
3: 0.0001671, 0.0024715, -0.0001227, 0.0023857, -0.0016364, 0.0020171
4: -0.0054205, 0.0012506, -0.0067831, 0.0015312, -0.0069516, 0.0080337
5: -0.0017375, 0.0063434, -0.0022090, 0.0080483, -0.0097858, 0.0085524
6: -0.0067387, 0.0017756, -0.0088435, 0.0019422, -0.0086809, 0.0106191
7: -0.0056964, -0.0007444, -0.0055517, 0.0000066, -0.0057031, 0.0048073
8: -0.0123488, -0.0020377, -0.0133984, -0.0024852, -0.0098636, 0.0113607
9: -0.0025315, 0.0069074, -0.0044090, 0.0075231, -0.0100547, 0.0113164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A2_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087565, upper bound: 0.0089304
time: 0.90 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087533, upper bound: 0.0088147
time: 1.01 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0058037, 0.0038690, -0.0065623, 0.0048620, -0.0106657, 0.0104312
1: 0.9974446, 1.0085392, 0.9974879, 1.0100453, -0.0108296, 0.0092348
2: -0.0048575, 0.0048726, -0.0058438, 0.0056582, -0.0105157, 0.0107164
3: 0.0000896, 0.0024036, -0.0001227, 0.0023857, -0.0016975, 0.0019539
4: -0.0057847, 0.0012739, -0.0067831, 0.0015312, -0.0073159, 0.0080570
5: -0.0018636, 0.0067991, -0.0022090, 0.0080483, -0.0099118, 0.0090081
6: -0.0073013, 0.0018201, -0.0088435, 0.0019422, -0.0092435, 0.0106636
7: -0.0055819, -0.0005436, -0.0055517, 0.0000066, -0.0055885, 0.0050081
8: -0.0126294, -0.0023919, -0.0133984, -0.0024852, -0.0101442, 0.0110065
9: -0.0030334, 0.0070720, -0.0044090, 0.0075231, -0.0105566, 0.0114810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087565, upper bound: 0.0090743
time: 0.90 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087533, upper bound: 0.0087459
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0060297, 0.0041648, -0.0057515, 0.0038006, -0.0098303, 0.0099163
1: 0.9975332, 1.0089878, 0.9973227, 1.0084354, -0.0092434, 0.0097646
2: -0.0051513, 0.0051066, -0.0047896, 0.0048185, -0.0099698, 0.0098962
3: 0.0000264, 0.0023669, 0.0001043, 0.0024542, -0.0017751, 0.0017176
4: -0.0060822, 0.0013505, -0.0057160, 0.0012561, -0.0073383, 0.0070665
5: -0.0019665, 0.0071712, -0.0018398, 0.0067131, -0.0086796, 0.0090110
6: -0.0077607, 0.0018565, -0.0071951, 0.0018117, -0.0095724, 0.0090516
7: -0.0055199, -0.0003797, -0.0056673, -0.0005815, -0.0049385, 0.0052876
8: -0.0128585, -0.0025833, -0.0125764, -0.0021278, -0.0107307, 0.0099931
9: -0.0034432, 0.0072064, -0.0029387, 0.0070409, -0.0104841, 0.0101451

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089124, upper bound: 0.0090305
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089124, upper bound: 0.0090837
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0062363, 0.0044353, -0.0057432, 0.0037897, -0.0100260, 0.0101785
1: 0.9975445, 1.0093979, 0.9973418, 1.0084188, -0.0092593, 0.0101276
2: -0.0054200, 0.0053206, -0.0047788, 0.0048099, -0.0102299, 0.0100994
3: -0.0000315, 0.0023622, 0.0001066, 0.0024463, -0.0018080, 0.0017335
4: -0.0063541, 0.0014206, -0.0057050, 0.0012533, -0.0076074, 0.0071256
5: -0.0020606, 0.0075115, -0.0018360, 0.0066994, -0.0087600, 0.0093475
6: -0.0081808, 0.0018897, -0.0071782, 0.0018104, -0.0099912, 0.0090679
7: -0.0055120, -0.0002298, -0.0056539, -0.0005875, -0.0049245, 0.0054241
8: -0.0130679, -0.0026079, -0.0125680, -0.0021691, -0.0108988, 0.0099601
9: -0.0038179, 0.0073293, -0.0029236, 0.0070360, -0.0108539, 0.0102529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0088923
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0089530
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0057648, 0.0038180, -0.0060239, 0.0041572, -0.0099220, 0.0098419
1: 0.9973230, 1.0084618, 0.9974802, 1.0089765, -0.0099496, 0.0091819
2: -0.0048069, 0.0048323, -0.0051438, 0.0051006, -0.0099075, 0.0099761
3: 0.0001005, 0.0024540, 0.0000280, 0.0023888, -0.0016767, 0.0018787
4: -0.0057335, 0.0012606, -0.0060745, 0.0013485, -0.0070820, 0.0073352
5: -0.0018458, 0.0067350, -0.0019638, 0.0071617, -0.0090076, 0.0086988
6: -0.0072221, 0.0018139, -0.0077490, 0.0018556, -0.0090777, 0.0095628
7: -0.0056670, -0.0005719, -0.0055570, -0.0003839, -0.0052831, 0.0049851
8: -0.0125899, -0.0021288, -0.0128526, -0.0024689, -0.0101210, 0.0107238
9: -0.0029628, 0.0070488, -0.0034327, 0.0072030, -0.0101657, 0.0104816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089239, upper bound: 0.0090788
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089254, upper bound: 0.0089704
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0060363, 0.0041735, -0.0060239, 0.0041572, -0.0101936, 0.0101974
1: 0.9974749, 1.0090009, 0.9974802, 1.0089765, -0.0098022, 0.0096634
2: -0.0051600, 0.0051135, -0.0051438, 0.0051006, -0.0102606, 0.0102573
3: 0.0000245, 0.0023910, 0.0000280, 0.0023888, -0.0017377, 0.0018143
4: -0.0060909, 0.0013528, -0.0060745, 0.0013485, -0.0074395, 0.0074273
5: -0.0019695, 0.0071822, -0.0019638, 0.0071617, -0.0091312, 0.0091460
6: -0.0077742, 0.0018576, -0.0077490, 0.0018556, -0.0096298, 0.0096065
7: -0.0055607, -0.0003749, -0.0055570, -0.0003839, -0.0051768, 0.0051821
8: -0.0128652, -0.0024574, -0.0128526, -0.0024689, -0.0103963, 0.0103952
9: -0.0034553, 0.0072104, -0.0034327, 0.0072030, -0.0106582, 0.0106431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089239, upper bound: 0.0090541
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089254, upper bound: 0.0090893
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0060297, 0.0041648, -0.0061786, 0.0043597, -0.0103894, 0.0103434
1: 0.9975332, 1.0089878, 0.9973202, 1.0092834, -0.0101076, 0.0097406
2: -0.0051513, 0.0051066, -0.0053449, 0.0052608, -0.0104122, 0.0104515
3: 0.0000264, 0.0023669, -0.0000153, 0.0024552, -0.0017669, 0.0018575
4: -0.0060822, 0.0013505, -0.0062781, 0.0014010, -0.0074832, 0.0076287
5: -0.0019665, 0.0071712, -0.0020343, 0.0074164, -0.0093829, 0.0092055
6: -0.0077607, 0.0018565, -0.0080634, 0.0018804, -0.0096412, 0.0099199
7: -0.0055199, -0.0003797, -0.0056690, -0.0002717, -0.0052483, 0.0052893
8: -0.0128585, -0.0025833, -0.0130094, -0.0021226, -0.0107359, 0.0104261
9: -0.0034432, 0.0072064, -0.0037132, 0.0072950, -0.0107382, 0.0109196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087710, upper bound: 0.0089444
time: 0.94 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087710, upper bound: 0.0089878
time: 0.97 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0062363, 0.0044353, -0.0061698, 0.0043483, -0.0105846, 0.0106051
1: 0.9975445, 1.0093979, 0.9973366, 1.0092659, -0.0101226, 0.0101048
2: -0.0054200, 0.0053206, -0.0053335, 0.0052517, -0.0106717, 0.0106541
3: -0.0000315, 0.0023622, -0.0000128, 0.0024484, -0.0018001, 0.0018733
4: -0.0063541, 0.0014206, -0.0062666, 0.0013980, -0.0077522, 0.0076872
5: -0.0020606, 0.0075115, -0.0020303, 0.0074020, -0.0094626, 0.0095418
6: -0.0081808, 0.0018897, -0.0080456, 0.0018790, -0.0100598, 0.0099353
7: -0.0055120, -0.0002298, -0.0056575, -0.0002781, -0.0052339, 0.0054276
8: -0.0130679, -0.0026079, -0.0130005, -0.0021582, -0.0109097, 0.0103926
9: -0.0038179, 0.0073293, -0.0036973, 0.0072897, -0.0111077, 0.0110266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087667, upper bound: 0.0087998
time: 1.02 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087667, upper bound: 0.0088489
time: 1.03 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0057648, 0.0038180, -0.0064444, 0.0047077, -0.0104725, 0.0102624
1: 0.9973230, 1.0084618, 0.9974887, 1.0098112, -0.0108112, 0.0091465
2: -0.0048069, 0.0048323, -0.0056905, 0.0055361, -0.0103430, 0.0105228
3: 0.0001005, 0.0024540, -0.0000897, 0.0023853, -0.0016656, 0.0020320
4: -0.0057335, 0.0012606, -0.0066280, 0.0014912, -0.0072247, 0.0078886
5: -0.0018458, 0.0067350, -0.0021554, 0.0078542, -0.0097000, 0.0088903
6: -0.0072221, 0.0018139, -0.0086038, 0.0019232, -0.0091453, 0.0104177
7: -0.0056670, -0.0005719, -0.0055510, -0.0000789, -0.0055881, 0.0049791
8: -0.0125899, -0.0021288, -0.0132789, -0.0024873, -0.0101026, 0.0111501
9: -0.0029628, 0.0070488, -0.0041953, 0.0074530, -0.0104158, 0.0112441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087697, upper bound: 0.0089798
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087684, upper bound: 0.0088677
time: 1.01 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0060363, 0.0041735, -0.0064444, 0.0047077, -0.0107441, 0.0106179
1: 0.9974749, 1.0090009, 0.9974887, 1.0098112, -0.0106746, 0.0096374
2: -0.0051600, 0.0051135, -0.0056905, 0.0055361, -0.0106961, 0.0108040
3: 0.0000245, 0.0023910, -0.0000897, 0.0023853, -0.0017302, 0.0019653
4: -0.0060909, 0.0013528, -0.0066280, 0.0014912, -0.0075821, 0.0079808
5: -0.0019695, 0.0071822, -0.0021554, 0.0078542, -0.0098237, 0.0093376
6: -0.0077742, 0.0018576, -0.0086038, 0.0019232, -0.0096975, 0.0104614
7: -0.0055607, -0.0003749, -0.0055510, -0.0000789, -0.0054818, 0.0051761
8: -0.0128652, -0.0024574, -0.0132789, -0.0024873, -0.0103779, 0.0108215
9: -0.0034553, 0.0072104, -0.0041953, 0.0074530, -0.0109083, 0.0114056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087697, upper bound: 0.0091408
time: 0.92 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087684, upper bound: 0.0090135
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0060542, 0.0041970, -0.0057564, 0.0038071, -0.0098614, 0.0099534
1: 0.9974862, 1.0090365, 0.9973788, 1.0084453, -0.0092189, 0.0098592
2: -0.0051833, 0.0051321, -0.0047960, 0.0048237, -0.0100069, 0.0099281
3: 0.0000195, 0.0023863, 0.0001029, 0.0024309, -0.0017916, 0.0017071
4: -0.0061145, 0.0013588, -0.0057225, 0.0012578, -0.0073723, 0.0070814
5: -0.0019777, 0.0072117, -0.0018420, 0.0067213, -0.0086990, 0.0090538
6: -0.0078107, 0.0018604, -0.0072052, 0.0018125, -0.0096232, 0.0090656
7: -0.0055527, -0.0003619, -0.0056280, -0.0005779, -0.0049748, 0.0052661
8: -0.0128834, -0.0024819, -0.0125814, -0.0022493, -0.0106341, 0.0100995
9: -0.0034878, 0.0072210, -0.0029477, 0.0070439, -0.0105316, 0.0101687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090190, upper bound: 0.0088979
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090190, upper bound: 0.0089577
time: 0.97 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0060458, 0.0041859, -0.0059538, 0.0040655, -0.0101113, 0.0101397
1: 0.9975051, 1.0090196, 0.9973925, 1.0088371, -0.0095584, 0.0098739
2: -0.0051722, 0.0051233, -0.0050527, 0.0050281, -0.0102003, 0.0101760
3: 0.0000219, 0.0023785, 0.0000476, 0.0024252, -0.0018061, 0.0017320
4: -0.0061033, 0.0013560, -0.0059823, 0.0013248, -0.0074281, 0.0073383
5: -0.0019738, 0.0071978, -0.0019319, 0.0070463, -0.0090201, 0.0091297
6: -0.0077934, 0.0018591, -0.0076065, 0.0018443, -0.0096377, 0.0094656
7: -0.0055396, -0.0003680, -0.0056183, -0.0004347, -0.0051049, 0.0052503
8: -0.0128748, -0.0025227, -0.0127816, -0.0022791, -0.0105957, 0.0102589
9: -0.0034724, 0.0072160, -0.0033056, 0.0071613, -0.0106337, 0.0105216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089021, upper bound: 0.0088935
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089021, upper bound: 0.0089581
time: 0.84 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0057718, 0.0038272, -0.0060593, 0.0042036, -0.0099754, 0.0098866
1: 0.9973224, 1.0084755, 0.9974740, 1.0090467, -0.0099197, 0.0092386
2: -0.0048160, 0.0048396, -0.0051898, 0.0051373, -0.0099533, 0.0100294
3: 0.0000986, 0.0024543, 0.0000181, 0.0023914, -0.0016974, 0.0018478
4: -0.0057427, 0.0012630, -0.0061212, 0.0013606, -0.0071033, 0.0073842
5: -0.0018490, 0.0067466, -0.0019800, 0.0072201, -0.0090691, 0.0087266
6: -0.0072364, 0.0018150, -0.0078210, 0.0018613, -0.0090977, 0.0096360
7: -0.0056674, -0.0005668, -0.0055613, -0.0003582, -0.0053092, 0.0049946
8: -0.0125970, -0.0021274, -0.0128885, -0.0024554, -0.0101417, 0.0107611
9: -0.0029756, 0.0070530, -0.0034970, 0.0072240, -0.0101996, 0.0105500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089363, upper bound: 0.0090835
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089357, upper bound: 0.0089775
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0060533, 0.0041957, -0.0060593, 0.0042036, -0.0102569, 0.0102550
1: 0.9974744, 1.0090346, 0.9974740, 1.0090467, -0.0097854, 0.0097619
2: -0.0051820, 0.0051310, -0.0051898, 0.0051373, -0.0103193, 0.0103209
3: 0.0000198, 0.0023912, 0.0000181, 0.0023914, -0.0017648, 0.0017864
4: -0.0061132, 0.0013585, -0.0061212, 0.0013606, -0.0074737, 0.0074797
5: -0.0019772, 0.0072101, -0.0019800, 0.0072201, -0.0091973, 0.0091900
6: -0.0078086, 0.0018603, -0.0078210, 0.0018613, -0.0096699, 0.0096813
7: -0.0055610, -0.0003626, -0.0055613, -0.0003582, -0.0052028, 0.0051987
8: -0.0128823, -0.0024563, -0.0128885, -0.0024554, -0.0104270, 0.0104322
9: -0.0034859, 0.0072204, -0.0034970, 0.0072240, -0.0107100, 0.0107174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0090561, upper bound: 0.0089363
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089357, upper bound: 0.0091115
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0060399, 0.0041782, -0.0061723, 0.0043516, -0.0103914, 0.0103505
1: 0.9975326, 1.0090080, 0.9973397, 1.0092710, -0.0100298, 0.0097816
2: -0.0051646, 0.0051172, -0.0053368, 0.0052544, -0.0104189, 0.0104540
3: 0.0000235, 0.0023671, -0.0000136, 0.0024471, -0.0017737, 0.0018320
4: -0.0060956, 0.0013540, -0.0062699, 0.0013989, -0.0074945, 0.0076239
5: -0.0019711, 0.0071880, -0.0020315, 0.0074062, -0.0093773, 0.0092195
6: -0.0077814, 0.0018581, -0.0080508, 0.0018794, -0.0096609, 0.0099089
7: -0.0055203, -0.0003723, -0.0056553, -0.0002762, -0.0052441, 0.0052830
8: -0.0128688, -0.0025821, -0.0130031, -0.0021649, -0.0107039, 0.0104210
9: -0.0034617, 0.0072125, -0.0037019, 0.0072912, -0.0107529, 0.0109144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087807, upper bound: 0.0089536
time: 0.96 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087807, upper bound: 0.0089979
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0062387, 0.0044385, -0.0061636, 0.0043402, -0.0105789, 0.0106021
1: 0.9975439, 1.0094029, 0.9973572, 1.0092539, -0.0100450, 0.0101413
2: -0.0054231, 0.0053231, -0.0053255, 0.0052454, -0.0106685, 0.0106486
3: -0.0000321, 0.0023624, -0.0000111, 0.0024398, -0.0018076, 0.0018474
4: -0.0063573, 0.0014214, -0.0062585, 0.0013959, -0.0077532, 0.0076799
5: -0.0020617, 0.0075155, -0.0020275, 0.0073919, -0.0094536, 0.0095430
6: -0.0081857, 0.0018901, -0.0080331, 0.0018780, -0.0100638, 0.0099232
7: -0.0055124, -0.0002281, -0.0056430, -0.0002825, -0.0052299, 0.0054150
8: -0.0130704, -0.0026067, -0.0129943, -0.0022029, -0.0108675, 0.0103876
9: -0.0038223, 0.0073307, -0.0036861, 0.0072861, -0.0111084, 0.0110169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087764, upper bound: 0.0088044
time: 1.17 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087764, upper bound: 0.0088628
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0057718, 0.0038272, -0.0064631, 0.0047322, -0.0105040, 0.0102903
1: 0.9973224, 1.0084755, 0.9974896, 1.0098482, -0.0107464, 0.0091576
2: -0.0048160, 0.0048396, -0.0057148, 0.0055555, -0.0103715, 0.0105544
3: 0.0000986, 0.0024543, -0.0000950, 0.0023849, -0.0016709, 0.0019985
4: -0.0057427, 0.0012630, -0.0066526, 0.0014975, -0.0072403, 0.0079156
5: -0.0018490, 0.0067466, -0.0021639, 0.0078849, -0.0097340, 0.0089104
6: -0.0072364, 0.0018150, -0.0086418, 0.0019262, -0.0091627, 0.0104568
7: -0.0056674, -0.0005668, -0.0055504, -0.0000653, -0.0056021, 0.0049836
8: -0.0125970, -0.0021274, -0.0132978, -0.0024892, -0.0101078, 0.0111704
9: -0.0029756, 0.0070530, -0.0042291, 0.0074642, -0.0104397, 0.0112822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088086, upper bound: 0.0089896
time: 1.13 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088078, upper bound: 0.0088835
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0060533, 0.0041957, -0.0064631, 0.0047322, -0.0107854, 0.0106587
1: 0.9974744, 1.0090346, 0.9974896, 1.0098482, -0.0106177, 0.0096777
2: -0.0051820, 0.0051310, -0.0057148, 0.0055555, -0.0107374, 0.0108459
3: 0.0000198, 0.0023912, -0.0000950, 0.0023849, -0.0017348, 0.0019349
4: -0.0061132, 0.0013585, -0.0066526, 0.0014975, -0.0076107, 0.0080111
5: -0.0019772, 0.0072101, -0.0021639, 0.0078849, -0.0098622, 0.0093739
6: -0.0078086, 0.0018603, -0.0086418, 0.0019262, -0.0097348, 0.0105021
7: -0.0055610, -0.0003626, -0.0055504, -0.0000653, -0.0054957, 0.0051878
8: -0.0128823, -0.0024563, -0.0132978, -0.0024892, -0.0103931, 0.0108415
9: -0.0034859, 0.0072204, -0.0042291, 0.0074642, -0.0109501, 0.0114496

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088086, upper bound: 0.0089623
time: 0.98 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088078, upper bound: 0.0090466
time: 0.92 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0059487, 0.0040589, -0.0061383, 0.0043070, -0.0102558, 0.0101972
1: 0.9972778, 1.0088271, 0.9975314, 1.0092036, -0.0099845, 0.0096311
2: -0.0050461, 0.0050228, -0.0052926, 0.0052191, -0.0102652, 0.0103154
3: 0.0000490, 0.0024728, -0.0000040, 0.0023677, -0.0017851, 0.0018129
4: -0.0059756, 0.0013231, -0.0062251, 0.0013874, -0.0073630, 0.0075482
5: -0.0019296, 0.0070380, -0.0020160, 0.0073501, -0.0092798, 0.0090539
6: -0.0075962, 0.0018435, -0.0079816, 0.0018740, -0.0094701, 0.0098250
7: -0.0056986, -0.0004384, -0.0055213, -0.0003009, -0.0053977, 0.0050829
8: -0.0127764, -0.0020310, -0.0129686, -0.0025792, -0.0101972, 0.0109376
9: -0.0032964, 0.0071583, -0.0036402, 0.0072710, -0.0105674, 0.0107985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A1_A1_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0087689
time: 0.84 seconds

## Relational analysis of IS_A2_A1_A1_A1_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0087710
time: 1.03 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0059408, 0.0040484, -0.0063564, 0.0045925, -0.0105333, 0.0104048
1: 0.9972953, 1.0088112, 0.9975427, 1.0096365, -0.0103708, 0.0096473
2: -0.0050357, 0.0050146, -0.0055761, 0.0054450, -0.0104807, 0.0105907
3: 0.0000513, 0.0024655, -0.0000651, 0.0023630, -0.0018012, 0.0018441
4: -0.0059651, 0.0013204, -0.0065121, 0.0014613, -0.0074265, 0.0078325
5: -0.0019260, 0.0070248, -0.0021153, 0.0077093, -0.0096353, 0.0091401
6: -0.0075800, 0.0018422, -0.0084249, 0.0019091, -0.0094890, 0.0102671
7: -0.0056864, -0.0004442, -0.0055133, -0.0001427, -0.0055437, 0.0050691
8: -0.0127683, -0.0020688, -0.0131897, -0.0026039, -0.0101644, 0.0111209
9: -0.0032820, 0.0071535, -0.0040357, 0.0074007, -0.0106827, 0.0111892

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A1_A1_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087595, upper bound: 0.0087637
time: 0.88 seconds

## Relational analysis of IS_A2_A1_A1_A1_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087595, upper bound: 0.0087637
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0059352, 0.0040412, -0.0065736, 0.0048769, -0.0108121, 0.0106148
1: 0.9973266, 1.0088003, 0.9975008, 1.0100678, -0.0107491, 0.0095331
2: -0.0050285, 0.0050088, -0.0058586, 0.0056699, -0.0106984, 0.0108674
3: 0.0000528, 0.0024526, -0.0001259, 0.0023803, -0.0017524, 0.0019026
4: -0.0059578, 0.0013185, -0.0067981, 0.0015350, -0.0074929, 0.0081165
5: -0.0019235, 0.0070157, -0.0022142, 0.0080670, -0.0099905, 0.0092299
6: -0.0075687, 0.0018413, -0.0088666, 0.0019440, -0.0095127, 0.0107079
7: -0.0056645, -0.0004482, -0.0055426, 0.0000149, -0.0056794, 0.0050944
8: -0.0127627, -0.0021364, -0.0134099, -0.0025131, -0.0102496, 0.0112734
9: -0.0032719, 0.0071502, -0.0044296, 0.0075299, -0.0108018, 0.0115798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A1_A1_B2_A1_B1

### Relational analysis result of IS_A2_A1_A1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087290, upper bound: 0.0088452
time: 0.76 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2_A1_B2

### Relational analysis result of IS_A2_A1_A1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087290, upper bound: 0.0088452
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0061539, 0.0043274, -0.0065648, 0.0048654, -0.0110193, 0.0108922
1: 0.9973322, 1.0092344, 0.9975172, 1.0100504, -0.0107637, 0.0099140
2: -0.0053128, 0.0052352, -0.0058471, 0.0056608, -0.0109736, 0.0110824
3: -0.0000084, 0.0024502, -0.0001234, 0.0023735, -0.0017848, 0.0019185
4: -0.0062456, 0.0013926, -0.0067865, 0.0015320, -0.0077776, 0.0081791
5: -0.0020230, 0.0073758, -0.0022102, 0.0080525, -0.0100756, 0.0095860
6: -0.0080132, 0.0018765, -0.0088487, 0.0019426, -0.0099558, 0.0107252
7: -0.0056606, -0.0002896, -0.0055311, 0.0000085, -0.0056691, 0.0052415
8: -0.0129844, -0.0021485, -0.0134010, -0.0025488, -0.0104355, 0.0112524
9: -0.0036684, 0.0072803, -0.0044137, 0.0075247, -0.0111931, 0.0116939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A1_A1_A1_B2_A2_B1

### Relational analysis result of IS_A2_A1_A1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087296, upper bound: 0.0087629
time: 0.75 seconds

## Relational analysis of IS_A2_A1_A1_A1_B2_A2_B2

### Relational analysis result of IS_A2_A1_A1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087296, upper bound: 0.0087629
time: 0.86 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.0061964, 0.0043830, -0.0058758, 0.0039634, -0.0101597, 0.0102588
1: 0.9974524, 1.0093187, 0.9973211, 1.0086824, -0.0093967, 0.0102861
2: -0.0053681, 0.0052793, -0.0049512, 0.0049472, -0.0103153, 0.0102305
3: -0.0000203, 0.0024004, 0.0000695, 0.0024548, -0.0019445, 0.0017110
4: -0.0063015, 0.0014071, -0.0058796, 0.0012983, -0.0075998, 0.0072866
5: -0.0020424, 0.0074458, -0.0018964, 0.0069178, -0.0089602, 0.0093421
6: -0.0080996, 0.0018833, -0.0074478, 0.0018317, -0.0099313, 0.0093311
7: -0.0055765, -0.0002588, -0.0056684, -0.0004913, -0.0050851, 0.0054096
8: -0.0130274, -0.0024086, -0.0127024, -0.0021245, -0.0109029, 0.0102938
9: -0.0037455, 0.0073055, -0.0031641, 0.0071149, -0.0108604, 0.0104697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_A2_B1_B1_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089412, upper bound: 0.0087226
time: 0.96 seconds

## Relational analysis of IS_A2_A1_A1_A2_B1_B1_B2

### Relational analysis result of IS_A2_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088255, upper bound: 0.0087236
time: 1.05 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.0061964, 0.0043830, -0.0061491, 0.0043212, -0.0105175, 0.0105321
1: 0.9974524, 1.0093187, 0.9974732, 1.0092248, -0.0098805, 0.0101477
2: -0.0053681, 0.0052793, -0.0053066, 0.0052303, -0.0105984, 0.0105859
3: -0.0000203, 0.0024004, -0.0000071, 0.0023918, -0.0018807, 0.0017751
4: -0.0063015, 0.0014071, -0.0062393, 0.0013910, -0.0076926, 0.0076464
5: -0.0020424, 0.0074458, -0.0020209, 0.0073679, -0.0094103, 0.0094666
6: -0.0080996, 0.0018833, -0.0080035, 0.0018757, -0.0099753, 0.0098868
7: -0.0055765, -0.0002588, -0.0055619, -0.0002931, -0.0052834, 0.0053032
8: -0.0130274, -0.0024086, -0.0129795, -0.0024535, -0.0105740, 0.0105709
9: -0.0037455, 0.0073055, -0.0036598, 0.0072774, -0.0110229, 0.0109653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_A2_B1_B2_B1

### Relational analysis result of IS_A2_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089412, upper bound: 0.0089157
time: 1.19 seconds

## Relational analysis of IS_A2_A1_A1_A2_B1_B2_B2

### Relational analysis result of IS_A2_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088255, upper bound: 0.0089168
time: 1.10 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.0061964, 0.0043830, -0.0062989, 0.0045173, -0.0107137, 0.0106820
1: 0.9974524, 1.0093187, 0.9973267, 1.0095223, -0.0101824, 0.0101737
2: -0.0053681, 0.0052793, -0.0055014, 0.0053855, -0.0107535, 0.0107807
3: -0.0000203, 0.0024004, -0.0000490, 0.0024525, -0.0018913, 0.0018146
4: -0.0063015, 0.0014071, -0.0064365, 0.0014419, -0.0077434, 0.0078436
5: -0.0020424, 0.0074458, -0.0020891, 0.0076147, -0.0096571, 0.0095349
6: -0.0080996, 0.0018833, -0.0083082, 0.0018998, -0.0099994, 0.0101915
7: -0.0055765, -0.0002588, -0.0056644, -0.0001844, -0.0053921, 0.0054056
8: -0.0130274, -0.0024086, -0.0131314, -0.0021368, -0.0108906, 0.0107228
9: -0.0037455, 0.0073055, -0.0039315, 0.0073665, -0.0111120, 0.0112371

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_A2_B2_B1_B1

### Relational analysis result of IS_A2_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088866, upper bound: 0.0087160
time: 0.96 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2_B1_B2

### Relational analysis result of IS_A2_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087882, upper bound: 0.0087178
time: 2.20 seconds

## BFS IS instance: IS_A2_A1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.0061964, 0.0043830, -0.0065558, 0.0048536, -0.0110500, 0.0109389
1: 0.9974524, 1.0093187, 0.9974881, 1.0100325, -0.0106521, 0.0100354
2: -0.0053681, 0.0052793, -0.0058354, 0.0056515, -0.0110196, 0.0111147
3: -0.0000203, 0.0024004, -0.0001209, 0.0023855, -0.0018302, 0.0018761
4: -0.0063015, 0.0014071, -0.0067747, 0.0015290, -0.0078305, 0.0081817
5: -0.0020424, 0.0074458, -0.0022061, 0.0080377, -0.0100801, 0.0096519
6: -0.0080996, 0.0018833, -0.0088304, 0.0019412, -0.0100407, 0.0107137
7: -0.0055765, -0.0002588, -0.0055514, 0.0000020, -0.0055784, 0.0052926
8: -0.0130274, -0.0024086, -0.0133919, -0.0024862, -0.0105413, 0.0109832
9: -0.0037455, 0.0073055, -0.0043974, 0.0075193, -0.0112648, 0.0117029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_A1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087929, upper bound: 0.0090223
time: 0.95 seconds

## Relational analysis of IS_A2_A1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_A1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087882, upper bound: 0.0089123
time: 1.13 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0062068, 0.0043967, -0.0058633, 0.0039470, -0.0101538, 0.0102600
1: 0.9974709, 1.0093396, 0.9973773, 1.0086575, -0.0093617, 0.0101695
2: -0.0053816, 0.0052901, -0.0049350, 0.0049343, -0.0103159, 0.0102251
3: -0.0000232, 0.0023927, 0.0000730, 0.0024316, -0.0018748, 0.0017077
4: -0.0063153, 0.0014106, -0.0058632, 0.0012941, -0.0076093, 0.0072738
5: -0.0020471, 0.0074629, -0.0018907, 0.0068973, -0.0089444, 0.0093536
6: -0.0081208, 0.0018850, -0.0074225, 0.0018297, -0.0099505, 0.0093075
7: -0.0055635, -0.0002512, -0.0056291, -0.0005004, -0.0050632, 0.0053779
8: -0.0130380, -0.0024485, -0.0126898, -0.0022460, -0.0107920, 0.0102413
9: -0.0037644, 0.0073117, -0.0031415, 0.0071075, -0.0108718, 0.0104532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A2_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083480, upper bound: 0.0082620
time: 0.91 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A2_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083047, upper bound: 0.0080799
time: 0.87 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0061982, 0.0043854, -0.0060729, 0.0042214, -0.0104196, 0.0104584
1: 0.9974874, 1.0093223, 0.9973909, 1.0090735, -0.0097289, 0.0101861
2: -0.0053705, 0.0052812, -0.0052075, 0.0051514, -0.0105218, 0.0104887
3: -0.0000208, 0.0023859, 0.0000143, 0.0024258, -0.0018914, 0.0017392
4: -0.0063040, 0.0014077, -0.0061390, 0.0013652, -0.0076692, 0.0075467
5: -0.0020432, 0.0074488, -0.0019862, 0.0072424, -0.0092856, 0.0094349
6: -0.0081034, 0.0018836, -0.0078486, 0.0018634, -0.0099668, 0.0097322
7: -0.0055519, -0.0002574, -0.0056194, -0.0003484, -0.0052036, 0.0053620
8: -0.0130293, -0.0024844, -0.0129023, -0.0022758, -0.0107535, 0.0104179
9: -0.0037488, 0.0073066, -0.0035216, 0.0072321, -0.0109809, 0.0108282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A2_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082523, upper bound: 0.0082577
time: 0.91 seconds

## Relational analysis of IS_A2_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A2_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082116, upper bound: 0.0080807
time: 1.00 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0061919, 0.0043772, -0.0062938, 0.0045106, -0.0107025, 0.0106710
1: 0.9975198, 1.0093100, 0.9973379, 1.0095123, -0.0101683, 0.0100724
2: -0.0053623, 0.0052746, -0.0054948, 0.0053802, -0.0107425, 0.0107694
3: -0.0000190, 0.0023724, -0.0000476, 0.0024479, -0.0018354, 0.0018123
4: -0.0062957, 0.0014055, -0.0064298, 0.0014401, -0.0077358, 0.0078354
5: -0.0020404, 0.0074385, -0.0020868, 0.0076063, -0.0096466, 0.0095252
6: -0.0080906, 0.0018826, -0.0082978, 0.0018990, -0.0099896, 0.0101804
7: -0.0055293, -0.0002620, -0.0056566, -0.0001881, -0.0053412, 0.0053946
8: -0.0130230, -0.0025545, -0.0131263, -0.0021608, -0.0108622, 0.0105718
9: -0.0037374, 0.0073029, -0.0039222, 0.0073635, -0.0111009, 0.0112251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_A1_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082143, upper bound: 0.0083535
time: 0.85 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_A1_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081753, upper bound: 0.0081607
time: 1.06 seconds

## BFS IS instance: IS_A2_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0064192, 0.0046747, -0.0062851, 0.0044992, -0.0109184, 0.0109599
1: 0.9975222, 1.0097611, 0.9973556, 1.0094950, -0.0101821, 0.0104768
2: -0.0056578, 0.0055100, -0.0054835, 0.0053712, -0.0110290, 0.0109935
3: -0.0000827, 0.0023714, -0.0000451, 0.0024406, -0.0018700, 0.0018275
4: -0.0065948, 0.0014826, -0.0064184, 0.0014372, -0.0080320, 0.0079010
5: -0.0021439, 0.0078127, -0.0020828, 0.0075919, -0.0097358, 0.0098956
6: -0.0085527, 0.0019192, -0.0082801, 0.0018976, -0.0104503, 0.0101993
7: -0.0055276, -0.0000972, -0.0056443, -0.0001944, -0.0053332, 0.0055472
8: -0.0132534, -0.0025597, -0.0131175, -0.0021989, -0.0110545, 0.0105578
9: -0.0041496, 0.0074381, -0.0039065, 0.0073583, -0.0115079, 0.0113446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A2_A1_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_A1_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082104, upper bound: 0.0082488
time: 1.03 seconds

## Relational analysis of IS_A2_A1_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_A1_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081726, upper bound: 0.0080665
time: 0.94 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0059378, 0.0040445, -0.0061573, 0.0043319, -0.0102697, 0.0102018
1: 0.9972900, 1.0088054, 0.9974726, 1.0092413, -0.0100619, 0.0095958
2: -0.0050318, 0.0050115, -0.0053173, 0.0052388, -0.0102706, 0.0103287
3: 0.0000521, 0.0024677, -0.0000094, 0.0023920, -0.0017796, 0.0018479
4: -0.0059612, 0.0013193, -0.0062502, 0.0013938, -0.0073550, 0.0075695
5: -0.0019246, 0.0070199, -0.0020246, 0.0073814, -0.0093061, 0.0090445
6: -0.0075739, 0.0018417, -0.0080202, 0.0018770, -0.0094509, 0.0098619
7: -0.0056901, -0.0004464, -0.0055623, -0.0002871, -0.0054030, 0.0051160
8: -0.0127653, -0.0020572, -0.0129879, -0.0024523, -0.0103130, 0.0109306
9: -0.0032765, 0.0071517, -0.0036747, 0.0072823, -0.0105589, 0.0108264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087889, upper bound: 0.0088643
time: 0.96 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087874, upper bound: 0.0087764
time: 0.93 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0061977, 0.0043848, -0.0061573, 0.0043319, -0.0105297, 0.0105421
1: 0.9974585, 1.0093215, 0.9974726, 1.0092413, -0.0099135, 0.0100781
2: -0.0053698, 0.0052807, -0.0053173, 0.0052388, -0.0106086, 0.0105980
3: -0.0000207, 0.0023979, -0.0000094, 0.0023920, -0.0018495, 0.0017811
4: -0.0063033, 0.0014075, -0.0062502, 0.0013938, -0.0076971, 0.0076577
5: -0.0020430, 0.0074480, -0.0020246, 0.0073814, -0.0094245, 0.0094726
6: -0.0081024, 0.0018835, -0.0080202, 0.0018770, -0.0099794, 0.0099038
7: -0.0055722, -0.0002578, -0.0055623, -0.0002871, -0.0052851, 0.0053045
8: -0.0130288, -0.0024216, -0.0129879, -0.0024523, -0.0105765, 0.0105663
9: -0.0037480, 0.0073063, -0.0036747, 0.0072823, -0.0110303, 0.0109810

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_A1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089189, upper bound: 0.0089256
time: 0.86 seconds

## Relational analysis of IS_A2_A1_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_A1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087874, upper bound: 0.0087200
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0059378, 0.0040445, -0.0065623, 0.0048620, -0.0107998, 0.0106068
1: 0.9972900, 1.0088054, 0.9974879, 1.0100453, -0.0108723, 0.0094900
2: -0.0050318, 0.0050115, -0.0058438, 0.0056582, -0.0106900, 0.0108552
3: 0.0000521, 0.0024677, -0.0001227, 0.0023857, -0.0017346, 0.0019700
4: -0.0059612, 0.0013193, -0.0067831, 0.0015312, -0.0074923, 0.0081024
5: -0.0019246, 0.0070199, -0.0022090, 0.0080483, -0.0099729, 0.0092289
6: -0.0075739, 0.0018417, -0.0088435, 0.0019422, -0.0095161, 0.0106852
7: -0.0056901, -0.0004464, -0.0055517, 0.0000066, -0.0056968, 0.0051053
8: -0.0127653, -0.0020572, -0.0133984, -0.0024852, -0.0102801, 0.0113412
9: -0.0032765, 0.0071517, -0.0044090, 0.0075231, -0.0107997, 0.0115608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087571, upper bound: 0.0088643
time: 0.92 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087566, upper bound: 0.0087750
time: 0.90 seconds

## BFS IS instance: IS_A2_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0061977, 0.0043848, -0.0065623, 0.0048620, -0.0110597, 0.0109471
1: 0.9974585, 1.0093215, 0.9974879, 1.0100453, -0.0107345, 0.0099792
2: -0.0053698, 0.0052807, -0.0058438, 0.0056582, -0.0110280, 0.0111245
3: -0.0000207, 0.0023979, -0.0001227, 0.0023857, -0.0017987, 0.0019039
4: -0.0063033, 0.0014075, -0.0067831, 0.0015312, -0.0078345, 0.0081906
5: -0.0020430, 0.0074480, -0.0022090, 0.0080483, -0.0100913, 0.0096570
6: -0.0081024, 0.0018835, -0.0088435, 0.0019422, -0.0100446, 0.0107270
7: -0.0055722, -0.0002578, -0.0055517, 0.0000066, -0.0055789, 0.0052939
8: -0.0130288, -0.0024216, -0.0133984, -0.0024852, -0.0105437, 0.0109768
9: -0.0037480, 0.0073063, -0.0044090, 0.0075231, -0.0112711, 0.0117154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087571, upper bound: 0.0090336
time: 0.94 seconds

## Relational analysis of IS_A2_A1_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087566, upper bound: 0.0087165
time: 0.94 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0062995, 0.0045180, -0.0059092, 0.0040071, -0.0103067, 0.0104273
1: 0.9975049, 1.0095235, 0.9973768, 1.0087485, -0.0095086, 0.0103707
2: -0.0055021, 0.0053861, -0.0049947, 0.0049819, -0.0104840, 0.0103808
3: -0.0000492, 0.0023786, 0.0000601, 0.0024318, -0.0018978, 0.0017435
4: -0.0064373, 0.0014420, -0.0059236, 0.0013096, -0.0077469, 0.0073657
5: -0.0020894, 0.0076156, -0.0019116, 0.0069729, -0.0090623, 0.0095272
6: -0.0083093, 0.0018999, -0.0075158, 0.0018371, -0.0101464, 0.0094158
7: -0.0055397, -0.0001840, -0.0056294, -0.0004671, -0.0050726, 0.0054454
8: -0.0131320, -0.0025223, -0.0127364, -0.0022450, -0.0108870, 0.0102141
9: -0.0039325, 0.0073669, -0.0032248, 0.0071348, -0.0110673, 0.0105917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A1_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089179, upper bound: 0.0087406
time: 1.14 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089179, upper bound: 0.0087925
time: 0.96 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0062910, 0.0045069, -0.0061163, 0.0042782, -0.0105692, 0.0106232
1: 0.9975201, 1.0095067, 0.9973906, 1.0091597, -0.0098634, 0.0103853
2: -0.0054911, 0.0053772, -0.0052640, 0.0051964, -0.0106874, 0.0106412
3: -0.0000468, 0.0023723, 0.0000021, 0.0024260, -0.0019124, 0.0017720
4: -0.0064261, 0.0014391, -0.0061962, 0.0013799, -0.0078060, 0.0076353
5: -0.0020855, 0.0076015, -0.0020059, 0.0073139, -0.0093994, 0.0096075
6: -0.0082919, 0.0018985, -0.0079368, 0.0018704, -0.0101624, 0.0098354
7: -0.0055291, -0.0001902, -0.0056197, -0.0003169, -0.0052122, 0.0054295
8: -0.0131234, -0.0025551, -0.0129463, -0.0022750, -0.0108484, 0.0103912
9: -0.0039171, 0.0073618, -0.0036003, 0.0072579, -0.0111750, 0.0109621

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A1_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088092, upper bound: 0.0087410
time: 0.98 seconds

## Relational analysis of IS_A2_A2_A1_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088092, upper bound: 0.0087937
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0062863, 0.0045007, -0.0063409, 0.0045723, -0.0108586, 0.0108416
1: 0.9975566, 1.0094973, 0.9973374, 1.0096059, -0.0102683, 0.0102665
2: -0.0054849, 0.0053724, -0.0055560, 0.0054290, -0.0109139, 0.0109284
3: -0.0000455, 0.0023571, -0.0000608, 0.0024481, -0.0018562, 0.0018283
4: -0.0064199, 0.0014375, -0.0064918, 0.0014561, -0.0078760, 0.0079293
5: -0.0020833, 0.0075938, -0.0021082, 0.0076838, -0.0097672, 0.0097020
6: -0.0082824, 0.0018978, -0.0083935, 0.0019066, -0.0101889, 0.0102913
7: -0.0055035, -0.0001936, -0.0056570, -0.0001539, -0.0053496, 0.0054634
8: -0.0131186, -0.0026342, -0.0131740, -0.0021597, -0.0109589, 0.0105398
9: -0.0039085, 0.0073590, -0.0040076, 0.0073915, -0.0113000, 0.0113667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087744, upper bound: 0.0088525
time: 0.89 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087744, upper bound: 0.0089119
time: 0.96 seconds

## BFS IS instance: IS_A2_A2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0065112, 0.0047952, -0.0063321, 0.0045607, -0.0110720, 0.0111273
1: 0.9975547, 1.0099437, 0.9973549, 1.0095885, -0.0102819, 0.0106588
2: -0.0057774, 0.0056053, -0.0055446, 0.0054198, -0.0111973, 0.0111499
3: -0.0001084, 0.0023579, -0.0000583, 0.0024408, -0.0018900, 0.0018435
4: -0.0067159, 0.0015139, -0.0064802, 0.0014531, -0.0081691, 0.0079941
5: -0.0021858, 0.0079643, -0.0021042, 0.0076693, -0.0098551, 0.0100685
6: -0.0087397, 0.0019340, -0.0083756, 0.0019052, -0.0106449, 0.0103096
7: -0.0055048, -0.0000304, -0.0056447, -0.0001603, -0.0053445, 0.0056143
8: -0.0133467, -0.0026300, -0.0131651, -0.0021977, -0.0111490, 0.0105351
9: -0.0043165, 0.0074928, -0.0039917, 0.0073863, -0.0117027, 0.0114845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087683, upper bound: 0.0087366
time: 0.97 seconds

## Relational analysis of IS_A2_A2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087683, upper bound: 0.0087883
time: 1.01 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0060187, 0.0041504, -0.0061981, 0.0043853, -0.0104039, 0.0103484
1: 0.9973114, 1.0089657, 0.9974722, 1.0093222, -0.0101805, 0.0097526
2: -0.0051370, 0.0050952, -0.0053703, 0.0052810, -0.0104180, 0.0104655
3: 0.0000295, 0.0024589, -0.0000208, 0.0023923, -0.0017947, 0.0018782
4: -0.0060676, 0.0013468, -0.0063038, 0.0014076, -0.0074753, 0.0076506
5: -0.0019615, 0.0071531, -0.0020432, 0.0074486, -0.0094100, 0.0091962
6: -0.0077383, 0.0018547, -0.0081031, 0.0018836, -0.0096219, 0.0099578
7: -0.0056752, -0.0003877, -0.0055627, -0.0002575, -0.0054177, 0.0051750
8: -0.0128473, -0.0021034, -0.0130292, -0.0024511, -0.0103962, 0.0109258
9: -0.0034232, 0.0071998, -0.0037486, 0.0073066, -0.0107297, 0.0109484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_A2_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088202, upper bound: 0.0089137
time: 0.95 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_A1_A2

### Relational analysis result of IS_A2_A2_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088195, upper bound: 0.0088284
time: 0.91 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0062868, 0.0045014, -0.0061981, 0.0043853, -0.0106721, 0.0106995
1: 0.9974921, 1.0094984, 0.9974722, 1.0093222, -0.0100440, 0.0102613
2: -0.0054856, 0.0053729, -0.0053703, 0.0052810, -0.0107667, 0.0107432
3: -0.0000456, 0.0023840, -0.0000208, 0.0023923, -0.0018710, 0.0018174
4: -0.0064206, 0.0014377, -0.0063038, 0.0014076, -0.0078282, 0.0077415
5: -0.0020836, 0.0075947, -0.0020432, 0.0074486, -0.0095321, 0.0096379
6: -0.0082835, 0.0018979, -0.0081031, 0.0018836, -0.0101670, 0.0100009
7: -0.0055487, -0.0001932, -0.0055627, -0.0002575, -0.0052912, 0.0053695
8: -0.0131191, -0.0024943, -0.0130292, -0.0024511, -0.0106681, 0.0105349
9: -0.0039095, 0.0073593, -0.0037486, 0.0073066, -0.0112160, 0.0111079

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_A2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089527, upper bound: 0.0089747
time: 0.88 seconds

## Relational analysis of IS_A2_A2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_A2_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088195, upper bound: 0.0087684
time: 0.95 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0060187, 0.0041504, -0.0066141, 0.0049299, -0.0109486, 0.0107645
1: 0.9973114, 1.0089657, 0.9974872, 1.0101483, -0.0109530, 0.0096458
2: -0.0051370, 0.0050952, -0.0059112, 0.0057119, -0.0108488, 0.0110064
3: 0.0000295, 0.0024589, -0.0001372, 0.0023860, -0.0017495, 0.0019830
4: -0.0060676, 0.0013468, -0.0068514, 0.0015488, -0.0076164, 0.0081981
5: -0.0019615, 0.0071531, -0.0022327, 0.0081337, -0.0100952, 0.0093857
6: -0.0077383, 0.0018547, -0.0089489, 0.0019505, -0.0096888, 0.0108036
7: -0.0056752, -0.0003877, -0.0055521, 0.0000442, -0.0057194, 0.0051644
8: -0.0128473, -0.0021034, -0.0134510, -0.0024839, -0.0103634, 0.0113476
9: -0.0034232, 0.0071998, -0.0045031, 0.0075540, -0.0109772, 0.0117029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087897, upper bound: 0.0089135
time: 1.12 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087864, upper bound: 0.0088270
time: 1.01 seconds

## BFS IS instance: IS_A2_A2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0062868, 0.0045014, -0.0066141, 0.0049299, -0.0112167, 0.0111155
1: 0.9974921, 1.0094984, 0.9974872, 1.0101483, -0.0108259, 0.0101533
2: -0.0054856, 0.0053729, -0.0059112, 0.0057119, -0.0111975, 0.0112841
3: -0.0000456, 0.0023840, -0.0001372, 0.0023860, -0.0018186, 0.0019167
4: -0.0064206, 0.0014377, -0.0068514, 0.0015488, -0.0079693, 0.0082891
5: -0.0020836, 0.0075947, -0.0022327, 0.0081337, -0.0102173, 0.0098273
6: -0.0082835, 0.0018979, -0.0089489, 0.0019505, -0.0102340, 0.0108467
7: -0.0055487, -0.0001932, -0.0055521, 0.0000442, -0.0055930, 0.0053589
8: -0.0131191, -0.0024943, -0.0134510, -0.0024839, -0.0106353, 0.0109567
9: -0.0039095, 0.0073593, -0.0045031, 0.0075540, -0.0114635, 0.0118624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087897, upper bound: 0.0090874
time: 0.93 seconds

## Relational analysis of IS_A2_A2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087864, upper bound: 0.0089727
time: 0.99 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: -0.0063066, 0.0045274, -0.0059166, 0.0040168, -0.0103234, 0.0104440
1: 0.9975052, 1.0095377, 0.9973761, 1.0087633, -0.0095354, 0.0103011
2: -0.0055114, 0.0053934, -0.0050043, 0.0049895, -0.0105009, 0.0103977
3: -0.0000512, 0.0023784, 0.0000580, 0.0024320, -0.0018718, 0.0017534
4: -0.0064467, 0.0014444, -0.0059333, 0.0013122, -0.0077588, 0.0073778
5: -0.0020926, 0.0076273, -0.0019150, 0.0069851, -0.0090777, 0.0095423
6: -0.0083238, 0.0019011, -0.0075309, 0.0018383, -0.0101621, 0.0094319
7: -0.0055394, -0.0001788, -0.0056298, -0.0004617, -0.0050777, 0.0054510
8: -0.0131392, -0.0025231, -0.0127438, -0.0022436, -0.0108956, 0.0102207
9: -0.0039454, 0.0073711, -0.0032382, 0.0071392, -0.0110846, 0.0106093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A2_B1_B1_B1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089304, upper bound: 0.0087747
time: 1.00 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089304, upper bound: 0.0087747
time: 0.93 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: -0.0062983, 0.0045165, -0.0061168, 0.0042789, -0.0105772, 0.0106333
1: 0.9975218, 1.0095212, 0.9973899, 1.0091609, -0.0098829, 0.0103153
2: -0.0055006, 0.0053849, -0.0052646, 0.0051968, -0.0106975, 0.0106494
3: -0.0000488, 0.0023716, 0.0000020, 0.0024263, -0.0018868, 0.0017812
4: -0.0064358, 0.0014416, -0.0061968, 0.0013801, -0.0078158, 0.0076385
5: -0.0020888, 0.0076137, -0.0020062, 0.0073147, -0.0094035, 0.0096198
6: -0.0083069, 0.0018997, -0.0079378, 0.0018705, -0.0101774, 0.0098375
7: -0.0055279, -0.0001848, -0.0056201, -0.0003165, -0.0052114, 0.0054353
8: -0.0131308, -0.0025587, -0.0129468, -0.0022736, -0.0108572, 0.0103881
9: -0.0039304, 0.0073662, -0.0036012, 0.0072582, -0.0111886, 0.0109674

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_B1

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088147, upper bound: 0.0087723
time: 0.93 seconds

## Relational analysis of IS_A2_A2_A2_B1_B1_B2_B2

### Relational analysis result of IS_A2_A2_A2_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088147, upper bound: 0.0088439
time: 0.98 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0062926, 0.0045090, -0.0063406, 0.0045718, -0.0108644, 0.0108496
1: 0.9975553, 1.0095097, 0.9973369, 1.0096052, -0.0103314, 0.0102115
2: -0.0054932, 0.0053789, -0.0055556, 0.0054286, -0.0109218, 0.0109345
3: -0.0000472, 0.0023578, -0.0000607, 0.0024483, -0.0018320, 0.0018561
4: -0.0064282, 0.0014397, -0.0064914, 0.0014560, -0.0078842, 0.0079311
5: -0.0020862, 0.0076042, -0.0021081, 0.0076832, -0.0097695, 0.0097123
6: -0.0082953, 0.0018988, -0.0083928, 0.0019065, -0.0102018, 0.0102916
7: -0.0055045, -0.0001890, -0.0056573, -0.0001542, -0.0053503, 0.0054684
8: -0.0131250, -0.0026310, -0.0131736, -0.0021586, -0.0109665, 0.0105427
9: -0.0039200, 0.0073628, -0.0040070, 0.0073913, -0.0113113, 0.0113698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_A2_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087834, upper bound: 0.0088870
time: 0.96 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087834, upper bound: 0.0089474
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0065172, 0.0048030, -0.0063317, 0.0045603, -0.0110774, 0.0111347
1: 0.9975573, 1.0099559, 0.9973543, 1.0095874, -0.0103451, 0.0106056
2: -0.0057852, 0.0056115, -0.0055441, 0.0054195, -0.0112046, 0.0111556
3: -0.0001101, 0.0023569, -0.0000582, 0.0024411, -0.0018685, 0.0018716
4: -0.0067238, 0.0015159, -0.0064797, 0.0014530, -0.0081768, 0.0079956
5: -0.0021885, 0.0079740, -0.0021040, 0.0076687, -0.0098572, 0.0100781
6: -0.0087518, 0.0019349, -0.0083748, 0.0019051, -0.0106569, 0.0103097
7: -0.0055031, -0.0000261, -0.0056451, -0.0001606, -0.0053425, 0.0056190
8: -0.0133527, -0.0026354, -0.0131647, -0.0021965, -0.0111562, 0.0105292
9: -0.0043273, 0.0074963, -0.0039910, 0.0073861, -0.0117133, 0.0114873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A2_A2_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_A2_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087802, upper bound: 0.0087708
time: 1.15 seconds

## Relational analysis of IS_A2_A2_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_A2_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087802, upper bound: 0.0088425
time: 0.92 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0060116, 0.0041412, -0.0062102, 0.0044011, -0.0104127, 0.0103513
1: 0.9973314, 1.0089519, 0.9974716, 1.0093462, -0.0102504, 0.0096998
2: -0.0051278, 0.0050879, -0.0053860, 0.0052935, -0.0104214, 0.0104739
3: 0.0000314, 0.0024506, -0.0000242, 0.0023925, -0.0017783, 0.0018957
4: -0.0060584, 0.0013444, -0.0063197, 0.0014117, -0.0074701, 0.0076641
5: -0.0019582, 0.0071415, -0.0020487, 0.0074685, -0.0094267, 0.0091902
6: -0.0077240, 0.0018536, -0.0081276, 0.0018855, -0.0096095, 0.0099812
7: -0.0056612, -0.0003928, -0.0055631, -0.0002488, -0.0054124, 0.0051703
8: -0.0128401, -0.0021468, -0.0130414, -0.0024499, -0.0103902, 0.0108947
9: -0.0034104, 0.0071956, -0.0037705, 0.0073137, -0.0107242, 0.0109662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_A2_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088289, upper bound: 0.0089444
time: 0.99 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_A2_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088286, upper bound: 0.0088577
time: 1.55 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0063062, 0.0045268, -0.0062102, 0.0044011, -0.0107073, 0.0107370
1: 0.9974928, 1.0095367, 0.9974716, 1.0093462, -0.0101052, 0.0102246
2: -0.0055109, 0.0053930, -0.0053860, 0.0052935, -0.0108044, 0.0107790
3: -0.0000510, 0.0023836, -0.0000242, 0.0023925, -0.0018516, 0.0018318
4: -0.0064461, 0.0014443, -0.0063197, 0.0014117, -0.0078579, 0.0077640
5: -0.0020924, 0.0076267, -0.0020487, 0.0074685, -0.0095609, 0.0096753
6: -0.0083229, 0.0019010, -0.0081276, 0.0018855, -0.0102085, 0.0100286
7: -0.0055482, -0.0001791, -0.0055631, -0.0002488, -0.0052994, 0.0053840
8: -0.0131388, -0.0024960, -0.0130414, -0.0024499, -0.0106889, 0.0105454
9: -0.0039447, 0.0073709, -0.0037705, 0.0073137, -0.0112584, 0.0111414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A2_B2_B1_A2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088289, upper bound: 0.0089101
time: 0.96 seconds

## Relational analysis of IS_A2_A2_A2_B2_B1_A2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088286, upper bound: 0.0090365
time: 1.19 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0060116, 0.0041412, -0.0066213, 0.0049394, -0.0109510, 0.0107625
1: 0.9973314, 1.0089519, 0.9974867, 1.0101626, -0.0110580, 0.0096033
2: -0.0051278, 0.0050879, -0.0059206, 0.0057194, -0.0108472, 0.0110085
3: 0.0000314, 0.0024506, -0.0001393, 0.0023862, -0.0017333, 0.0020187
4: -0.0060584, 0.0013444, -0.0068609, 0.0015512, -0.0076096, 0.0082053
5: -0.0019582, 0.0071415, -0.0022359, 0.0081456, -0.0101038, 0.0093774
6: -0.0077240, 0.0018536, -0.0089636, 0.0019517, -0.0096757, 0.0108172
7: -0.0056612, -0.0003928, -0.0055525, 0.0000495, -0.0057106, 0.0051596
8: -0.0128401, -0.0021468, -0.0134583, -0.0024829, -0.0103573, 0.0113115
9: -0.0034104, 0.0071956, -0.0045162, 0.0075583, -0.0109687, 0.0117118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088054, upper bound: 0.0089444
time: 1.23 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088058, upper bound: 0.0088576
time: 0.93 seconds

## BFS IS instance: IS_A2_A2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0063062, 0.0045268, -0.0066213, 0.0049394, -0.0112456, 0.0111482
1: 0.9974928, 1.0095367, 0.9974867, 1.0101626, -0.0109170, 0.0101244
2: -0.0055109, 0.0053930, -0.0059206, 0.0057194, -0.0112303, 0.0113136
3: -0.0000510, 0.0023836, -0.0001393, 0.0023862, -0.0017975, 0.0019512
4: -0.0064461, 0.0014443, -0.0068609, 0.0015512, -0.0079973, 0.0083052
5: -0.0020924, 0.0076267, -0.0022359, 0.0081456, -0.0102380, 0.0098626
6: -0.0083229, 0.0019010, -0.0089636, 0.0019517, -0.0102746, 0.0108646
7: -0.0055482, -0.0001791, -0.0055525, 0.0000495, -0.0055977, 0.0053733
8: -0.0131388, -0.0024960, -0.0134583, -0.0024829, -0.0106560, 0.0109623
9: -0.0039447, 0.0073709, -0.0045162, 0.0075583, -0.0115030, 0.0118870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of IS_A2_A2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_A2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088054, upper bound: 0.0091449
time: 0.93 seconds

## Relational analysis of IS_A2_A2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_A2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088058, upper bound: 0.0090327
time: 1.04 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.64 seconds
IS_A1_A1_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0084307, upper bound: 0.0083740
IS_A1_A1_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0083949, upper bound: 0.0082160
IS_A1_A1_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0083240, upper bound: 0.0083781
IS_A1_A1_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082855, upper bound: 0.0082167
IS_A1_A1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088848, upper bound: 0.0090387
IS_A1_A1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088814, upper bound: 0.0089104
IS_A1_A1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088848, upper bound: 0.0091255
IS_A1_A1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088814, upper bound: 0.0089793
IS_A1_A1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0083060, upper bound: 0.0082619
IS_A1_A1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0081803, upper bound: 0.0082329
IS_A1_A1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082971, upper bound: 0.0081452
IS_A1_A1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0081727, upper bound: 0.0081158
IS_A1_A1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087415, upper bound: 0.0089354
IS_A1_A1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087375, upper bound: 0.0088238
IS_A1_A1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087415, upper bound: 0.0090764
IS_A1_A1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087375, upper bound: 0.0089324
IS_A1_A1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0083271, upper bound: 0.0085013
IS_A1_A1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082900, upper bound: 0.0083112
IS_A1_A1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0083233, upper bound: 0.0083537
IS_A1_A1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082847, upper bound: 0.0081731
IS_A1_A1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088854, upper bound: 0.0090190
IS_A1_A1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088824, upper bound: 0.0089021
IS_A1_A1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088854, upper bound: 0.0089912
IS_A1_A1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088824, upper bound: 0.0089801
IS_A1_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087850, upper bound: 0.0089112
IS_A1_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087850, upper bound: 0.0089189
IS_A1_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082983, upper bound: 0.0081123
IS_A1_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0081727, upper bound: 0.0080839
IS_A1_A1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087565, upper bound: 0.0089304
IS_A1_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087533, upper bound: 0.0088147
IS_A1_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087565, upper bound: 0.0090743
IS_A1_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087533, upper bound: 0.0087459
IS_A1_A2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089124, upper bound: 0.0090305
IS_A1_A2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089124, upper bound: 0.0090837
IS_A1_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0088923
IS_A1_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0089530
IS_A1_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089239, upper bound: 0.0090788
IS_A1_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089254, upper bound: 0.0089704
IS_A1_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089239, upper bound: 0.0090541
IS_A1_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089254, upper bound: 0.0090893
IS_A1_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087710, upper bound: 0.0089444
IS_A1_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087710, upper bound: 0.0089878
IS_A1_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087667, upper bound: 0.0087998
IS_A1_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087667, upper bound: 0.0088489
IS_A1_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087697, upper bound: 0.0089798
IS_A1_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087684, upper bound: 0.0088677
IS_A1_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087697, upper bound: 0.0091408
IS_A1_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087684, upper bound: 0.0090135
IS_A1_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0090190, upper bound: 0.0088979
IS_A1_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0090190, upper bound: 0.0089577
IS_A1_A2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089021, upper bound: 0.0088935
IS_A1_A2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089021, upper bound: 0.0089581
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089363, upper bound: 0.0090835
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089357, upper bound: 0.0089775
IS_A1_A2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0090561, upper bound: 0.0089363
IS_A1_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089357, upper bound: 0.0091115
IS_A1_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087807, upper bound: 0.0089536
IS_A1_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087807, upper bound: 0.0089979
IS_A1_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087764, upper bound: 0.0088044
IS_A1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087764, upper bound: 0.0088628
IS_A1_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088086, upper bound: 0.0089896
IS_A1_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088078, upper bound: 0.0088835
IS_A1_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088086, upper bound: 0.0089623
IS_A1_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088078, upper bound: 0.0090466
IS_A2_A1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0087689
IS_A2_A1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0087710
IS_A2_A1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087595, upper bound: 0.0087637
IS_A2_A1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087595, upper bound: 0.0087637
IS_A2_A1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087290, upper bound: 0.0088452
IS_A2_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087290, upper bound: 0.0088452
IS_A2_A1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087296, upper bound: 0.0087629
IS_A2_A1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087296, upper bound: 0.0087629
IS_A2_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089412, upper bound: 0.0087226
IS_A2_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088255, upper bound: 0.0087236
IS_A2_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089412, upper bound: 0.0089157
IS_A2_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088255, upper bound: 0.0089168
IS_A2_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088866, upper bound: 0.0087160
IS_A2_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087882, upper bound: 0.0087178
IS_A2_A1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087929, upper bound: 0.0090223
IS_A2_A1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087882, upper bound: 0.0089123
IS_A2_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0083480, upper bound: 0.0082620
IS_A2_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0083047, upper bound: 0.0080799
IS_A2_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082523, upper bound: 0.0082577
IS_A2_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082116, upper bound: 0.0080807
IS_A2_A1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082143, upper bound: 0.0083535
IS_A2_A1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0081753, upper bound: 0.0081607
IS_A2_A1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0082104, upper bound: 0.0082488
IS_A2_A1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0081726, upper bound: 0.0080665
IS_A2_A1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087889, upper bound: 0.0088643
IS_A2_A1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087874, upper bound: 0.0087764
IS_A2_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089189, upper bound: 0.0089256
IS_A2_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087874, upper bound: 0.0087200
IS_A2_A1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087571, upper bound: 0.0088643
IS_A2_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087566, upper bound: 0.0087750
IS_A2_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087571, upper bound: 0.0090336
IS_A2_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087566, upper bound: 0.0087165
IS_A2_A2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089179, upper bound: 0.0087406
IS_A2_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089179, upper bound: 0.0087925
IS_A2_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088092, upper bound: 0.0087410
IS_A2_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088092, upper bound: 0.0087937
IS_A2_A2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087744, upper bound: 0.0088525
IS_A2_A2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087744, upper bound: 0.0089119
IS_A2_A2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087683, upper bound: 0.0087366
IS_A2_A2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087683, upper bound: 0.0087883
IS_A2_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088202, upper bound: 0.0089137
IS_A2_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088195, upper bound: 0.0088284
IS_A2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089527, upper bound: 0.0089747
IS_A2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088195, upper bound: 0.0087684
IS_A2_A2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087897, upper bound: 0.0089135
IS_A2_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087864, upper bound: 0.0088270
IS_A2_A2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087897, upper bound: 0.0090874
IS_A2_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087864, upper bound: 0.0089727
IS_A2_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089304, upper bound: 0.0087747
IS_A2_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0089304, upper bound: 0.0087747
IS_A2_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088147, upper bound: 0.0087723
IS_A2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088147, upper bound: 0.0088439
IS_A2_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087834, upper bound: 0.0088870
IS_A2_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087834, upper bound: 0.0089474
IS_A2_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087802, upper bound: 0.0087708
IS_A2_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0087802, upper bound: 0.0088425
IS_A2_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088289, upper bound: 0.0089444
IS_A2_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088286, upper bound: 0.0088577
IS_A2_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088289, upper bound: 0.0089101
IS_A2_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088286, upper bound: 0.0090365
IS_A2_A2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088054, upper bound: 0.0089444
IS_A2_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088058, upper bound: 0.0088576
IS_A2_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088054, upper bound: 0.0091449
IS_A2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.64
Output dim: 1, lower bound: -0.0088058, upper bound: 0.0090327

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0057066, 0.0037418, -0.0058593, 0.0039417, -0.0096483, 0.0096011
1: 0.9974536, 1.0083461, 0.9973778, 1.0086496, -0.0093835, 0.0092316
2: -0.0047312, 0.0047720, -0.0049298, 0.0049302, -0.0096613, 0.0097017
3: 0.0001168, 0.0023999, 0.0000741, 0.0024313, -0.0017388, 0.0017080
4: -0.0056569, 0.0012409, -0.0058579, 0.0012927, -0.0069496, 0.0070988
5: -0.0018193, 0.0066391, -0.0018889, 0.0068906, -0.0087099, 0.0085280
6: -0.0071038, 0.0018045, -0.0074143, 0.0018291, -0.0089328, 0.0092188
7: -0.0055757, -0.0006141, -0.0056286, -0.0005033, -0.0050724, 0.0050146
8: -0.0125309, -0.0024111, -0.0126857, -0.0022473, -0.0102835, 0.0102746
9: -0.0028572, 0.0070142, -0.0031342, 0.0071051, -0.0099623, 0.0101484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0074809, upper bound: 0.0075978
time: 0.83 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0073747, upper bound: 0.0073901
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0054472, 0.0034023, -0.0057333, 0.0037768, -0.0092240, 0.0091355
1: 0.9973829, 1.0078312, 0.9973821, 1.0083994, -0.0092187, 0.0088137
2: -0.0043939, 0.0045034, -0.0047659, 0.0047997, -0.0091936, 0.0092693
3: 0.0001894, 0.0024292, 0.0001094, 0.0024295, -0.0017069, 0.0017086
4: -0.0053155, 0.0012036, -0.0056920, 0.0012500, -0.0065655, 0.0068956
5: -0.0017012, 0.0062120, -0.0018315, 0.0066831, -0.0083843, 0.0080435
6: -0.0065765, 0.0017628, -0.0071581, 0.0018088, -0.0083853, 0.0089209
7: -0.0056251, -0.0008022, -0.0056256, -0.0005947, -0.0050304, 0.0048234
8: -0.0122679, -0.0022582, -0.0125580, -0.0022567, -0.0100112, 0.0102997
9: -0.0023869, 0.0068600, -0.0029057, 0.0070301, -0.0094170, 0.0097656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083365, upper bound: 0.0081957
time: 0.80 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083365, upper bound: 0.0082160
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0056980, 0.0037306, -0.0060719, 0.0042201, -0.0099181, 0.0098025
1: 0.9974711, 1.0083292, 0.9973916, 1.0090716, -0.0097486, 0.0092494
2: -0.0047201, 0.0047631, -0.0052062, 0.0051503, -0.0098704, 0.0099693
3: 0.0001192, 0.0023926, 0.0000146, 0.0024256, -0.0017555, 0.0017362
4: -0.0056456, 0.0012380, -0.0061377, 0.0013648, -0.0070104, 0.0073757
5: -0.0018154, 0.0066250, -0.0019857, 0.0072407, -0.0090562, 0.0086107
6: -0.0070864, 0.0018031, -0.0078465, 0.0018633, -0.0089497, 0.0096496
7: -0.0055634, -0.0006203, -0.0056190, -0.0003491, -0.0052143, 0.0049987
8: -0.0125222, -0.0024490, -0.0129012, -0.0022772, -0.0102450, 0.0104522
9: -0.0028417, 0.0070091, -0.0035197, 0.0072315, -0.0100732, 0.0105289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0074259, upper bound: 0.0076147
time: 0.69 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0073290, upper bound: 0.0073950
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0054387, 0.0033912, -0.0059432, 0.0040516, -0.0094904, 0.0093344
1: 0.9974029, 1.0078145, 0.9973961, 1.0088160, -0.0095809, 0.0088315
2: -0.0043829, 0.0044946, -0.0050389, 0.0050171, -0.0094000, 0.0095335
3: 0.0001918, 0.0024209, 0.0000506, 0.0024238, -0.0017236, 0.0017349
4: -0.0053044, 0.0011944, -0.0059683, 0.0013212, -0.0066255, 0.0071627
5: -0.0016973, 0.0061981, -0.0019271, 0.0070288, -0.0087262, 0.0081252
6: -0.0065593, 0.0017614, -0.0075849, 0.0018426, -0.0084019, 0.0093463
7: -0.0056111, -0.0008084, -0.0056159, -0.0004424, -0.0051687, 0.0048075
8: -0.0122594, -0.0023015, -0.0127708, -0.0022867, -0.0099726, 0.0104693
9: -0.0023715, 0.0068549, -0.0032864, 0.0071550, -0.0095265, 0.0101413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082172, upper bound: 0.0081946
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082172, upper bound: 0.0082167
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0055129, 0.0034883, -0.0061456, 0.0043166, -0.0098295, 0.0096339
1: 0.9973258, 1.0079616, 0.9974849, 1.0092180, -0.0099879, 0.0087887
2: -0.0044794, 0.0045714, -0.0053020, 0.0052267, -0.0097060, 0.0098734
3: 0.0001710, 0.0024529, -0.0000061, 0.0023869, -0.0016581, 0.0018257
4: -0.0054020, 0.0012300, -0.0062347, 0.0013898, -0.0067918, 0.0074647
5: -0.0017311, 0.0063202, -0.0020193, 0.0073621, -0.0090932, 0.0083395
6: -0.0067100, 0.0017733, -0.0079964, 0.0018751, -0.0085852, 0.0097697
7: -0.0056651, -0.0007545, -0.0055538, -0.0002956, -0.0053695, 0.0047992
8: -0.0123345, -0.0021345, -0.0129760, -0.0024787, -0.0098558, 0.0108414
9: -0.0025060, 0.0068990, -0.0036534, 0.0072753, -0.0097814, 0.0105525

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090349
time: 0.93 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090387
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0057222, 0.0037623, -0.0061368, 0.0043051, -0.0100273, 0.0098991
1: 0.9973339, 1.0083773, 0.9975037, 1.0092002, -0.0100040, 0.0091570
2: -0.0047515, 0.0047882, -0.0052906, 0.0052176, -0.0099691, 0.0100788
3: 0.0001125, 0.0024495, -0.0000036, 0.0023791, -0.0016873, 0.0018421
4: -0.0056775, 0.0012462, -0.0062231, 0.0013868, -0.0070643, 0.0074693
5: -0.0018264, 0.0066649, -0.0020153, 0.0073476, -0.0091741, 0.0086801
6: -0.0071356, 0.0018070, -0.0079785, 0.0018737, -0.0090093, 0.0097855
7: -0.0056594, -0.0006027, -0.0055406, -0.0003020, -0.0053574, 0.0049379
8: -0.0125467, -0.0021522, -0.0129671, -0.0025195, -0.0100273, 0.0108148
9: -0.0028856, 0.0070235, -0.0036375, 0.0072701, -0.0101557, 0.0106610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0089071
time: 0.91 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0089104
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0057645, 0.0038177, -0.0061456, 0.0043166, -0.0100811, 0.0099633
1: 0.9974977, 1.0084614, 0.9974849, 1.0092180, -0.0098510, 0.0092524
2: -0.0048065, 0.0048320, -0.0053020, 0.0052267, -0.0100332, 0.0101340
3: 0.0001006, 0.0023815, -0.0000061, 0.0023869, -0.0017253, 0.0017610
4: -0.0057331, 0.0012606, -0.0062347, 0.0013898, -0.0071229, 0.0074953
5: -0.0018457, 0.0067345, -0.0020193, 0.0073621, -0.0092078, 0.0087538
6: -0.0072216, 0.0018138, -0.0079964, 0.0018751, -0.0090967, 0.0098102
7: -0.0055447, -0.0005721, -0.0055538, -0.0002956, -0.0052490, 0.0049817
8: -0.0125896, -0.0025069, -0.0129760, -0.0024787, -0.0101109, 0.0104691
9: -0.0029623, 0.0070487, -0.0036534, 0.0072753, -0.0102376, 0.0107021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089877, upper bound: 0.0091235
time: 0.89 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089877, upper bound: 0.0091255
time: 1.10 seconds

## BFS IS instance: IS_A1_A1_A1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0059864, 0.0041081, -0.0061368, 0.0043051, -0.0102914, 0.0102449
1: 0.9975027, 1.0089020, 0.9975037, 1.0092002, -0.0098668, 0.0096542
2: -0.0050950, 0.0050617, -0.0052906, 0.0052176, -0.0103126, 0.0103523
3: 0.0000385, 0.0023795, -0.0000036, 0.0023791, -0.0017570, 0.0017784
4: -0.0060251, 0.0013358, -0.0062231, 0.0013868, -0.0074120, 0.0075589
5: -0.0019467, 0.0070999, -0.0020153, 0.0073476, -0.0092944, 0.0091152
6: -0.0076726, 0.0018495, -0.0079785, 0.0018737, -0.0095464, 0.0098280
7: -0.0055413, -0.0004111, -0.0055406, -0.0003020, -0.0052393, 0.0051295
8: -0.0128145, -0.0025173, -0.0129671, -0.0025195, -0.0102951, 0.0104498
9: -0.0033646, 0.0071806, -0.0036375, 0.0072701, -0.0106347, 0.0108181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089884, upper bound: 0.0089767
time: 0.93 seconds

## Relational analysis of IS_A1_A1_A1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089884, upper bound: 0.0089793
time: 1.04 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0057810, 0.0038393, -0.0062035, 0.0043924, -0.0101735, 0.0100428
1: 0.9974979, 1.0084941, 0.9973425, 1.0093330, -0.0100444, 0.0093289
2: -0.0048280, 0.0048491, -0.0053774, 0.0052867, -0.0101147, 0.0102265
3: 0.0000960, 0.0023815, -0.0000223, 0.0024460, -0.0017378, 0.0018107
4: -0.0057549, 0.0012662, -0.0063110, 0.0014095, -0.0071643, 0.0075772
5: -0.0018532, 0.0067617, -0.0020457, 0.0074576, -0.0093108, 0.0088074
6: -0.0072552, 0.0018165, -0.0081142, 0.0018845, -0.0091396, 0.0099307
7: -0.0055446, -0.0005601, -0.0056535, -0.0002536, -0.0052910, 0.0050934
8: -0.0126064, -0.0025071, -0.0130347, -0.0021706, -0.0104358, 0.0105276
9: -0.0029923, 0.0070585, -0.0037585, 0.0073098, -0.0103021, 0.0108170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_A1_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0074263, upper bound: 0.0072034
time: 0.84 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0072698, upper bound: 0.0071216
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0056494, 0.0036670, -0.0059483, 0.0040583, -0.0097077, 0.0096153
1: 0.9975022, 1.0082328, 0.9972762, 1.0088263, -0.0096343, 0.0091610
2: -0.0046569, 0.0047128, -0.0050455, 0.0050224, -0.0096793, 0.0097584
3: 0.0001328, 0.0023797, 0.0000492, 0.0024735, -0.0017393, 0.0017775
4: -0.0055817, 0.0012215, -0.0059750, 0.0013229, -0.0069046, 0.0071966
5: -0.0017933, 0.0065451, -0.0019294, 0.0070372, -0.0088305, 0.0084745
6: -0.0069877, 0.0017953, -0.0075953, 0.0018434, -0.0088311, 0.0093906
7: -0.0055416, -0.0006555, -0.0056998, -0.0004387, -0.0051029, 0.0050443
8: -0.0124730, -0.0025164, -0.0127760, -0.0020272, -0.0104458, 0.0102595
9: -0.0027537, 0.0069803, -0.0032956, 0.0071580, -0.0099117, 0.0102759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_A1_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0073374, upper bound: 0.0071663
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0070795, upper bound: 0.0070921
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0059988, 0.0041244, -0.0061947, 0.0043808, -0.0103796, 0.0103191
1: 0.9975029, 1.0089265, 0.9973599, 1.0093155, -0.0100583, 0.0097328
2: -0.0051112, 0.0050747, -0.0053658, 0.0052775, -0.0103887, 0.0104405
3: 0.0000350, 0.0023795, -0.0000198, 0.0024387, -0.0017756, 0.0018254
4: -0.0060415, 0.0013400, -0.0062993, 0.0014065, -0.0074480, 0.0076393
5: -0.0019524, 0.0071204, -0.0020416, 0.0074429, -0.0093953, 0.0091621
6: -0.0076980, 0.0018515, -0.0080961, 0.0018830, -0.0095810, 0.0099476
7: -0.0055412, -0.0004021, -0.0056412, -0.0002600, -0.0052811, 0.0052391
8: -0.0128272, -0.0025177, -0.0130257, -0.0022086, -0.0106186, 0.0105080
9: -0.0033873, 0.0071880, -0.0037424, 0.0073045, -0.0106918, 0.0109304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_A1_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0074205, upper bound: 0.0071408
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0072530, upper bound: 0.0070428
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058678, 0.0039530, -0.0059398, 0.0040471, -0.0099149, 0.0098927
1: 0.9975073, 1.0086665, 0.9972959, 1.0088092, -0.0096481, 0.0095600
2: -0.0049409, 0.0049390, -0.0050344, 0.0050135, -0.0099544, 0.0099734
3: 0.0000717, 0.0023776, 0.0000516, 0.0024653, -0.0017726, 0.0017921
4: -0.0058692, 0.0012956, -0.0059638, 0.0013200, -0.0071892, 0.0072594
5: -0.0018928, 0.0069047, -0.0019255, 0.0070232, -0.0089159, 0.0088302
6: -0.0074317, 0.0018305, -0.0075779, 0.0018420, -0.0092737, 0.0094083
7: -0.0055380, -0.0004971, -0.0056861, -0.0004449, -0.0050930, 0.0051890
8: -0.0126944, -0.0025276, -0.0127673, -0.0020698, -0.0106246, 0.0102397
9: -0.0031497, 0.0071101, -0.0032801, 0.0071529, -0.0103026, 0.0103903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_A1_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0073348, upper bound: 0.0070977
time: 0.92 seconds

## Relational analysis of IS_A1_A1_A1_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0070659, upper bound: 0.0069891
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0055129, 0.0034883, -0.0065524, 0.0048491, -0.0103620, 0.0100407
1: 0.9973258, 1.0079616, 0.9975005, 1.0100259, -0.0108238, 0.0087150
2: -0.0044794, 0.0045714, -0.0058310, 0.0056480, -0.0101273, 0.0104024
3: 0.0001710, 0.0024529, -0.0001200, 0.0023805, -0.0016326, 0.0019710
4: -0.0054020, 0.0012300, -0.0067701, 0.0015278, -0.0069298, 0.0080001
5: -0.0017311, 0.0063202, -0.0022045, 0.0080320, -0.0097632, 0.0085247
6: -0.0067100, 0.0017733, -0.0088234, 0.0019406, -0.0086506, 0.0105967
7: -0.0056651, -0.0007545, -0.0055428, -0.0000006, -0.0056646, 0.0047883
8: -0.0123345, -0.0021345, -0.0133884, -0.0025126, -0.0098219, 0.0112539
9: -0.0025060, 0.0068990, -0.0043911, 0.0075173, -0.0100233, 0.0112902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087366, upper bound: 0.0089302
time: 1.02 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087366, upper bound: 0.0089354
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0057222, 0.0037623, -0.0065437, 0.0048377, -0.0105599, 0.0103060
1: 0.9973339, 1.0083773, 0.9975169, 1.0100083, -0.0108399, 0.0090937
2: -0.0047515, 0.0047882, -0.0058196, 0.0056389, -0.0103905, 0.0106078
3: 0.0001125, 0.0024495, -0.0001175, 0.0023736, -0.0016663, 0.0019874
4: -0.0056775, 0.0012462, -0.0067587, 0.0015249, -0.0072023, 0.0080049
5: -0.0018264, 0.0066649, -0.0022006, 0.0080177, -0.0098441, 0.0088655
6: -0.0071356, 0.0018070, -0.0088057, 0.0019392, -0.0090748, 0.0106127
7: -0.0056594, -0.0006027, -0.0055313, -0.0000069, -0.0056525, 0.0049286
8: -0.0125467, -0.0021522, -0.0133795, -0.0025482, -0.0099986, 0.0112273
9: -0.0028856, 0.0070235, -0.0043753, 0.0075121, -0.0103977, 0.0113988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087372, upper bound: 0.0088216
time: 0.91 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087372, upper bound: 0.0088238
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0057645, 0.0038177, -0.0065524, 0.0048491, -0.0106136, 0.0103701
1: 0.9974977, 1.0084614, 0.9975005, 1.0100259, -0.0106986, 0.0091740
2: -0.0048065, 0.0048320, -0.0058310, 0.0056480, -0.0104545, 0.0106629
3: 0.0001006, 0.0023815, -0.0001200, 0.0023805, -0.0016939, 0.0019077
4: -0.0057331, 0.0012606, -0.0067701, 0.0015278, -0.0072609, 0.0080307
5: -0.0018457, 0.0067345, -0.0022045, 0.0080320, -0.0098777, 0.0089391
6: -0.0072216, 0.0018138, -0.0088234, 0.0019406, -0.0091622, 0.0106372
7: -0.0055447, -0.0005721, -0.0055428, -0.0000006, -0.0055441, 0.0049708
8: -0.0125896, -0.0025069, -0.0133884, -0.0025126, -0.0100770, 0.0108815
9: -0.0029623, 0.0070487, -0.0043911, 0.0075173, -0.0104796, 0.0114398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090721
time: 1.20 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090764
time: 0.93 seconds

## BFS IS instance: IS_A1_A1_A1_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0059864, 0.0041081, -0.0065437, 0.0048377, -0.0108241, 0.0106518
1: 0.9975027, 1.0089020, 0.9975169, 1.0100083, -0.0107145, 0.0095885
2: -0.0050950, 0.0050617, -0.0058196, 0.0056389, -0.0107339, 0.0108814
3: 0.0000385, 0.0023795, -0.0001175, 0.0023736, -0.0017322, 0.0019251
4: -0.0060251, 0.0013358, -0.0067587, 0.0015249, -0.0075500, 0.0080945
5: -0.0019467, 0.0070999, -0.0022006, 0.0080177, -0.0099644, 0.0093005
6: -0.0076726, 0.0018495, -0.0088057, 0.0019392, -0.0096118, 0.0106552
7: -0.0055413, -0.0004111, -0.0055313, -0.0000069, -0.0055344, 0.0051202
8: -0.0128145, -0.0025173, -0.0133795, -0.0025482, -0.0102664, 0.0108622
9: -0.0033646, 0.0071806, -0.0043753, 0.0075121, -0.0108767, 0.0115559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089332, upper bound: 0.0089307
time: 1.01 seconds

## Relational analysis of IS_A1_A1_A1_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089332, upper bound: 0.0089324
time: 1.01 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -0.0057002, 0.0037335, -0.0058765, 0.0039643, -0.0096646, 0.0096100
1: 0.9975087, 1.0083336, 0.9973317, 1.0086838, -0.0094264, 0.0091670
2: -0.0047229, 0.0047654, -0.0049522, 0.0049480, -0.0096710, 0.0097176
3: 0.0001186, 0.0023770, 0.0000692, 0.0024505, -0.0017161, 0.0017173
4: -0.0056485, 0.0012387, -0.0058806, 0.0012986, -0.0069471, 0.0071193
5: -0.0018164, 0.0066287, -0.0018967, 0.0069191, -0.0087355, 0.0085254
6: -0.0070909, 0.0018035, -0.0074494, 0.0018319, -0.0089227, 0.0092529
7: -0.0055370, -0.0006187, -0.0056610, -0.0004908, -0.0050462, 0.0050423
8: -0.0125244, -0.0025306, -0.0127032, -0.0021473, -0.0103771, 0.0101726
9: -0.0028457, 0.0070104, -0.0031655, 0.0071153, -0.0099611, 0.0101759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083267, upper bound: 0.0085013
time: 0.92 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083267, upper bound: 0.0085013
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -0.0054443, 0.0033985, -0.0057531, 0.0038028, -0.0092471, 0.0091517
1: 0.9974395, 1.0078256, 0.9973360, 1.0084386, -0.0092572, 0.0087529
2: -0.0043902, 0.0045004, -0.0047917, 0.0048202, -0.0092105, 0.0092922
3: 0.0001902, 0.0024057, 0.0001038, 0.0024487, -0.0016853, 0.0017156
4: -0.0053117, 0.0011775, -0.0057182, 0.0012567, -0.0065684, 0.0068957
5: -0.0016999, 0.0062073, -0.0018405, 0.0067158, -0.0084157, 0.0080478
6: -0.0065707, 0.0017623, -0.0071985, 0.0018120, -0.0083827, 0.0089608
7: -0.0055854, -0.0008043, -0.0056579, -0.0005803, -0.0050051, 0.0048537
8: -0.0122650, -0.0023809, -0.0125781, -0.0021567, -0.0101083, 0.0101972
9: -0.0023817, 0.0068583, -0.0029417, 0.0070419, -0.0094236, 0.0098000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082882, upper bound: 0.0083112
time: 0.86 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082882, upper bound: 0.0083112
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.0059190, 0.0040199, -0.0058682, 0.0039534, -0.0098724, 0.0098881
1: 0.9975148, 1.0087680, 0.9973515, 1.0086672, -0.0094425, 0.0095604
2: -0.0050074, 0.0049920, -0.0049413, 0.0049394, -0.0099468, 0.0099333
3: 0.0000574, 0.0023745, 0.0000716, 0.0024422, -0.0017484, 0.0017333
4: -0.0059365, 0.0013130, -0.0058696, 0.0012957, -0.0072322, 0.0071826
5: -0.0019161, 0.0069889, -0.0018929, 0.0069053, -0.0088214, 0.0088819
6: -0.0075357, 0.0018387, -0.0074324, 0.0018305, -0.0093662, 0.0092711
7: -0.0055328, -0.0004600, -0.0056469, -0.0004968, -0.0050360, 0.0051869
8: -0.0127462, -0.0025435, -0.0126947, -0.0021907, -0.0105555, 0.0101512
9: -0.0032425, 0.0071406, -0.0031503, 0.0071104, -0.0103528, 0.0102909

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083233, upper bound: 0.0083537
time: 0.99 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083233, upper bound: 0.0083537
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.0056511, 0.0036692, -0.0057448, 0.0037918, -0.0094429, 0.0094139
1: 0.9974539, 1.0082358, 0.9973560, 1.0084220, -0.0092753, 0.0091311
2: -0.0046590, 0.0047145, -0.0047808, 0.0048116, -0.0094706, 0.0094954
3: 0.0001324, 0.0023997, 0.0001061, 0.0024404, -0.0017163, 0.0017322
4: -0.0055838, 0.0012221, -0.0057071, 0.0012539, -0.0068377, 0.0069292
5: -0.0017940, 0.0065478, -0.0018367, 0.0067020, -0.0084961, 0.0083845
6: -0.0069910, 0.0017956, -0.0071814, 0.0018106, -0.0088016, 0.0089770
7: -0.0055754, -0.0006543, -0.0056440, -0.0005864, -0.0049890, 0.0049897
8: -0.0124746, -0.0024120, -0.0125696, -0.0021998, -0.0102749, 0.0101576
9: -0.0027566, 0.0069812, -0.0029265, 0.0070369, -0.0097936, 0.0099077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082838, upper bound: 0.0081731
time: 0.96 seconds

## Relational analysis of IS_A1_A1_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082838, upper bound: 0.0081731
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0055107, 0.0034854, -0.0061538, 0.0043273, -0.0098380, 0.0096392
1: 0.9973381, 1.0079573, 0.9974844, 1.0092342, -0.0100950, 0.0087181
2: -0.0044765, 0.0045691, -0.0053127, 0.0052352, -0.0097117, 0.0098818
3: 0.0001717, 0.0024478, -0.0000084, 0.0023872, -0.0016338, 0.0018579
4: -0.0053991, 0.0012243, -0.0062455, 0.0013926, -0.0067917, 0.0074697
5: -0.0017301, 0.0063166, -0.0020230, 0.0073756, -0.0091057, 0.0083396
6: -0.0067056, 0.0017730, -0.0080130, 0.0018765, -0.0085820, 0.0097860
7: -0.0056565, -0.0007562, -0.0055542, -0.0002897, -0.0053668, 0.0047980
8: -0.0123323, -0.0021612, -0.0129843, -0.0024776, -0.0098547, 0.0108231
9: -0.0025020, 0.0068977, -0.0036683, 0.0072802, -0.0097822, 0.0105660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090162
time: 0.91 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090190
time: 0.93 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0057245, 0.0037652, -0.0061451, 0.0043159, -0.0100403, 0.0099103
1: 0.9973476, 1.0083816, 0.9975032, 1.0092168, -0.0101113, 0.0090905
2: -0.0047544, 0.0047905, -0.0053013, 0.0052261, -0.0099805, 0.0100918
3: 0.0001118, 0.0024439, -0.0000059, 0.0023794, -0.0016642, 0.0018737
4: -0.0056804, 0.0012470, -0.0062340, 0.0013896, -0.0070700, 0.0074810
5: -0.0018275, 0.0066686, -0.0020190, 0.0073613, -0.0091887, 0.0086876
6: -0.0071401, 0.0018074, -0.0079953, 0.0018751, -0.0090152, 0.0098027
7: -0.0056498, -0.0006011, -0.0055410, -0.0002960, -0.0053538, 0.0049399
8: -0.0125490, -0.0021818, -0.0129754, -0.0025183, -0.0100307, 0.0107936
9: -0.0028897, 0.0070249, -0.0036524, 0.0072750, -0.0101647, 0.0106773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0088980
time: 0.91 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0089021
time: 0.99 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0057853, 0.0038448, -0.0061538, 0.0043273, -0.0101126, 0.0099986
1: 0.9975043, 1.0085024, 0.9974844, 1.0092342, -0.0099477, 0.0092139
2: -0.0048335, 0.0048535, -0.0053127, 0.0052352, -0.0100687, 0.0101662
3: 0.0000948, 0.0023788, -0.0000084, 0.0023872, -0.0016962, 0.0017914
4: -0.0057604, 0.0012676, -0.0062455, 0.0013926, -0.0071531, 0.0075131
5: -0.0018552, 0.0067687, -0.0020230, 0.0073756, -0.0092308, 0.0087917
6: -0.0072638, 0.0018172, -0.0080130, 0.0018765, -0.0091402, 0.0098302
7: -0.0055401, -0.0005570, -0.0055542, -0.0002897, -0.0052504, 0.0049972
8: -0.0126107, -0.0025210, -0.0129843, -0.0024776, -0.0101331, 0.0104632
9: -0.0029999, 0.0070610, -0.0036683, 0.0072802, -0.0102801, 0.0107293

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089879, upper bound: 0.0091157
time: 0.86 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089879, upper bound: 0.0091167
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0060092, 0.0041380, -0.0061451, 0.0043159, -0.0103251, 0.0102831
1: 0.9975101, 1.0089470, 0.9975032, 1.0092168, -0.0099656, 0.0096170
2: -0.0051247, 0.0050854, -0.0053013, 0.0052261, -0.0103508, 0.0103868
3: 0.0000321, 0.0023764, -0.0000059, 0.0023794, -0.0017292, 0.0018082
4: -0.0060552, 0.0013436, -0.0062340, 0.0013896, -0.0074448, 0.0075776
5: -0.0019572, 0.0071375, -0.0020190, 0.0073613, -0.0093184, 0.0091565
6: -0.0077191, 0.0018532, -0.0079953, 0.0018751, -0.0095941, 0.0098485
7: -0.0055361, -0.0003946, -0.0055410, -0.0002960, -0.0052400, 0.0051464
8: -0.0128377, -0.0025335, -0.0129754, -0.0025183, -0.0103194, 0.0104419
9: -0.0034061, 0.0071942, -0.0036524, 0.0072750, -0.0106811, 0.0108467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089889, upper bound: 0.0089793
time: 0.98 seconds

## Relational analysis of IS_A1_A1_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089889, upper bound: 0.0089800
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0057905, 0.0038518, -0.0061344, 0.0043019, -0.0100924, 0.0099861
1: 0.9975044, 1.0085130, 0.9973212, 1.0091957, -0.0099523, 0.0093496
2: -0.0048404, 0.0048590, -0.0052874, 0.0052151, -0.0100554, 0.0101464
3: 0.0000933, 0.0023788, -0.0000029, 0.0024548, -0.0017371, 0.0018104
4: -0.0057674, 0.0012694, -0.0062199, 0.0013860, -0.0071534, 0.0074893
5: -0.0018576, 0.0067774, -0.0020142, 0.0073436, -0.0092012, 0.0087916
6: -0.0072745, 0.0018180, -0.0079735, 0.0018733, -0.0091479, 0.0097916
7: -0.0055401, -0.0005532, -0.0056682, -0.0003037, -0.0052363, 0.0051151
8: -0.0126160, -0.0025211, -0.0129646, -0.0021249, -0.0104911, 0.0104435
9: -0.0030095, 0.0070642, -0.0036331, 0.0072687, -0.0102782, 0.0106972

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A2_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082965, upper bound: 0.0082414
time: 0.98 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081704, upper bound: 0.0082189
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0057905, 0.0038518, -0.0061242, 0.0042885, -0.0100791, 0.0099759
1: 0.9975044, 1.0085130, 0.9973408, 1.0091753, -0.0098906, 0.0093160
2: -0.0048404, 0.0048590, -0.0052742, 0.0052045, -0.0100449, 0.0101331
3: 0.0000933, 0.0023788, -0.0000001, 0.0024467, -0.0017259, 0.0017917
4: -0.0057674, 0.0012694, -0.0062065, 0.0013826, -0.0071499, 0.0074759
5: -0.0018576, 0.0067774, -0.0020095, 0.0073268, -0.0091844, 0.0087869
6: -0.0072745, 0.0018180, -0.0079528, 0.0018717, -0.0091462, 0.0097708
7: -0.0055401, -0.0005532, -0.0056545, -0.0003112, -0.0052289, 0.0051014
8: -0.0126160, -0.0025211, -0.0129542, -0.0021672, -0.0104488, 0.0104331
9: -0.0030095, 0.0070642, -0.0036145, 0.0072626, -0.0102721, 0.0106787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 178

## Relational analysis of IS_A1_A1_A2_B2_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082965, upper bound: 0.0082414
time: 0.95 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0081704, upper bound: 0.0082189
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0060101, 0.0041392, -0.0061917, 0.0043769, -0.0103870, 0.0103309
1: 0.9975104, 1.0089489, 0.9973593, 1.0093093, -0.0100995, 0.0097179
2: -0.0051259, 0.0050864, -0.0053620, 0.0052744, -0.0104003, 0.0104483
3: 0.0000319, 0.0023764, -0.0000190, 0.0024390, -0.0017631, 0.0018454
4: -0.0060564, 0.0013439, -0.0062954, 0.0014055, -0.0074618, 0.0076392
5: -0.0019576, 0.0071390, -0.0020403, 0.0074380, -0.0093956, 0.0091793
6: -0.0077209, 0.0018533, -0.0080900, 0.0018826, -0.0096035, 0.0099434
7: -0.0055359, -0.0003939, -0.0056415, -0.0002622, -0.0052737, 0.0052476
8: -0.0128386, -0.0025339, -0.0130227, -0.0022074, -0.0106312, 0.0104888
9: -0.0034077, 0.0071948, -0.0037370, 0.0073027, -0.0107105, 0.0109317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 173

## Relational analysis of IS_A1_A1_A2_B2_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082886, upper bound: 0.0081123
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082886, upper bound: 0.0081123
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0058773, 0.0039654, -0.0059341, 0.0040397, -0.0099170, 0.0098995
1: 0.9975148, 1.0086854, 0.9972952, 1.0087980, -0.0096859, 0.0095551
2: -0.0049532, 0.0049488, -0.0050270, 0.0050076, -0.0099608, 0.0099759
3: 0.0000690, 0.0023745, 0.0000531, 0.0024656, -0.0017659, 0.0018111
4: -0.0058816, 0.0012988, -0.0059563, 0.0013181, -0.0071997, 0.0072552
5: -0.0018971, 0.0069203, -0.0019229, 0.0070138, -0.0089109, 0.0088433
6: -0.0074509, 0.0018320, -0.0075664, 0.0018411, -0.0092921, 0.0093984
7: -0.0055328, -0.0004902, -0.0056864, -0.0004490, -0.0050838, 0.0051962
8: -0.0127040, -0.0025435, -0.0127616, -0.0020687, -0.0106353, 0.0102181
9: -0.0031669, 0.0071158, -0.0032699, 0.0071495, -0.0103164, 0.0103857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_A2_B2_B1_A2_B2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0073573, upper bound: 0.0071111
time: 0.78 seconds

## Relational analysis of IS_A1_A1_A2_B2_B1_A2_B2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0070844, upper bound: 0.0070131
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0055107, 0.0034854, -0.0065588, 0.0048575, -0.0103682, 0.0100442
1: 0.9973381, 1.0079573, 0.9975001, 1.0100384, -0.0109200, 0.0087005
2: -0.0044765, 0.0045691, -0.0058393, 0.0056546, -0.0101311, 0.0104084
3: 0.0001717, 0.0024478, -0.0001217, 0.0023806, -0.0016282, 0.0019977
4: -0.0053991, 0.0012243, -0.0067785, 0.0015300, -0.0069290, 0.0080028
5: -0.0017301, 0.0063166, -0.0022075, 0.0080425, -0.0097726, 0.0085240
6: -0.0067056, 0.0017730, -0.0088364, 0.0019416, -0.0086472, 0.0106094
7: -0.0056565, -0.0007562, -0.0055431, 0.0000041, -0.0056606, 0.0047870
8: -0.0123323, -0.0021612, -0.0133948, -0.0025116, -0.0098207, 0.0112336
9: -0.0025020, 0.0068977, -0.0044027, 0.0075211, -0.0100231, 0.0113004

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087378, upper bound: 0.0089240
time: 1.14 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087378, upper bound: 0.0089304
time: 0.93 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0057245, 0.0037652, -0.0065502, 0.0048463, -0.0105707, 0.0103155
1: 0.9973476, 1.0083816, 0.9975165, 1.0100214, -0.0109367, 0.0090803
2: -0.0047544, 0.0047905, -0.0058282, 0.0056457, -0.0104002, 0.0106187
3: 0.0001118, 0.0024439, -0.0001194, 0.0023738, -0.0016616, 0.0020136
4: -0.0056804, 0.0012470, -0.0067673, 0.0015271, -0.0072075, 0.0080143
5: -0.0018275, 0.0066686, -0.0022036, 0.0080285, -0.0098559, 0.0088721
6: -0.0071401, 0.0018074, -0.0088190, 0.0019402, -0.0090804, 0.0106264
7: -0.0056498, -0.0006011, -0.0055317, -0.0000021, -0.0056477, 0.0049305
8: -0.0125490, -0.0021818, -0.0133862, -0.0025471, -0.0100019, 0.0112044
9: -0.0028897, 0.0070249, -0.0043872, 0.0075160, -0.0104057, 0.0114121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087381, upper bound: 0.0088102
time: 0.88 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0087381, upper bound: 0.0088147
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0057853, 0.0038448, -0.0065588, 0.0048575, -0.0106427, 0.0104036
1: 0.9975043, 1.0085024, 0.9975001, 1.0100384, -0.0107794, 0.0091878
2: -0.0048335, 0.0048535, -0.0058393, 0.0056546, -0.0104880, 0.0106927
3: 0.0000948, 0.0023788, -0.0001217, 0.0023806, -0.0016886, 0.0019341
4: -0.0057604, 0.0012676, -0.0067785, 0.0015300, -0.0072904, 0.0080461
5: -0.0018552, 0.0067687, -0.0022075, 0.0080425, -0.0098977, 0.0089762
6: -0.0072638, 0.0018172, -0.0088364, 0.0019416, -0.0092054, 0.0106535
7: -0.0055401, -0.0005570, -0.0055431, 0.0000041, -0.0055442, 0.0049862
8: -0.0126107, -0.0025210, -0.0133948, -0.0025116, -0.0100990, 0.0108738
9: -0.0029999, 0.0070610, -0.0044027, 0.0075211, -0.0105210, 0.0114637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=8, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090700
time: 1.02 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090743
time: 0.96 seconds

## BFS IS instance: IS_A1_A1_A2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0060092, 0.0041380, -0.0065502, 0.0048463, -0.0108555, 0.0106882
1: 0.9975101, 1.0089470, 0.9975165, 1.0100214, -0.0107979, 0.0096019
2: -0.0051247, 0.0050854, -0.0058282, 0.0056457, -0.0107704, 0.0109136
3: 0.0000321, 0.0023764, -0.0001194, 0.0023738, -0.0017257, 0.0019509
4: -0.0060552, 0.0013436, -0.0067673, 0.0015271, -0.0075823, 0.0081109
5: -0.0019572, 0.0071375, -0.0022036, 0.0080285, -0.0099856, 0.0093411
6: -0.0077191, 0.0018532, -0.0088190, 0.0019402, -0.0096593, 0.0106722
7: -0.0055361, -0.0003946, -0.0055317, -0.0000021, -0.0055339, 0.0051371
8: -0.0128377, -0.0025335, -0.0133862, -0.0025471, -0.0102906, 0.0108527
9: -0.0034061, 0.0071942, -0.0043872, 0.0075160, -0.0109221, 0.0115814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=16, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A1_A2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_A2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089339, upper bound: 0.0089378
time: 0.93 seconds

## Relational analysis of IS_A1_A1_A2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_A2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089339, upper bound: 0.0089399
time: 1.07 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0060297, 0.0041648, -0.0055263, 0.0035058, -0.0095355, 0.0096911
1: 0.9975332, 1.0089878, 0.9972792, 1.0079883, -0.0087662, 0.0098323
2: -0.0051513, 0.0051066, -0.0044968, 0.0045853, -0.0097366, 0.0096034
3: 0.0000264, 0.0023669, 0.0001673, 0.0024722, -0.0018145, 0.0016377
4: -0.0060822, 0.0013505, -0.0054196, 0.0012514, -0.0073335, 0.0067701
5: -0.0019665, 0.0071712, -0.0017372, 0.0063423, -0.0083087, 0.0089085
6: -0.0077607, 0.0018565, -0.0067373, 0.0017755, -0.0095362, 0.0085938
7: -0.0055199, -0.0003797, -0.0056977, -0.0007448, -0.0047751, 0.0053180
8: -0.0128585, -0.0025833, -0.0123481, -0.0020339, -0.0108245, 0.0097649
9: -0.0034432, 0.0072064, -0.0025303, 0.0069070, -0.0103502, 0.0097367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083067, upper bound: 0.0085089
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082749, upper bound: 0.0083667
time: 0.80 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0060297, 0.0041648, -0.0055889, 0.0035878, -0.0096175, 0.0097537
1: 0.9975332, 1.0089878, 0.9973258, 1.0081127, -0.0088250, 0.0097568
2: -0.0051513, 0.0051066, -0.0045782, 0.0046502, -0.0098015, 0.0096849
3: 0.0000264, 0.0023669, 0.0001498, 0.0024529, -0.0017685, 0.0016262
4: -0.0060822, 0.0013505, -0.0055020, 0.0012299, -0.0073121, 0.0068526
5: -0.0019665, 0.0071712, -0.0017657, 0.0064454, -0.0084119, 0.0089370
6: -0.0077607, 0.0018565, -0.0068646, 0.0017856, -0.0095463, 0.0087211
7: -0.0055199, -0.0003797, -0.0056651, -0.0006994, -0.0048205, 0.0052854
8: -0.0128585, -0.0025833, -0.0124116, -0.0021347, -0.0107238, 0.0098283
9: -0.0034432, 0.0072064, -0.0026439, 0.0069443, -0.0103875, 0.0098503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083067, upper bound: 0.0086173
time: 0.93 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082749, upper bound: 0.0084625
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0062363, 0.0044353, -0.0055181, 0.0034952, -0.0097315, 0.0099535
1: 0.9975445, 1.0093979, 0.9972972, 1.0079721, -0.0087824, 0.0101918
2: -0.0054200, 0.0053206, -0.0044862, 0.0045769, -0.0099969, 0.0098068
3: -0.0000315, 0.0023622, 0.0001696, 0.0024647, -0.0018415, 0.0016536
4: -0.0063541, 0.0014206, -0.0054089, 0.0012430, -0.0075972, 0.0068295
5: -0.0020606, 0.0075115, -0.0017335, 0.0063288, -0.0083894, 0.0092450
6: -0.0081808, 0.0018897, -0.0067207, 0.0017742, -0.0099550, 0.0086105
7: -0.0055120, -0.0002298, -0.0056850, -0.0007507, -0.0047612, 0.0054552
8: -0.0130679, -0.0026079, -0.0123399, -0.0020731, -0.0109949, 0.0097320
9: -0.0038179, 0.0073293, -0.0025156, 0.0069022, -0.0107201, 0.0098448

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083060, upper bound: 0.0083918
time: 1.06 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082771, upper bound: 0.0082470
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0062363, 0.0044353, -0.0055807, 0.0035770, -0.0098133, 0.0100160
1: 0.9975445, 1.0093979, 0.9973450, 1.0080963, -0.0088384, 0.0101197
2: -0.0054200, 0.0053206, -0.0045675, 0.0046416, -0.0100616, 0.0098882
3: -0.0000315, 0.0023622, 0.0001521, 0.0024450, -0.0018015, 0.0016405
4: -0.0063541, 0.0014206, -0.0054912, 0.0012211, -0.0075752, 0.0069118
5: -0.0020606, 0.0075115, -0.0017620, 0.0064318, -0.0084924, 0.0092735
6: -0.0081808, 0.0018897, -0.0068479, 0.0017843, -0.0099651, 0.0087376
7: -0.0055120, -0.0002298, -0.0056517, -0.0007054, -0.0048066, 0.0054219
8: -0.0130679, -0.0026079, -0.0124033, -0.0021761, -0.0108918, 0.0097954
9: -0.0038179, 0.0073293, -0.0026290, 0.0069394, -0.0107573, 0.0099583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 178

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0083060, upper bound: 0.0084949
time: 0.98 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0082771, upper bound: 0.0083380
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -0.0057488, 0.0037971, -0.0060204, 0.0041527, -0.0099015, 0.0098175
1: 0.9973798, 1.0084300, 0.9974918, 1.0089695, -0.0099018, 0.0091417
2: -0.0047861, 0.0048158, -0.0051393, 0.0050970, -0.0098831, 0.0099550
3: 0.0001050, 0.0024305, 0.0000290, 0.0023840, -0.0016687, 0.0018615
4: -0.0057125, 0.0012552, -0.0060699, 0.0013474, -0.0070598, 0.0073252
5: -0.0018386, 0.0067087, -0.0019623, 0.0071560, -0.0089945, 0.0086710
6: -0.0071897, 0.0018113, -0.0077419, 0.0018550, -0.0090447, 0.0095532
7: -0.0056272, -0.0005834, -0.0055488, -0.0003864, -0.0052408, 0.0049654
8: -0.0125737, -0.0022516, -0.0128491, -0.0024941, -0.0100796, 0.0105974
9: -0.0029339, 0.0070394, -0.0034264, 0.0072009, -0.0101347, 0.0104657

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088572, upper bound: 0.0090419
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088572, upper bound: 0.0090788
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -0.0059460, 0.0040553, -0.0060116, 0.0041411, -0.0100871, 0.0100668
1: 0.9973940, 1.0088216, 0.9975100, 1.0089517, -0.0099190, 0.0094842
2: -0.0050425, 0.0050200, -0.0051277, 0.0050878, -0.0101303, 0.0101477
3: 0.0000498, 0.0024247, 0.0000315, 0.0023765, -0.0016980, 0.0018781
4: -0.0059720, 0.0013221, -0.0060583, 0.0013444, -0.0073164, 0.0073804
5: -0.0019284, 0.0070334, -0.0019582, 0.0071414, -0.0090697, 0.0089916
6: -0.0075906, 0.0018430, -0.0077238, 0.0018536, -0.0094442, 0.0095668
7: -0.0056174, -0.0004404, -0.0055362, -0.0003929, -0.0052246, 0.0050958
8: -0.0127736, -0.0022820, -0.0128401, -0.0025332, -0.0102404, 0.0105581
9: -0.0032914, 0.0071566, -0.0034103, 0.0071956, -0.0104871, 0.0105669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088602, upper bound: 0.0089237
time: 0.93 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0088602, upper bound: 0.0089704
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: -0.0060182, 0.0041498, -0.0060204, 0.0041527, -0.0101709, 0.0101702
1: 0.9975329, 1.0089650, 0.9974918, 1.0089695, -0.0097546, 0.0096193
2: -0.0051364, 0.0050948, -0.0051393, 0.0050970, -0.0102335, 0.0102340
3: 0.0000296, 0.0023670, 0.0000290, 0.0023840, -0.0017295, 0.0017961
4: -0.0060671, 0.0013466, -0.0060699, 0.0013474, -0.0074144, 0.0074166
5: -0.0019613, 0.0071524, -0.0019623, 0.0071560, -0.0091172, 0.0091146
6: -0.0077374, 0.0018547, -0.0077419, 0.0018550, -0.0095924, 0.0095965
7: -0.0055201, -0.0003880, -0.0055488, -0.0003864, -0.0051337, 0.0051608
8: -0.0128468, -0.0025828, -0.0128491, -0.0024941, -0.0103528, 0.0102662
9: -0.0034224, 0.0071996, -0.0034264, 0.0072009, -0.0106233, 0.0106260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089939, upper bound: 0.0091520
time: 0.96 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089939, upper bound: 0.0092170
time: 1.24 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: -0.0062262, 0.0044221, -0.0060116, 0.0041411, -0.0103672, 0.0104336
1: 0.9975442, 1.0093780, 0.9975100, 1.0089517, -0.0097728, 0.0099934
2: -0.0054068, 0.0053101, -0.0051277, 0.0050878, -0.0104946, 0.0104379
3: -0.0000286, 0.0023623, 0.0000315, 0.0023765, -0.0017616, 0.0018131
4: -0.0063408, 0.0014172, -0.0060583, 0.0013444, -0.0076851, 0.0074754
5: -0.0020560, 0.0074948, -0.0019582, 0.0071414, -0.0091973, 0.0094531
6: -0.0081602, 0.0018881, -0.0077238, 0.0018536, -0.0100138, 0.0096119
7: -0.0055122, -0.0002372, -0.0055362, -0.0003929, -0.0051194, 0.0052990
8: -0.0130577, -0.0026072, -0.0128401, -0.0025332, -0.0105245, 0.0102329
9: -0.0037996, 0.0073233, -0.0034103, 0.0071956, -0.0109952, 0.0107336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089960, upper bound: 0.0090206
time: 0.96 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0089960, upper bound: 0.0090893
time: 1.02 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.86 seconds
IS_A1_A1_A1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0074809, upper bound: 0.0075978
IS_A1_A1_A1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0073747, upper bound: 0.0073901
IS_A1_A1_A1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083365, upper bound: 0.0081957
IS_A1_A1_A1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083365, upper bound: 0.0082160
IS_A1_A1_A1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0074259, upper bound: 0.0076147
IS_A1_A1_A1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0073290, upper bound: 0.0073950
IS_A1_A1_A1_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082172, upper bound: 0.0081946
IS_A1_A1_A1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082172, upper bound: 0.0082167
IS_A1_A1_A1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090349
IS_A1_A1_A1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090387
IS_A1_A1_A1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0089071
IS_A1_A1_A1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0089104
IS_A1_A1_A1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089877, upper bound: 0.0091235
IS_A1_A1_A1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089877, upper bound: 0.0091255
IS_A1_A1_A1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089884, upper bound: 0.0089767
IS_A1_A1_A1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089884, upper bound: 0.0089793
IS_A1_A1_A1_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0074263, upper bound: 0.0072034
IS_A1_A1_A1_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0072698, upper bound: 0.0071216
IS_A1_A1_A1_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0073374, upper bound: 0.0071663
IS_A1_A1_A1_B2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0070795, upper bound: 0.0070921
IS_A1_A1_A1_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0074205, upper bound: 0.0071408
IS_A1_A1_A1_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0072530, upper bound: 0.0070428
IS_A1_A1_A1_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0073348, upper bound: 0.0070977
IS_A1_A1_A1_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0070659, upper bound: 0.0069891
IS_A1_A1_A1_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087366, upper bound: 0.0089302
IS_A1_A1_A1_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087366, upper bound: 0.0089354
IS_A1_A1_A1_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087372, upper bound: 0.0088216
IS_A1_A1_A1_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087372, upper bound: 0.0088238
IS_A1_A1_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090721
IS_A1_A1_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090764
IS_A1_A1_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089332, upper bound: 0.0089307
IS_A1_A1_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089332, upper bound: 0.0089324
IS_A1_A1_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083267, upper bound: 0.0085013
IS_A1_A1_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083267, upper bound: 0.0085013
IS_A1_A1_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082882, upper bound: 0.0083112
IS_A1_A1_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082882, upper bound: 0.0083112
IS_A1_A1_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083233, upper bound: 0.0083537
IS_A1_A1_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083233, upper bound: 0.0083537
IS_A1_A1_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082838, upper bound: 0.0081731
IS_A1_A1_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082838, upper bound: 0.0081731
IS_A1_A1_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090162
IS_A1_A1_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088558, upper bound: 0.0090190
IS_A1_A1_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0088980
IS_A1_A1_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088554, upper bound: 0.0089021
IS_A1_A1_A2_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089879, upper bound: 0.0091157
IS_A1_A1_A2_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089879, upper bound: 0.0091167
IS_A1_A1_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089889, upper bound: 0.0089793
IS_A1_A1_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089889, upper bound: 0.0089800
IS_A1_A1_A2_B2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082965, upper bound: 0.0082414
IS_A1_A1_A2_B2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0081704, upper bound: 0.0082189
IS_A1_A1_A2_B2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082965, upper bound: 0.0082414
IS_A1_A1_A2_B2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0081704, upper bound: 0.0082189
IS_A1_A1_A2_B2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082886, upper bound: 0.0081123
IS_A1_A1_A2_B2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082886, upper bound: 0.0081123
IS_A1_A1_A2_B2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0073573, upper bound: 0.0071111
IS_A1_A1_A2_B2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0070844, upper bound: 0.0070131
IS_A1_A1_A2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087378, upper bound: 0.0089240
IS_A1_A1_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087378, upper bound: 0.0089304
IS_A1_A1_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087381, upper bound: 0.0088102
IS_A1_A1_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0087381, upper bound: 0.0088147
IS_A1_A1_A2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090700
IS_A1_A1_A2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089337, upper bound: 0.0090743
IS_A1_A1_A2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089339, upper bound: 0.0089378
IS_A1_A1_A2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089339, upper bound: 0.0089399
IS_A1_A2_B1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083067, upper bound: 0.0085089
IS_A1_A2_B1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082749, upper bound: 0.0083667
IS_A1_A2_B1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083067, upper bound: 0.0086173
IS_A1_A2_B1_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082749, upper bound: 0.0084625
IS_A1_A2_B1_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083060, upper bound: 0.0083918
IS_A1_A2_B1_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082771, upper bound: 0.0082470
IS_A1_A2_B1_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0083060, upper bound: 0.0084949
IS_A1_A2_B1_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0082771, upper bound: 0.0083380
IS_A1_A2_B1_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088572, upper bound: 0.0090419
IS_A1_A2_B1_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088572, upper bound: 0.0090788
IS_A1_A2_B1_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088602, upper bound: 0.0089237
IS_A1_A2_B1_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0088602, upper bound: 0.0089704
IS_A1_A2_B1_B1_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089939, upper bound: 0.0091520
IS_A1_A2_B1_B1_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089939, upper bound: 0.0092170
IS_A1_A2_B1_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089960, upper bound: 0.0090206
IS_A1_A2_B1_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.86
Output dim: 1, lower bound: -0.0089960, upper bound: 0.0090893
IS_A1_A2_B1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087710, upper bound: 0.0089444
IS_A1_A2_B1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087710, upper bound: 0.0089878
IS_A1_A2_B1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087667, upper bound: 0.0087998
IS_A1_A2_B1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087667, upper bound: 0.0088489
IS_A1_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087697, upper bound: 0.0089798
IS_A1_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087684, upper bound: 0.0088677
IS_A1_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087697, upper bound: 0.0091408
IS_A1_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087684, upper bound: 0.0090135
IS_A1_A2_B2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0090190, upper bound: 0.0088979
IS_A1_A2_B2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0090190, upper bound: 0.0089577
IS_A1_A2_B2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089021, upper bound: 0.0088935
IS_A1_A2_B2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089021, upper bound: 0.0089581
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089363, upper bound: 0.0090835
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089357, upper bound: 0.0089775
IS_A1_A2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0090561, upper bound: 0.0089363
IS_A1_A2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089357, upper bound: 0.0091115
IS_A1_A2_B2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087807, upper bound: 0.0089536
IS_A1_A2_B2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087807, upper bound: 0.0089979
IS_A1_A2_B2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087764, upper bound: 0.0088044
IS_A1_A2_B2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087764, upper bound: 0.0088628
IS_A1_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088086, upper bound: 0.0089896
IS_A1_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088078, upper bound: 0.0088835
IS_A1_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088086, upper bound: 0.0089623
IS_A1_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088078, upper bound: 0.0090466
IS_A2_A1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0087689
IS_A2_A1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089104, upper bound: 0.0087710
IS_A2_A1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087595, upper bound: 0.0087637
IS_A2_A1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087595, upper bound: 0.0087637
IS_A2_A1_A1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087290, upper bound: 0.0088452
IS_A2_A1_A1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087290, upper bound: 0.0088452
IS_A2_A1_A1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087296, upper bound: 0.0087629
IS_A2_A1_A1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087296, upper bound: 0.0087629
IS_A2_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089412, upper bound: 0.0087226
IS_A2_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088255, upper bound: 0.0087236
IS_A2_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089412, upper bound: 0.0089157
IS_A2_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088255, upper bound: 0.0089168
IS_A2_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088866, upper bound: 0.0087160
IS_A2_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087882, upper bound: 0.0087178
IS_A2_A1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087929, upper bound: 0.0090223
IS_A2_A1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087882, upper bound: 0.0089123
IS_A2_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0083480, upper bound: 0.0082620
IS_A2_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0083047, upper bound: 0.0080799
IS_A2_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0082523, upper bound: 0.0082577
IS_A2_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0082116, upper bound: 0.0080807
IS_A2_A1_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0082143, upper bound: 0.0083535
IS_A2_A1_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0081753, upper bound: 0.0081607
IS_A2_A1_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0082104, upper bound: 0.0082488
IS_A2_A1_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0081726, upper bound: 0.0080665
IS_A2_A1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087889, upper bound: 0.0088643
IS_A2_A1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087874, upper bound: 0.0087764
IS_A2_A1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089189, upper bound: 0.0089256
IS_A2_A1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087874, upper bound: 0.0087200
IS_A2_A1_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087571, upper bound: 0.0088643
IS_A2_A1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087566, upper bound: 0.0087750
IS_A2_A1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087571, upper bound: 0.0090336
IS_A2_A1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087566, upper bound: 0.0087165
IS_A2_A2_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089179, upper bound: 0.0087406
IS_A2_A2_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089179, upper bound: 0.0087925
IS_A2_A2_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088092, upper bound: 0.0087410
IS_A2_A2_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088092, upper bound: 0.0087937
IS_A2_A2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087744, upper bound: 0.0088525
IS_A2_A2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087744, upper bound: 0.0089119
IS_A2_A2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087683, upper bound: 0.0087366
IS_A2_A2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087683, upper bound: 0.0087883
IS_A2_A2_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088202, upper bound: 0.0089137
IS_A2_A2_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088195, upper bound: 0.0088284
IS_A2_A2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089527, upper bound: 0.0089747
IS_A2_A2_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088195, upper bound: 0.0087684
IS_A2_A2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087897, upper bound: 0.0089135
IS_A2_A2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087864, upper bound: 0.0088270
IS_A2_A2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087897, upper bound: 0.0090874
IS_A2_A2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087864, upper bound: 0.0089727
IS_A2_A2_A2_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089304, upper bound: 0.0087747
IS_A2_A2_A2_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0089304, upper bound: 0.0087747
IS_A2_A2_A2_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088147, upper bound: 0.0087723
IS_A2_A2_A2_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088147, upper bound: 0.0088439
IS_A2_A2_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087834, upper bound: 0.0088870
IS_A2_A2_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087834, upper bound: 0.0089474
IS_A2_A2_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087802, upper bound: 0.0087708
IS_A2_A2_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0087802, upper bound: 0.0088425
IS_A2_A2_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088289, upper bound: 0.0089444
IS_A2_A2_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088286, upper bound: 0.0088577
IS_A2_A2_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088289, upper bound: 0.0089101
IS_A2_A2_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088286, upper bound: 0.0090365
IS_A2_A2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088054, upper bound: 0.0089444
IS_A2_A2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088058, upper bound: 0.0088576
IS_A2_A2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088054, upper bound: 0.0091449
IS_A2_A2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.86
Output dim: 1, lower bound: -0.0088058, upper bound: 0.0090327

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.71 + 597.59 = 601.30 seconds
