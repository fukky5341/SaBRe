## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00285264


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0004659, 0.0021812, -0.0004659, 0.0021812, -0.0026471, 0.0026471)
1: (0.9912602, 0.9968659, 0.9912602, 0.9968659, -0.0056057, 0.0056057)
2: (-0.0082389, -0.0037419, -0.0082389, -0.0037419, -0.0044970, 0.0044970)
3: (0.0021052, 0.0054169, 0.0021052, 0.0054169, -0.0033117, 0.0033117)
4: (0.0013744, 0.0070714, 0.0013744, 0.0070714, -0.0056970, 0.0056970)
5: (0.0022582, 0.0085310, 0.0022582, 0.0085310, -0.0062728, 0.0062728)
6: (-0.0043501, 0.0017389, -0.0043501, 0.0017389, -0.0060890, 0.0060890)
7: (-0.0089512, -0.0062681, -0.0089512, -0.0062681, -0.0026832, 0.0026832)
8: (0.0031680, 0.0082528, 0.0031680, 0.0082528, -0.0050824, 0.0050824)
9: (-0.0052176, -0.0013860, -0.0052176, -0.0013860, -0.0038315, 0.0038315)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.21 + 1.92 = 3.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -0.0032964, upper bound: 0.0032964

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 229
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 229

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032252, upper bound: 0.0031246
time: 1.13 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0032525, upper bound: 0.0032525
time: 1.05 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 1, lower bound: -0.0032252, upper bound: 0.0031246
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.32
Output dim: 1, lower bound: -0.0032525, upper bound: 0.0032525

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0002911, 0.0022048, -0.0004396, 0.0021807, -0.0024717, 0.0026444
1: 0.9912101, 0.9964957, 0.9912613, 0.9968100, -0.0055999, 0.0052344
2: -0.0081515, -0.0036603, -0.0082258, -0.0037437, -0.0044078, 0.0045655
3: 0.0023239, 0.0054465, 0.0021381, 0.0054162, -0.0030923, 0.0033083
4: 0.0013099, 0.0071529, 0.0013759, 0.0070696, -0.0057597, 0.0057770
5: 0.0026726, 0.0085870, 0.0023207, 0.0085298, -0.0058572, 0.0062664
6: -0.0044142, 0.0013558, -0.0043487, 0.0016812, -0.0060953, 0.0057045
7: -0.0089752, -0.0064453, -0.0089507, -0.0062947, -0.0026805, 0.0025054
8: 0.0030608, 0.0082295, 0.0031704, 0.0082493, -0.0051863, 0.0050590
9: -0.0052518, -0.0016391, -0.0052168, -0.0014242, -0.0038276, 0.0035777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0031246
time: 1.05 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0031246
time: 1.17 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0004069, 0.0021798, -0.0004659, 0.0021812, -0.0025880, 0.0026457
1: 0.9912631, 0.9967408, 0.9912602, 0.9968659, -0.0056028, 0.0054806
2: -0.0082094, -0.0037466, -0.0082389, -0.0037419, -0.0044675, 0.0044924
3: 0.0021791, 0.0054152, 0.0021052, 0.0054169, -0.0032378, 0.0033101
4: 0.0013781, 0.0070668, 0.0013744, 0.0070714, -0.0056933, 0.0056923
5: 0.0023982, 0.0085278, 0.0022582, 0.0085310, -0.0061329, 0.0062696
6: -0.0043465, 0.0016095, -0.0043501, 0.0017389, -0.0060853, 0.0059596
7: -0.0089499, -0.0063279, -0.0089512, -0.0062681, -0.0026818, 0.0026234
8: 0.0031741, 0.0082449, 0.0031680, 0.0082528, -0.0050763, 0.0050736
9: -0.0052156, -0.0014715, -0.0052176, -0.0013860, -0.0038296, 0.0037461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 229
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 229

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0032252
time: 1.33 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0032525
time: 1.08 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.71 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0031246
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0031246
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0032252
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.71
Output dim: 1, lower bound: -0.0031246, upper bound: 0.0032525

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002911, 0.0022048, -0.0002911, 0.0022048, -0.0024959, 0.0024959
1: 0.9912101, 0.9964957, 0.9912101, 0.9964957, -0.0052856, 0.0052856
2: -0.0081515, -0.0036603, -0.0081515, -0.0036603, -0.0044912, 0.0044912
3: 0.0023239, 0.0054465, 0.0023239, 0.0054465, -0.0031226, 0.0031226
4: 0.0013099, 0.0071529, 0.0013099, 0.0071529, -0.0058430, 0.0058430
5: 0.0026726, 0.0085870, 0.0026726, 0.0085870, -0.0059145, 0.0059145
6: -0.0044142, 0.0013558, -0.0044142, 0.0013558, -0.0057699, 0.0057699
7: -0.0089752, -0.0064453, -0.0089752, -0.0064453, -0.0025299, 0.0025299
8: 0.0030608, 0.0082295, 0.0030608, 0.0082295, -0.0051687, 0.0051687
9: -0.0052518, -0.0016391, -0.0052518, -0.0016391, -0.0036127, 0.0036127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031141
time: 1.06 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
time: 1.17 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002911, 0.0022048, -0.0004069, 0.0021798, -0.0024709, 0.0026117
1: 0.9912101, 0.9964957, 0.9912631, 0.9967408, -0.0055307, 0.0052326
2: -0.0081515, -0.0036603, -0.0082094, -0.0037466, -0.0044049, 0.0045491
3: 0.0023239, 0.0054465, 0.0021791, 0.0054152, -0.0030913, 0.0032674
4: 0.0013099, 0.0071529, 0.0013781, 0.0070668, -0.0057569, 0.0057748
5: 0.0026726, 0.0085870, 0.0023982, 0.0085278, -0.0058553, 0.0061889
6: -0.0044142, 0.0013558, -0.0043465, 0.0016095, -0.0060236, 0.0057023
7: -0.0089752, -0.0064453, -0.0089499, -0.0063279, -0.0026473, 0.0025046
8: 0.0030608, 0.0082295, 0.0031741, 0.0082449, -0.0051795, 0.0050553
9: -0.0052518, -0.0016391, -0.0052156, -0.0014715, -0.0037803, 0.0035765

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031141
time: 1.12 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
time: 1.27 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004069, 0.0021798, -0.0002911, 0.0022048, -0.0026117, 0.0024709
1: 0.9912631, 0.9967408, 0.9912101, 0.9964957, -0.0052326, 0.0055307
2: -0.0082094, -0.0037466, -0.0081515, -0.0036603, -0.0045491, 0.0044049
3: 0.0021791, 0.0054152, 0.0023239, 0.0054465, -0.0032674, 0.0030913
4: 0.0013781, 0.0070668, 0.0013099, 0.0071529, -0.0057748, 0.0057569
5: 0.0023982, 0.0085278, 0.0026726, 0.0085870, -0.0061889, 0.0058553
6: -0.0043465, 0.0016095, -0.0044142, 0.0013558, -0.0057023, 0.0060236
7: -0.0089499, -0.0063279, -0.0089752, -0.0064453, -0.0025046, 0.0026473
8: 0.0031741, 0.0082449, 0.0030608, 0.0082295, -0.0050553, 0.0051795
9: -0.0052156, -0.0014715, -0.0052518, -0.0016391, -0.0035765, 0.0037803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031874
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031860
time: 0.96 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004069, 0.0021798, -0.0004069, 0.0021798, -0.0025867, 0.0025867
1: 0.9912631, 0.9967408, 0.9912631, 0.9967408, -0.0054777, 0.0054777
2: -0.0082094, -0.0037466, -0.0082094, -0.0037466, -0.0044628, 0.0044628
3: 0.0021791, 0.0054152, 0.0021791, 0.0054152, -0.0032362, 0.0032362
4: 0.0013781, 0.0070668, 0.0013781, 0.0070668, -0.0056887, 0.0056887
5: 0.0023982, 0.0085278, 0.0023982, 0.0085278, -0.0061297, 0.0061297
6: -0.0043465, 0.0016095, -0.0043465, 0.0016095, -0.0059560, 0.0059560
7: -0.0089499, -0.0063279, -0.0089499, -0.0063279, -0.0026220, 0.0026220
8: 0.0031741, 0.0082449, 0.0031741, 0.0082449, -0.0050675, 0.0050675
9: -0.0052156, -0.0014715, -0.0052156, -0.0014715, -0.0037441, 0.0037441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0032163
time: 1.20 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0032164
time: 1.09 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.48 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031141
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031141
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031874
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031860
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0032163
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.48
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0032164

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0021124, -0.0002911, 0.0022048, -0.0024890, 0.0024035
1: 0.9914058, 0.9964809, 0.9912101, 0.9964957, -0.0050898, 0.0052708
2: -0.0081480, -0.0039794, -0.0081515, -0.0036603, -0.0044878, 0.0041721
3: 0.0023326, 0.0053309, 0.0023239, 0.0054465, -0.0031139, 0.0030070
4: 0.0015621, 0.0068344, 0.0013099, 0.0071529, -0.0055907, 0.0055246
5: 0.0026889, 0.0083681, 0.0026726, 0.0085870, -0.0058981, 0.0056955
6: -0.0041639, 0.0013406, -0.0044142, 0.0013558, -0.0055196, 0.0057548
7: -0.0088815, -0.0064523, -0.0089752, -0.0064453, -0.0024363, 0.0025229
8: 0.0034801, 0.0082285, 0.0030608, 0.0082295, -0.0047493, 0.0051539
9: -0.0051180, -0.0016491, -0.0052518, -0.0016391, -0.0034789, 0.0036027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
time: 1.44 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003524, 0.0021505, -0.0002907, 0.0021985, -0.0025509, 0.0024413
1: 0.9913251, 0.9966255, 0.9912236, 0.9964949, -0.0051697, 0.0054019
2: -0.0081822, -0.0038478, -0.0081513, -0.0036822, -0.0045000, 0.0043035
3: 0.0022472, 0.0053786, 0.0023243, 0.0054385, -0.0031913, 0.0030542
4: 0.0014581, 0.0069658, 0.0013272, 0.0071310, -0.0056729, 0.0056386
5: 0.0025273, 0.0084584, 0.0026734, 0.0085720, -0.0060447, 0.0057850
6: -0.0042671, 0.0014901, -0.0043970, 0.0013550, -0.0056221, 0.0058871
7: -0.0089202, -0.0063831, -0.0089688, -0.0064456, -0.0024745, 0.0025857
8: 0.0033071, 0.0082376, 0.0030896, 0.0082294, -0.0049223, 0.0051355
9: -0.0051732, -0.0015503, -0.0052426, -0.0016396, -0.0035336, 0.0036923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
time: 1.15 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0021124, -0.0004069, 0.0021798, -0.0024640, 0.0025193
1: 0.9914058, 0.9964809, 0.9912631, 0.9967408, -0.0053350, 0.0052179
2: -0.0081480, -0.0039794, -0.0082094, -0.0037466, -0.0044015, 0.0042300
3: 0.0023326, 0.0053309, 0.0021791, 0.0054152, -0.0030827, 0.0031518
4: 0.0015621, 0.0068344, 0.0013781, 0.0070668, -0.0055046, 0.0054563
5: 0.0026889, 0.0083681, 0.0023982, 0.0085278, -0.0058389, 0.0059699
6: -0.0041639, 0.0013406, -0.0043465, 0.0016095, -0.0057734, 0.0056871
7: -0.0088815, -0.0064523, -0.0089499, -0.0063279, -0.0025537, 0.0024976
8: 0.0034801, 0.0082285, 0.0031741, 0.0082449, -0.0047582, 0.0050407
9: -0.0051180, -0.0016491, -0.0052156, -0.0014715, -0.0036465, 0.0035665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.08 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031710, upper bound: 0.0030817
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031710, upper bound: 0.0030817
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003524, 0.0021505, -0.0004065, 0.0021729, -0.0025253, 0.0025570
1: 0.9913251, 0.9966255, 0.9912778, 0.9967400, -0.0054149, 0.0053477
2: -0.0081822, -0.0038478, -0.0082092, -0.0037706, -0.0044116, 0.0043614
3: 0.0022472, 0.0053786, 0.0021795, 0.0054065, -0.0031593, 0.0031990
4: 0.0014581, 0.0069658, 0.0013971, 0.0070428, -0.0055847, 0.0055687
5: 0.0025273, 0.0084584, 0.0023991, 0.0085113, -0.0059841, 0.0060593
6: -0.0042671, 0.0014901, -0.0043276, 0.0016086, -0.0058758, 0.0058177
7: -0.0089202, -0.0063831, -0.0089428, -0.0063283, -0.0025919, 0.0025597
8: 0.0033071, 0.0082376, 0.0032057, 0.0082449, -0.0049309, 0.0050190
9: -0.0051732, -0.0015503, -0.0052056, -0.0014721, -0.0037011, 0.0036552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031860, upper bound: 0.0030817
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031860, upper bound: 0.0030817
time: 1.12 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003998, 0.0020883, -0.0002911, 0.0022048, -0.0026046, 0.0023794
1: 0.9914569, 0.9967259, 0.9912101, 0.9964957, -0.0050387, 0.0055158
2: -0.0082059, -0.0040628, -0.0081515, -0.0036603, -0.0045456, 0.0040887
3: 0.0021879, 0.0053007, 0.0023239, 0.0054465, -0.0032586, 0.0029768
4: 0.0016280, 0.0067513, 0.0013099, 0.0071529, -0.0055249, 0.0054414
5: 0.0024149, 0.0083109, 0.0026726, 0.0085870, -0.0061721, 0.0056383
6: -0.0040985, 0.0015940, -0.0044142, 0.0013558, -0.0054543, 0.0060081
7: -0.0088571, -0.0063350, -0.0089752, -0.0064453, -0.0024118, 0.0026402
8: 0.0035896, 0.0082440, 0.0030608, 0.0082295, -0.0046399, 0.0051589
9: -0.0050831, -0.0014817, -0.0052518, -0.0016391, -0.0034440, 0.0037701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030816, upper bound: 0.0031710
time: 1.49 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030816, upper bound: 0.0031860
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004623, 0.0021186, -0.0002907, 0.0021985, -0.0026608, 0.0024093
1: 0.9913927, 0.9968583, 0.9912236, 0.9964949, -0.0051022, 0.0056347
2: -0.0082372, -0.0039580, -0.0081513, -0.0036822, -0.0045550, 0.0041933
3: 0.0021097, 0.0053386, 0.0023243, 0.0054385, -0.0033289, 0.0030143
4: 0.0015452, 0.0068558, 0.0013272, 0.0071310, -0.0055858, 0.0055286
5: 0.0022667, 0.0083827, 0.0026734, 0.0085720, -0.0063053, 0.0057094
6: -0.0041807, 0.0017310, -0.0043970, 0.0013550, -0.0055357, 0.0061280
7: -0.0088878, -0.0062717, -0.0089688, -0.0064456, -0.0024422, 0.0026971
8: 0.0034520, 0.0082523, 0.0030896, 0.0082294, -0.0047774, 0.0051391
9: -0.0051270, -0.0013912, -0.0052426, -0.0016396, -0.0034874, 0.0038514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031710
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031860
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003998, 0.0020883, -0.0004069, 0.0021798, -0.0025796, 0.0024952
1: 0.9914569, 0.9967259, 0.9912631, 0.9967408, -0.0052839, 0.0054628
2: -0.0082059, -0.0040628, -0.0082094, -0.0037466, -0.0044593, 0.0041466
3: 0.0021879, 0.0053007, 0.0021791, 0.0054152, -0.0032273, 0.0031216
4: 0.0016280, 0.0067513, 0.0013781, 0.0070668, -0.0054388, 0.0053732
5: 0.0024149, 0.0083109, 0.0023982, 0.0085278, -0.0061129, 0.0059128
6: -0.0040985, 0.0015940, -0.0043465, 0.0016095, -0.0057080, 0.0059405
7: -0.0088571, -0.0063350, -0.0089499, -0.0063279, -0.0025292, 0.0026148
8: 0.0035896, 0.0082440, 0.0031741, 0.0082449, -0.0046505, 0.0050458
9: -0.0050831, -0.0014817, -0.0052156, -0.0014715, -0.0036116, 0.0037339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031474, upper bound: 0.0031888
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031474, upper bound: 0.0032164
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004623, 0.0021186, -0.0004065, 0.0021729, -0.0026352, 0.0025251
1: 0.9913927, 0.9968583, 0.9912778, 0.9967400, -0.0053474, 0.0055805
2: -0.0082372, -0.0039580, -0.0082092, -0.0037706, -0.0044665, 0.0042512
3: 0.0021097, 0.0053386, 0.0021795, 0.0054065, -0.0032969, 0.0031591
4: 0.0015452, 0.0068558, 0.0013971, 0.0070428, -0.0054975, 0.0054587
5: 0.0022667, 0.0083827, 0.0023991, 0.0085113, -0.0062446, 0.0059836
6: -0.0041807, 0.0017310, -0.0043276, 0.0016086, -0.0057893, 0.0060586
7: -0.0088878, -0.0062717, -0.0089428, -0.0063283, -0.0025595, 0.0026712
8: 0.0034520, 0.0082523, 0.0032057, 0.0082449, -0.0047871, 0.0050231
9: -0.0051270, -0.0013912, -0.0052056, -0.0014721, -0.0036549, 0.0038143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031867, upper bound: 0.0031887
time: 1.72 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031867, upper bound: 0.0032164
time: 1.26 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.20 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0030817
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031710, upper bound: 0.0030817
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031710, upper bound: 0.0030817
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031860, upper bound: 0.0030817
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031860, upper bound: 0.0030817
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030816, upper bound: 0.0031710
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030816, upper bound: 0.0031860
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031710
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0030817, upper bound: 0.0031860
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031474, upper bound: 0.0031888
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031474, upper bound: 0.0032164
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031867, upper bound: 0.0031887
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.20
Output dim: 1, lower bound: -0.0031867, upper bound: 0.0032164

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0021124, -0.0002842, 0.0021124, -0.0023966, 0.0023966
1: 0.9914058, 0.9964809, 0.9914058, 0.9964809, -0.0050751, 0.0050751
2: -0.0081480, -0.0039794, -0.0081480, -0.0039794, -0.0041686, 0.0041686
3: 0.0023326, 0.0053309, 0.0023326, 0.0053309, -0.0029983, 0.0029983
4: 0.0015621, 0.0068344, 0.0015621, 0.0068344, -0.0052723, 0.0052723
5: 0.0026889, 0.0083681, 0.0026889, 0.0083681, -0.0056791, 0.0056791
6: -0.0041639, 0.0013406, -0.0041639, 0.0013406, -0.0055045, 0.0055045
7: -0.0088815, -0.0064523, -0.0088815, -0.0064523, -0.0024293, 0.0024293
8: 0.0034801, 0.0082285, 0.0034801, 0.0082285, -0.0047327, 0.0047326
9: -0.0051180, -0.0016491, -0.0051180, -0.0016491, -0.0034689, 0.0034689

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030400, upper bound: 0.0029003
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030962
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0021124, -0.0003524, 0.0021505, -0.0024347, 0.0024648
1: 0.9914058, 0.9964809, 0.9913251, 0.9966255, -0.0052196, 0.0051558
2: -0.0081480, -0.0039794, -0.0081822, -0.0038478, -0.0043003, 0.0042027
3: 0.0023326, 0.0053309, 0.0022472, 0.0053786, -0.0030460, 0.0030837
4: 0.0015621, 0.0068344, 0.0014581, 0.0069658, -0.0054036, 0.0053764
5: 0.0026889, 0.0083681, 0.0025273, 0.0084584, -0.0057695, 0.0058408
6: -0.0041639, 0.0013406, -0.0042671, 0.0014901, -0.0056540, 0.0056078
7: -0.0088815, -0.0064523, -0.0089202, -0.0063831, -0.0024984, 0.0024679
8: 0.0034801, 0.0082285, 0.0033071, 0.0082376, -0.0047422, 0.0049108
9: -0.0051180, -0.0016491, -0.0051732, -0.0015503, -0.0035677, 0.0035241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030400, upper bound: 0.0029003
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030962
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003524, 0.0021505, -0.0002842, 0.0021124, -0.0024648, 0.0024347
1: 0.9913251, 0.9966255, 0.9914058, 0.9964809, -0.0051558, 0.0052196
2: -0.0081822, -0.0038478, -0.0081480, -0.0039794, -0.0042027, 0.0043003
3: 0.0022472, 0.0053786, 0.0023326, 0.0053309, -0.0030837, 0.0030460
4: 0.0014581, 0.0069658, 0.0015621, 0.0068344, -0.0053764, 0.0054036
5: 0.0025273, 0.0084584, 0.0026889, 0.0083681, -0.0058408, 0.0057695
6: -0.0042671, 0.0014901, -0.0041639, 0.0013406, -0.0056078, 0.0056540
7: -0.0089202, -0.0063831, -0.0088815, -0.0064523, -0.0024679, 0.0024984
8: 0.0033071, 0.0082376, 0.0034801, 0.0082285, -0.0049108, 0.0047422
9: -0.0051732, -0.0015503, -0.0051180, -0.0016491, -0.0035241, 0.0035677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030464, upper bound: 0.0028871
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030616
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003524, 0.0021505, -0.0003524, 0.0021505, -0.0025029, 0.0025029
1: 0.9913251, 0.9966255, 0.9913251, 0.9966255, -0.0053003, 0.0053003
2: -0.0081822, -0.0038478, -0.0081822, -0.0038478, -0.0043344, 0.0043344
3: 0.0022472, 0.0053786, 0.0022472, 0.0053786, -0.0031314, 0.0031314
4: 0.0014581, 0.0069658, 0.0014581, 0.0069658, -0.0055077, 0.0055077
5: 0.0025273, 0.0084584, 0.0025273, 0.0084584, -0.0059312, 0.0059312
6: -0.0042671, 0.0014901, -0.0042671, 0.0014901, -0.0057573, 0.0057573
7: -0.0089202, -0.0063831, -0.0089202, -0.0063831, -0.0025371, 0.0025371
8: 0.0033071, 0.0082376, 0.0033071, 0.0082376, -0.0049152, 0.0049152
9: -0.0051732, -0.0015503, -0.0051732, -0.0015503, -0.0036229, 0.0036229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030464, upper bound: 0.0028871
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030616
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0021124, -0.0003998, 0.0020883, -0.0023725, 0.0025122
1: 0.9914058, 0.9964809, 0.9914569, 0.9967259, -0.0053200, 0.0050240
2: -0.0081480, -0.0039794, -0.0082059, -0.0040628, -0.0040853, 0.0042264
3: 0.0023326, 0.0053309, 0.0021879, 0.0053007, -0.0029681, 0.0031430
4: 0.0015621, 0.0068344, 0.0016280, 0.0067513, -0.0051892, 0.0052064
5: 0.0026889, 0.0083681, 0.0024149, 0.0083109, -0.0056220, 0.0059532
6: -0.0041639, 0.0013406, -0.0040985, 0.0015940, -0.0057579, 0.0054392
7: -0.0088815, -0.0064523, -0.0088571, -0.0063350, -0.0025465, 0.0024048
8: 0.0034801, 0.0082285, 0.0035896, 0.0082440, -0.0047376, 0.0046241
9: -0.0051180, -0.0016491, -0.0050831, -0.0014817, -0.0036363, 0.0034340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031247, upper bound: 0.0029003
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031490, upper bound: 0.0030962
time: 1.36 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002842, 0.0021124, -0.0004623, 0.0021186, -0.0024028, 0.0025748
1: 0.9914058, 0.9964809, 0.9913927, 0.9968583, -0.0054525, 0.0050883
2: -0.0081480, -0.0039794, -0.0082372, -0.0039580, -0.0041900, 0.0042577
3: 0.0023326, 0.0053309, 0.0021097, 0.0053386, -0.0030061, 0.0032212
4: 0.0015621, 0.0068344, 0.0015452, 0.0068558, -0.0052936, 0.0052892
5: 0.0026889, 0.0083681, 0.0022667, 0.0083827, -0.0056938, 0.0061013
6: -0.0041639, 0.0013406, -0.0041807, 0.0017310, -0.0058949, 0.0055213
7: -0.0088815, -0.0064523, -0.0088878, -0.0062717, -0.0026099, 0.0024356
8: 0.0034801, 0.0082285, 0.0034520, 0.0082523, -0.0047457, 0.0047645
9: -0.0051180, -0.0016491, -0.0051270, -0.0013912, -0.0037268, 0.0034779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031247, upper bound: 0.0029003
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031490, upper bound: 0.0030962
time: 1.49 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003524, 0.0021505, -0.0003998, 0.0020883, -0.0024407, 0.0025503
1: 0.9913251, 0.9966255, 0.9914569, 0.9967259, -0.0054007, 0.0051686
2: -0.0081822, -0.0038478, -0.0082059, -0.0040628, -0.0041194, 0.0043581
3: 0.0022472, 0.0053786, 0.0021879, 0.0053007, -0.0030535, 0.0031907
4: 0.0014581, 0.0069658, 0.0016280, 0.0067513, -0.0052932, 0.0053378
5: 0.0025273, 0.0084584, 0.0024149, 0.0083109, -0.0057837, 0.0060435
6: -0.0042671, 0.0014901, -0.0040985, 0.0015940, -0.0058611, 0.0055887
7: -0.0089202, -0.0063831, -0.0088571, -0.0063350, -0.0025851, 0.0024740
8: 0.0033071, 0.0082376, 0.0035896, 0.0082440, -0.0049157, 0.0046336
9: -0.0051732, -0.0015503, -0.0050831, -0.0014817, -0.0036915, 0.0035328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031346, upper bound: 0.0028871
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031645, upper bound: 0.0030616
time: 1.37 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003524, 0.0021505, -0.0004623, 0.0021186, -0.0024710, 0.0026129
1: 0.9913251, 0.9966255, 0.9913927, 0.9968583, -0.0055332, 0.0052328
2: -0.0081822, -0.0038478, -0.0082372, -0.0039580, -0.0042241, 0.0043894
3: 0.0022472, 0.0053786, 0.0021097, 0.0053386, -0.0030914, 0.0032689
4: 0.0014581, 0.0069658, 0.0015452, 0.0068558, -0.0053977, 0.0054206
5: 0.0025273, 0.0084584, 0.0022667, 0.0083827, -0.0058555, 0.0061917
6: -0.0042671, 0.0014901, -0.0041807, 0.0017310, -0.0059981, 0.0056708
7: -0.0089202, -0.0063831, -0.0088878, -0.0062717, -0.0026485, 0.0025047
8: 0.0033071, 0.0082376, 0.0034520, 0.0082523, -0.0049189, 0.0047713
9: -0.0051732, -0.0015503, -0.0051270, -0.0013912, -0.0037820, 0.0035767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031346, upper bound: 0.0028871
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031645, upper bound: 0.0030616
time: 1.44 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003998, 0.0020883, -0.0002842, 0.0021124, -0.0025122, 0.0023725
1: 0.9914569, 0.9967259, 0.9914058, 0.9964809, -0.0050240, 0.0053200
2: -0.0082059, -0.0040628, -0.0081480, -0.0039794, -0.0042264, 0.0040853
3: 0.0021879, 0.0053007, 0.0023326, 0.0053309, -0.0031430, 0.0029681
4: 0.0016280, 0.0067513, 0.0015621, 0.0068344, -0.0052064, 0.0051892
5: 0.0024149, 0.0083109, 0.0026889, 0.0083681, -0.0059532, 0.0056220
6: -0.0040985, 0.0015940, -0.0041639, 0.0013406, -0.0054392, 0.0057579
7: -0.0088571, -0.0063350, -0.0088815, -0.0064523, -0.0024048, 0.0025465
8: 0.0035896, 0.0082440, 0.0034801, 0.0082285, -0.0046241, 0.0047376
9: -0.0050831, -0.0014817, -0.0051180, -0.0016491, -0.0034340, 0.0036363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030402, upper bound: 0.0030446
time: 1.22 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031598
time: 1.38 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003998, 0.0020883, -0.0003524, 0.0021505, -0.0025503, 0.0024407
1: 0.9914569, 0.9967259, 0.9913251, 0.9966255, -0.0051686, 0.0054007
2: -0.0082059, -0.0040628, -0.0081822, -0.0038478, -0.0043581, 0.0041194
3: 0.0021879, 0.0053007, 0.0022472, 0.0053786, -0.0031907, 0.0030535
4: 0.0016280, 0.0067513, 0.0014581, 0.0069658, -0.0053378, 0.0052932
5: 0.0024149, 0.0083109, 0.0025273, 0.0084584, -0.0060435, 0.0057837
6: -0.0040985, 0.0015940, -0.0042671, 0.0014901, -0.0055887, 0.0058611
7: -0.0088571, -0.0063350, -0.0089202, -0.0063831, -0.0024740, 0.0025851
8: 0.0035896, 0.0082440, 0.0033071, 0.0082376, -0.0046336, 0.0049157
9: -0.0050831, -0.0014817, -0.0051732, -0.0015503, -0.0035328, 0.0036915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030402, upper bound: 0.0030446
time: 1.17 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031662
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004623, 0.0021186, -0.0002842, 0.0021124, -0.0025748, 0.0024028
1: 0.9913927, 0.9968583, 0.9914058, 0.9964809, -0.0050883, 0.0054525
2: -0.0082372, -0.0039580, -0.0081480, -0.0039794, -0.0042577, 0.0041900
3: 0.0021097, 0.0053386, 0.0023326, 0.0053309, -0.0032212, 0.0030061
4: 0.0015452, 0.0068558, 0.0015621, 0.0068344, -0.0052892, 0.0052936
5: 0.0022667, 0.0083827, 0.0026889, 0.0083681, -0.0061013, 0.0056938
6: -0.0041807, 0.0017310, -0.0041639, 0.0013406, -0.0055213, 0.0058949
7: -0.0088878, -0.0062717, -0.0088815, -0.0064523, -0.0024356, 0.0026099
8: 0.0034520, 0.0082523, 0.0034801, 0.0082285, -0.0047645, 0.0047457
9: -0.0051270, -0.0013912, -0.0051180, -0.0016491, -0.0034779, 0.0037268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030475, upper bound: 0.0030422
time: 1.19 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031490
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004623, 0.0021186, -0.0003524, 0.0021505, -0.0026129, 0.0024710
1: 0.9913927, 0.9968583, 0.9913251, 0.9966255, -0.0052328, 0.0055332
2: -0.0082372, -0.0039580, -0.0081822, -0.0038478, -0.0043894, 0.0042241
3: 0.0021097, 0.0053386, 0.0022472, 0.0053786, -0.0032689, 0.0030914
4: 0.0015452, 0.0068558, 0.0014581, 0.0069658, -0.0054206, 0.0053977
5: 0.0022667, 0.0083827, 0.0025273, 0.0084584, -0.0061917, 0.0058555
6: -0.0041807, 0.0017310, -0.0042671, 0.0014901, -0.0056708, 0.0059981
7: -0.0088878, -0.0062717, -0.0089202, -0.0063831, -0.0025047, 0.0026485
8: 0.0034520, 0.0082523, 0.0033071, 0.0082376, -0.0047713, 0.0049189
9: -0.0051270, -0.0013912, -0.0051732, -0.0015503, -0.0035767, 0.0037820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030475, upper bound: 0.0030422
time: 1.52 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031490
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003998, 0.0020883, -0.0003998, 0.0020883, -0.0024881, 0.0024881
1: 0.9914569, 0.9967259, 0.9914569, 0.9967259, -0.0052689, 0.0052689
2: -0.0082059, -0.0040628, -0.0082059, -0.0040628, -0.0041431, 0.0041431
3: 0.0021879, 0.0053007, 0.0021879, 0.0053007, -0.0031128, 0.0031128
4: 0.0016280, 0.0067513, 0.0016280, 0.0067513, -0.0051233, 0.0051233
5: 0.0024149, 0.0083109, 0.0024149, 0.0083109, -0.0058960, 0.0058960
6: -0.0040985, 0.0015940, -0.0040985, 0.0015940, -0.0056925, 0.0056925
7: -0.0088571, -0.0063350, -0.0088571, -0.0063350, -0.0025221, 0.0025221
8: 0.0035896, 0.0082440, 0.0035896, 0.0082440, -0.0046287, 0.0046287
9: -0.0050831, -0.0014817, -0.0050831, -0.0014817, -0.0036014, 0.0036014

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031178, upper bound: 0.0030822
time: 1.29 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031250, upper bound: 0.0031754
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003998, 0.0020883, -0.0004623, 0.0021186, -0.0025184, 0.0025506
1: 0.9914569, 0.9967259, 0.9913927, 0.9968583, -0.0054014, 0.0053332
2: -0.0082059, -0.0040628, -0.0082372, -0.0039580, -0.0042478, 0.0041744
3: 0.0021879, 0.0053007, 0.0021097, 0.0053386, -0.0031507, 0.0031910
4: 0.0016280, 0.0067513, 0.0015452, 0.0068558, -0.0052278, 0.0052061
5: 0.0024149, 0.0083109, 0.0022667, 0.0083827, -0.0059678, 0.0060442
6: -0.0040985, 0.0015940, -0.0041807, 0.0017310, -0.0058295, 0.0057747
7: -0.0088571, -0.0063350, -0.0088878, -0.0062717, -0.0025854, 0.0025528
8: 0.0035896, 0.0082440, 0.0034520, 0.0082523, -0.0046374, 0.0047696
9: -0.0050831, -0.0014817, -0.0051270, -0.0013912, -0.0036919, 0.0036453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031178, upper bound: 0.0030829
time: 1.30 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031250, upper bound: 0.0031933
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004623, 0.0021186, -0.0003998, 0.0020883, -0.0025506, 0.0025184
1: 0.9913927, 0.9968583, 0.9914569, 0.9967259, -0.0053332, 0.0054014
2: -0.0082372, -0.0039580, -0.0082059, -0.0040628, -0.0041744, 0.0042478
3: 0.0021097, 0.0053386, 0.0021879, 0.0053007, -0.0031910, 0.0031507
4: 0.0015452, 0.0068558, 0.0016280, 0.0067513, -0.0052061, 0.0052278
5: 0.0022667, 0.0083827, 0.0024149, 0.0083109, -0.0060442, 0.0059678
6: -0.0041807, 0.0017310, -0.0040985, 0.0015940, -0.0057747, 0.0058295
7: -0.0088878, -0.0062717, -0.0088571, -0.0063350, -0.0025528, 0.0025854
8: 0.0034520, 0.0082523, 0.0035896, 0.0082440, -0.0047696, 0.0046374
9: -0.0051270, -0.0013912, -0.0050831, -0.0014817, -0.0036453, 0.0036919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0030781
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031655, upper bound: 0.0031654
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004623, 0.0021186, -0.0004623, 0.0021186, -0.0025810, 0.0025810
1: 0.9913927, 0.9968583, 0.9913927, 0.9968583, -0.0054656, 0.0054656
2: -0.0082372, -0.0039580, -0.0082372, -0.0039580, -0.0042791, 0.0042791
3: 0.0021097, 0.0053386, 0.0021097, 0.0053386, -0.0032290, 0.0032290
4: 0.0015452, 0.0068558, 0.0015452, 0.0068558, -0.0053105, 0.0053105
5: 0.0022667, 0.0083827, 0.0022667, 0.0083827, -0.0061160, 0.0061160
6: -0.0041807, 0.0017310, -0.0041807, 0.0017310, -0.0059117, 0.0059117
7: -0.0088878, -0.0062717, -0.0088878, -0.0062717, -0.0026161, 0.0026161
8: 0.0034520, 0.0082523, 0.0034520, 0.0082523, -0.0047735, 0.0047735
9: -0.0051270, -0.0013912, -0.0051270, -0.0013912, -0.0037358, 0.0037358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0030781
time: 1.27 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0031655, upper bound: 0.0031654
time: 1.23 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.74 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030400, upper bound: 0.0029003
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030962
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030400, upper bound: 0.0029003
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030962
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030464, upper bound: 0.0028871
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030616
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030464, upper bound: 0.0028871
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0030616
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031247, upper bound: 0.0029003
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031490, upper bound: 0.0030962
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031247, upper bound: 0.0029003
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031490, upper bound: 0.0030962
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031346, upper bound: 0.0028871
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031645, upper bound: 0.0030616
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031346, upper bound: 0.0028871
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031645, upper bound: 0.0030616
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030402, upper bound: 0.0030446
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031598
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030402, upper bound: 0.0030446
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031662
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030475, upper bound: 0.0030422
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031490
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030475, upper bound: 0.0030422
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0030616, upper bound: 0.0031490
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031178, upper bound: 0.0030822
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031250, upper bound: 0.0031754
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031178, upper bound: 0.0030829
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031250, upper bound: 0.0031933
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0030781
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031655, upper bound: 0.0031654
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031597, upper bound: 0.0030781
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.74
Output dim: 1, lower bound: -0.0031655, upper bound: 0.0031654

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0002697, 0.0021118, -0.0022903, 0.0023716
1: 0.9914280, 0.9962574, 0.9914073, 0.9964502, -0.0050222, 0.0048501
2: -0.0080952, -0.0040155, -0.0081408, -0.0039817, -0.0041135, 0.0041253
3: 0.0024647, 0.0053178, 0.0023507, 0.0053300, -0.0028653, 0.0029671
4: 0.0015907, 0.0067984, 0.0015640, 0.0068322, -0.0052415, 0.0052345
5: 0.0029392, 0.0083433, 0.0027232, 0.0083665, -0.0054273, 0.0056201
6: -0.0041356, 0.0011092, -0.0041621, 0.0013089, -0.0054445, 0.0052713
7: -0.0088709, -0.0065593, -0.0088809, -0.0064669, -0.0024040, 0.0023215
8: 0.0035275, 0.0082144, 0.0034831, 0.0082266, -0.0046838, 0.0047202
9: -0.0051029, -0.0018020, -0.0051171, -0.0016701, -0.0034329, 0.0033151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0029003
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0029003
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0002842, 0.0021124, -0.0023726, 0.0023936
1: 0.9914120, 0.9964303, 0.9914058, 0.9964809, -0.0050689, 0.0050244
2: -0.0081361, -0.0039896, -0.0081480, -0.0039794, -0.0041566, 0.0041584
3: 0.0023625, 0.0053272, 0.0023326, 0.0053309, -0.0029684, 0.0029946
4: 0.0015702, 0.0068243, 0.0015621, 0.0068344, -0.0052642, 0.0052621
5: 0.0027457, 0.0083611, 0.0026889, 0.0083681, -0.0056224, 0.0056722
6: -0.0041559, 0.0012881, -0.0041639, 0.0013406, -0.0054965, 0.0054520
7: -0.0088785, -0.0064765, -0.0088815, -0.0064523, -0.0024263, 0.0024050
8: 0.0034935, 0.0082253, 0.0034801, 0.0082285, -0.0047193, 0.0047275
9: -0.0051138, -0.0016838, -0.0051180, -0.0016491, -0.0034647, 0.0034343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030700
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031019
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0003376, 0.0021499, -0.0023284, 0.0024396
1: 0.9914280, 0.9962574, 0.9913265, 0.9965941, -0.0051661, 0.0049309
2: -0.0080952, -0.0040155, -0.0081748, -0.0038501, -0.0042451, 0.0041592
3: 0.0024647, 0.0053178, 0.0022657, 0.0053777, -0.0029130, 0.0030521
4: 0.0015907, 0.0067984, 0.0014599, 0.0069635, -0.0053728, 0.0053385
5: 0.0029392, 0.0083433, 0.0025623, 0.0084568, -0.0055175, 0.0057810
6: -0.0041356, 0.0011092, -0.0042653, 0.0014577, -0.0055933, 0.0053744
7: -0.0088709, -0.0065593, -0.0089195, -0.0063981, -0.0024728, 0.0023602
8: 0.0035275, 0.0082144, 0.0033102, 0.0082357, -0.0046932, 0.0048983
9: -0.0051029, -0.0018020, -0.0051722, -0.0015718, -0.0035312, 0.0033702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0029003
time: 1.17 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0029003
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0003524, 0.0021505, -0.0024107, 0.0024619
1: 0.9914120, 0.9964303, 0.9913251, 0.9966255, -0.0052134, 0.0051051
2: -0.0081361, -0.0039896, -0.0081822, -0.0038478, -0.0042883, 0.0041925
3: 0.0023625, 0.0053272, 0.0022472, 0.0053786, -0.0030161, 0.0030800
4: 0.0015702, 0.0068243, 0.0014581, 0.0069658, -0.0053956, 0.0053662
5: 0.0027457, 0.0083611, 0.0025273, 0.0084584, -0.0057127, 0.0058338
6: -0.0041559, 0.0012881, -0.0042671, 0.0014901, -0.0056460, 0.0055553
7: -0.0088785, -0.0064765, -0.0089202, -0.0063831, -0.0024954, 0.0024436
8: 0.0034935, 0.0082253, 0.0033071, 0.0082376, -0.0047288, 0.0049045
9: -0.0051138, -0.0016838, -0.0051732, -0.0015503, -0.0035634, 0.0034894

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030687
time: 1.20 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030962
time: 1.09 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0002697, 0.0021118, -0.0023554, 0.0024031
1: 0.9913613, 0.9963951, 0.9914073, 0.9964502, -0.0050889, 0.0049878
2: -0.0081277, -0.0039068, -0.0081408, -0.0039817, -0.0041460, 0.0042340
3: 0.0023833, 0.0053572, 0.0023507, 0.0053300, -0.0029468, 0.0030065
4: 0.0015047, 0.0069069, 0.0015640, 0.0068322, -0.0053275, 0.0053430
5: 0.0027850, 0.0084179, 0.0027232, 0.0083665, -0.0055815, 0.0056947
6: -0.0042209, 0.0012518, -0.0041621, 0.0013089, -0.0055297, 0.0054139
7: -0.0089029, -0.0064934, -0.0088809, -0.0064669, -0.0024359, 0.0023875
8: 0.0033846, 0.0082231, 0.0034831, 0.0082266, -0.0048301, 0.0047284
9: -0.0051485, -0.0017078, -0.0051171, -0.0016701, -0.0034784, 0.0034093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0028871
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0028871
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0002842, 0.0021124, -0.0024439, 0.0024314
1: 0.9913321, 0.9965813, 0.9914058, 0.9964809, -0.0051489, 0.0051755
2: -0.0081717, -0.0038591, -0.0081480, -0.0039794, -0.0041923, 0.0042890
3: 0.0022733, 0.0053745, 0.0023326, 0.0053309, -0.0030575, 0.0030419
4: 0.0014670, 0.0069545, 0.0015621, 0.0068344, -0.0053674, 0.0053924
5: 0.0025768, 0.0084506, 0.0026889, 0.0083681, -0.0057913, 0.0057617
6: -0.0042583, 0.0014443, -0.0041639, 0.0013406, -0.0055989, 0.0056082
7: -0.0089169, -0.0064043, -0.0088815, -0.0064523, -0.0024646, 0.0024773
8: 0.0033220, 0.0082349, 0.0034801, 0.0082285, -0.0048959, 0.0047366
9: -0.0051685, -0.0015806, -0.0051180, -0.0016491, -0.0035194, 0.0035375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030400
time: 1.37 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030616
time: 1.20 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0003376, 0.0021499, -0.0023935, 0.0024711
1: 0.9913613, 0.9963951, 0.9913265, 0.9965941, -0.0052328, 0.0050685
2: -0.0081277, -0.0039068, -0.0081748, -0.0038501, -0.0042776, 0.0042680
3: 0.0023833, 0.0053572, 0.0022657, 0.0053777, -0.0029944, 0.0030915
4: 0.0015047, 0.0069069, 0.0014599, 0.0069635, -0.0054588, 0.0054470
5: 0.0027850, 0.0084179, 0.0025623, 0.0084568, -0.0056718, 0.0058556
6: -0.0042209, 0.0012518, -0.0042653, 0.0014577, -0.0056786, 0.0055171
7: -0.0089029, -0.0064934, -0.0089195, -0.0063981, -0.0025048, 0.0024261
8: 0.0033846, 0.0082231, 0.0033102, 0.0082357, -0.0048355, 0.0049021
9: -0.0051485, -0.0017078, -0.0051722, -0.0015718, -0.0035767, 0.0034645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.09 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0028871
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0028871
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0003524, 0.0021505, -0.0024820, 0.0024997
1: 0.9913321, 0.9965813, 0.9913251, 0.9966255, -0.0052934, 0.0052562
2: -0.0081717, -0.0038591, -0.0081822, -0.0038478, -0.0043239, 0.0043231
3: 0.0022733, 0.0053745, 0.0022472, 0.0053786, -0.0031052, 0.0031273
4: 0.0014670, 0.0069545, 0.0014581, 0.0069658, -0.0054988, 0.0054964
5: 0.0025768, 0.0084506, 0.0025273, 0.0084584, -0.0058816, 0.0059234
6: -0.0042583, 0.0014443, -0.0042671, 0.0014901, -0.0057484, 0.0057115
7: -0.0089169, -0.0064043, -0.0089202, -0.0063831, -0.0025338, 0.0025159
8: 0.0033220, 0.0082349, 0.0033071, 0.0082376, -0.0049004, 0.0049099
9: -0.0051685, -0.0015806, -0.0051732, -0.0015503, -0.0036181, 0.0035926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030399
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030616
time: 1.22 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0003860, 0.0020876, -0.0022662, 0.0024880
1: 0.9914280, 0.9962574, 0.9914583, 0.9966967, -0.0052687, 0.0047991
2: -0.0080952, -0.0040155, -0.0081990, -0.0040650, -0.0040302, 0.0041835
3: 0.0024647, 0.0053178, 0.0022051, 0.0052999, -0.0028351, 0.0031127
4: 0.0015907, 0.0067984, 0.0016298, 0.0067490, -0.0051584, 0.0051686
5: 0.0029392, 0.0083433, 0.0024476, 0.0083094, -0.0053701, 0.0058957
6: -0.0041356, 0.0011092, -0.0040968, 0.0015638, -0.0056994, 0.0052059
7: -0.0088709, -0.0065593, -0.0088564, -0.0063490, -0.0025219, 0.0022971
8: 0.0035275, 0.0082144, 0.0035925, 0.0082421, -0.0046885, 0.0046117
9: -0.0051029, -0.0018020, -0.0050822, -0.0015017, -0.0036013, 0.0032802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030474, upper bound: 0.0029003
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030474, upper bound: 0.0029003
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0003998, 0.0020883, -0.0023485, 0.0025093
1: 0.9914120, 0.9964303, 0.9914569, 0.9967259, -0.0053138, 0.0049734
2: -0.0081361, -0.0039896, -0.0082059, -0.0040628, -0.0040733, 0.0042162
3: 0.0023625, 0.0053272, 0.0021879, 0.0053007, -0.0029382, 0.0031393
4: 0.0015702, 0.0068243, 0.0016280, 0.0067513, -0.0051811, 0.0051963
5: 0.0027457, 0.0083611, 0.0024149, 0.0083109, -0.0055652, 0.0059462
6: -0.0041559, 0.0012881, -0.0040985, 0.0015940, -0.0057499, 0.0053867
7: -0.0088785, -0.0064765, -0.0088571, -0.0063350, -0.0025435, 0.0023806
8: 0.0034935, 0.0082253, 0.0035896, 0.0082440, -0.0047242, 0.0046199
9: -0.0051138, -0.0016838, -0.0050831, -0.0014817, -0.0036320, 0.0033994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030478, upper bound: 0.0030701
time: 1.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030478, upper bound: 0.0031019
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0004478, 0.0021180, -0.0022965, 0.0025498
1: 0.9914280, 0.9962574, 0.9913940, 0.9968277, -0.0053996, 0.0048634
2: -0.0080952, -0.0040155, -0.0082299, -0.0039603, -0.0041349, 0.0042144
3: 0.0024647, 0.0053178, 0.0021278, 0.0053378, -0.0028731, 0.0031900
4: 0.0015907, 0.0067984, 0.0015470, 0.0068536, -0.0052629, 0.0052514
5: 0.0029392, 0.0083433, 0.0023011, 0.0083812, -0.0054420, 0.0060422
6: -0.0041356, 0.0011092, -0.0041789, 0.0016992, -0.0058348, 0.0052881
7: -0.0088709, -0.0065593, -0.0088872, -0.0062864, -0.0025846, 0.0023278
8: 0.0035275, 0.0082144, 0.0034549, 0.0082504, -0.0046966, 0.0047522
9: -0.0051029, -0.0018020, -0.0051261, -0.0014122, -0.0036907, 0.0033241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030422, upper bound: 0.0029003
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030422, upper bound: 0.0029003
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0004623, 0.0021186, -0.0023788, 0.0025718
1: 0.9914120, 0.9964303, 0.9913927, 0.9968583, -0.0054463, 0.0050376
2: -0.0081361, -0.0039896, -0.0082372, -0.0039580, -0.0041780, 0.0042475
3: 0.0023625, 0.0053272, 0.0021097, 0.0053386, -0.0029761, 0.0032175
4: 0.0015702, 0.0068243, 0.0015452, 0.0068558, -0.0052856, 0.0052790
5: 0.0027457, 0.0083611, 0.0022667, 0.0083827, -0.0056371, 0.0060943
6: -0.0041559, 0.0012881, -0.0041807, 0.0017310, -0.0058869, 0.0054688
7: -0.0088785, -0.0064765, -0.0088878, -0.0062717, -0.0026069, 0.0024113
8: 0.0034935, 0.0082253, 0.0034520, 0.0082523, -0.0047324, 0.0047597
9: -0.0051138, -0.0016838, -0.0051270, -0.0013912, -0.0037225, 0.0034432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030424, upper bound: 0.0030689
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030424, upper bound: 0.0030962
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0003860, 0.0020876, -0.0023313, 0.0025195
1: 0.9913613, 0.9963951, 0.9914583, 0.9966967, -0.0053354, 0.0049368
2: -0.0081277, -0.0039068, -0.0081990, -0.0040650, -0.0040627, 0.0042922
3: 0.0023833, 0.0053572, 0.0022051, 0.0052999, -0.0029166, 0.0031521
4: 0.0015047, 0.0069069, 0.0016298, 0.0067490, -0.0052443, 0.0052771
5: 0.0027850, 0.0084179, 0.0024476, 0.0083094, -0.0055244, 0.0059704
6: -0.0042209, 0.0012518, -0.0040968, 0.0015638, -0.0057847, 0.0053486
7: -0.0089029, -0.0064934, -0.0088564, -0.0063490, -0.0025539, 0.0023631
8: 0.0033846, 0.0082231, 0.0035925, 0.0082421, -0.0048349, 0.0046199
9: -0.0051485, -0.0017078, -0.0050822, -0.0015017, -0.0036469, 0.0033744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0028871
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0028871
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0003998, 0.0020883, -0.0024198, 0.0025471
1: 0.9913321, 0.9965813, 0.9914569, 0.9967259, -0.0053938, 0.0051244
2: -0.0081717, -0.0038591, -0.0082059, -0.0040628, -0.0041090, 0.0043468
3: 0.0022733, 0.0053745, 0.0021879, 0.0053007, -0.0030274, 0.0031866
4: 0.0014670, 0.0069545, 0.0016280, 0.0067513, -0.0052843, 0.0053265
5: 0.0025768, 0.0084506, 0.0024149, 0.0083109, -0.0057341, 0.0060357
6: -0.0042583, 0.0014443, -0.0040985, 0.0015940, -0.0058523, 0.0055429
7: -0.0089169, -0.0064043, -0.0088571, -0.0063350, -0.0025818, 0.0024528
8: 0.0033220, 0.0082349, 0.0035896, 0.0082440, -0.0049009, 0.0046290
9: -0.0051685, -0.0015806, -0.0050831, -0.0014817, -0.0036868, 0.0035026

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0030402
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0030616
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0004478, 0.0021180, -0.0023616, 0.0025813
1: 0.9913613, 0.9963951, 0.9913940, 0.9968277, -0.0054663, 0.0050011
2: -0.0081277, -0.0039068, -0.0082299, -0.0039603, -0.0041675, 0.0043232
3: 0.0023833, 0.0053572, 0.0021278, 0.0053378, -0.0029545, 0.0032294
4: 0.0015047, 0.0069069, 0.0015470, 0.0068536, -0.0053489, 0.0053599
5: 0.0027850, 0.0084179, 0.0023011, 0.0083812, -0.0055962, 0.0061168
6: -0.0042209, 0.0012518, -0.0041789, 0.0016992, -0.0059201, 0.0054307
7: -0.0089029, -0.0064934, -0.0088872, -0.0062864, -0.0026165, 0.0023938
8: 0.0033846, 0.0082231, 0.0034549, 0.0082504, -0.0048391, 0.0047583
9: -0.0051485, -0.0017078, -0.0051261, -0.0014122, -0.0037363, 0.0034183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0028871
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0028871
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0004623, 0.0021186, -0.0024501, 0.0026096
1: 0.9913321, 0.9965813, 0.9913927, 0.9968583, -0.0055262, 0.0051886
2: -0.0081717, -0.0038591, -0.0082372, -0.0039580, -0.0042137, 0.0043781
3: 0.0022733, 0.0053745, 0.0021097, 0.0053386, -0.0030653, 0.0032648
4: 0.0014670, 0.0069545, 0.0015452, 0.0068558, -0.0053888, 0.0054093
5: 0.0025768, 0.0084506, 0.0022667, 0.0083827, -0.0058060, 0.0061839
6: -0.0042583, 0.0014443, -0.0041807, 0.0017310, -0.0059893, 0.0056250
7: -0.0089169, -0.0064043, -0.0088878, -0.0062717, -0.0026452, 0.0024835
8: 0.0033220, 0.0082349, 0.0034520, 0.0082523, -0.0049041, 0.0047673
9: -0.0051685, -0.0015806, -0.0051270, -0.0013912, -0.0037773, 0.0035464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0030402
time: 1.44 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0030615
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0002697, 0.0021118, -0.0024115, 0.0023554
1: 0.9914624, 0.9965141, 0.9914073, 0.9964502, -0.0049878, 0.0051068
2: -0.0081559, -0.0040717, -0.0081408, -0.0039817, -0.0041741, 0.0040691
3: 0.0023130, 0.0052974, 0.0023507, 0.0053300, -0.0030170, 0.0029468
4: 0.0016351, 0.0067424, 0.0015640, 0.0068322, -0.0051971, 0.0051784
5: 0.0026519, 0.0083048, 0.0027232, 0.0083665, -0.0057146, 0.0055815
6: -0.0040915, 0.0013748, -0.0041621, 0.0013089, -0.0054004, 0.0055369
7: -0.0088545, -0.0064364, -0.0088809, -0.0064669, -0.0023875, 0.0024444
8: 0.0036013, 0.0082306, 0.0034831, 0.0082266, -0.0046102, 0.0047258
9: -0.0050794, -0.0016265, -0.0051171, -0.0016701, -0.0034093, 0.0034906

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030474
time: 1.49 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030478
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0002842, 0.0021124, -0.0024859, 0.0023698
1: 0.9914626, 0.9966701, 0.9914058, 0.9964809, -0.0050183, 0.0052643
2: -0.0081927, -0.0040720, -0.0081480, -0.0039794, -0.0042133, 0.0040760
3: 0.0022208, 0.0052973, 0.0023326, 0.0053309, -0.0031100, 0.0029648
4: 0.0016353, 0.0067420, 0.0015621, 0.0068344, -0.0051991, 0.0051799
5: 0.0024773, 0.0083045, 0.0026889, 0.0083681, -0.0058908, 0.0056156
6: -0.0040913, 0.0015363, -0.0041639, 0.0013406, -0.0054319, 0.0057002
7: -0.0088544, -0.0063617, -0.0088815, -0.0064523, -0.0024021, 0.0025198
8: 0.0036018, 0.0082405, 0.0034801, 0.0082285, -0.0046119, 0.0047319
9: -0.0050792, -0.0015198, -0.0051180, -0.0016491, -0.0034301, 0.0035982

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031393
time: 1.06 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031635
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0003376, 0.0021499, -0.0024496, 0.0024233
1: 0.9914624, 0.9965141, 0.9913265, 0.9965941, -0.0051317, 0.0051876
2: -0.0081559, -0.0040717, -0.0081748, -0.0038501, -0.0043057, 0.0041031
3: 0.0023130, 0.0052974, 0.0022657, 0.0053777, -0.0030647, 0.0030317
4: 0.0016351, 0.0067424, 0.0014599, 0.0069635, -0.0053284, 0.0052825
5: 0.0026519, 0.0083048, 0.0025623, 0.0084568, -0.0058049, 0.0057425
6: -0.0040915, 0.0013748, -0.0042653, 0.0014577, -0.0055492, 0.0056401
7: -0.0088545, -0.0064364, -0.0089195, -0.0063981, -0.0024564, 0.0024831
8: 0.0036013, 0.0082306, 0.0033102, 0.0082357, -0.0046196, 0.0049038
9: -0.0050794, -0.0016265, -0.0051722, -0.0015718, -0.0035076, 0.0035457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030446
time: 1.13 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030446
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0003524, 0.0021505, -0.0025240, 0.0024380
1: 0.9914626, 0.9966701, 0.9913251, 0.9966255, -0.0051628, 0.0053450
2: -0.0081927, -0.0040720, -0.0081822, -0.0038478, -0.0043449, 0.0041101
3: 0.0022208, 0.0052973, 0.0022472, 0.0053786, -0.0031577, 0.0030501
4: 0.0016353, 0.0067420, 0.0014581, 0.0069658, -0.0053305, 0.0052840
5: 0.0024773, 0.0083045, 0.0025273, 0.0084584, -0.0059811, 0.0057773
6: -0.0040913, 0.0015363, -0.0042671, 0.0014901, -0.0055814, 0.0058035
7: -0.0088544, -0.0063617, -0.0089202, -0.0063831, -0.0024713, 0.0025585
8: 0.0036018, 0.0082405, 0.0033071, 0.0082376, -0.0046214, 0.0049089
9: -0.0050792, -0.0015198, -0.0051732, -0.0015503, -0.0035289, 0.0036534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031400
time: 1.18 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031662
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0002697, 0.0021118, -0.0024694, 0.0023812
1: 0.9914078, 0.9966366, 0.9914073, 0.9964502, -0.0050424, 0.0052293
2: -0.0081848, -0.0039827, -0.0081408, -0.0039817, -0.0042031, 0.0041581
3: 0.0022406, 0.0053297, 0.0023507, 0.0053300, -0.0030894, 0.0029790
4: 0.0015647, 0.0068312, 0.0015640, 0.0068322, -0.0052674, 0.0052672
5: 0.0025148, 0.0083658, 0.0027232, 0.0083665, -0.0058517, 0.0056426
6: -0.0041613, 0.0015016, -0.0041621, 0.0013089, -0.0054702, 0.0056637
7: -0.0088806, -0.0063778, -0.0088809, -0.0064669, -0.0024137, 0.0025031
8: 0.0034844, 0.0082383, 0.0034831, 0.0082266, -0.0047291, 0.0047328
9: -0.0051167, -0.0015428, -0.0051171, -0.0016701, -0.0034466, 0.0035743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030422
time: 1.24 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030424
time: 1.17 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0002842, 0.0021124, -0.0025513, 0.0023998
1: 0.9913990, 0.9968084, 0.9914058, 0.9964809, -0.0050820, 0.0054026
2: -0.0082254, -0.0039682, -0.0081480, -0.0039794, -0.0042460, 0.0041798
3: 0.0021390, 0.0053349, 0.0023326, 0.0053309, -0.0031918, 0.0030024
4: 0.0015533, 0.0068456, 0.0015621, 0.0068344, -0.0052811, 0.0052835
5: 0.0023224, 0.0083758, 0.0026889, 0.0083681, -0.0060457, 0.0056868
6: -0.0041727, 0.0016796, -0.0041639, 0.0013406, -0.0055133, 0.0058435
7: -0.0088848, -0.0062955, -0.0088815, -0.0064523, -0.0024326, 0.0025861
8: 0.0034654, 0.0082492, 0.0034801, 0.0082285, -0.0047512, 0.0047396
9: -0.0051227, -0.0014252, -0.0051180, -0.0016491, -0.0034736, 0.0036928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031247
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031490
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0003376, 0.0021499, -0.0025075, 0.0024491
1: 0.9914078, 0.9966366, 0.9913265, 0.9965941, -0.0051864, 0.0053101
2: -0.0081848, -0.0039827, -0.0081748, -0.0038501, -0.0043347, 0.0041921
3: 0.0022406, 0.0053297, 0.0022657, 0.0053777, -0.0031371, 0.0030640
4: 0.0015647, 0.0068312, 0.0014599, 0.0069635, -0.0053987, 0.0053713
5: 0.0025148, 0.0083658, 0.0025623, 0.0084568, -0.0059419, 0.0058035
6: -0.0041613, 0.0015016, -0.0042653, 0.0014577, -0.0056190, 0.0057669
7: -0.0088806, -0.0063778, -0.0089195, -0.0063981, -0.0024825, 0.0025417
8: 0.0034844, 0.0082383, 0.0033102, 0.0082357, -0.0047362, 0.0049070
9: -0.0051167, -0.0015428, -0.0051722, -0.0015718, -0.0035449, 0.0036295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030421
time: 0.95 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030422
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0003524, 0.0021505, -0.0025894, 0.0024681
1: 0.9913990, 0.9968084, 0.9913251, 0.9966255, -0.0052265, 0.0054833
2: -0.0082254, -0.0039682, -0.0081822, -0.0038478, -0.0043776, 0.0042139
3: 0.0021390, 0.0053349, 0.0022472, 0.0053786, -0.0032395, 0.0030877
4: 0.0015533, 0.0068456, 0.0014581, 0.0069658, -0.0054125, 0.0053875
5: 0.0023224, 0.0083758, 0.0025273, 0.0084584, -0.0061360, 0.0058485
6: -0.0041727, 0.0016796, -0.0042671, 0.0014901, -0.0056628, 0.0059467
7: -0.0088848, -0.0062955, -0.0089202, -0.0063831, -0.0025017, 0.0026247
8: 0.0034654, 0.0082492, 0.0033071, 0.0082376, -0.0047579, 0.0049133
9: -0.0051227, -0.0014252, -0.0051732, -0.0015503, -0.0035724, 0.0037480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031247
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031490
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0003860, 0.0020876, -0.0023874, 0.0024717
1: 0.9914624, 0.9965141, 0.9914583, 0.9966967, -0.0052343, 0.0050558
2: -0.0081559, -0.0040717, -0.0081990, -0.0040650, -0.0040908, 0.0041273
3: 0.0023130, 0.0052974, 0.0022051, 0.0052999, -0.0029869, 0.0030923
4: 0.0016351, 0.0067424, 0.0016298, 0.0067490, -0.0051140, 0.0051126
5: 0.0026519, 0.0083048, 0.0024476, 0.0083094, -0.0056574, 0.0058572
6: -0.0040915, 0.0013748, -0.0040968, 0.0015638, -0.0056554, 0.0054716
7: -0.0088545, -0.0064364, -0.0088564, -0.0063490, -0.0025055, 0.0024200
8: 0.0036013, 0.0082306, 0.0035925, 0.0082421, -0.0046141, 0.0046176
9: -0.0050794, -0.0016265, -0.0050822, -0.0015017, -0.0035777, 0.0034557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030512, upper bound: 0.0030823
time: 1.21 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030512, upper bound: 0.0030823
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0003998, 0.0020883, -0.0024618, 0.0024854
1: 0.9914626, 0.9966701, 0.9914569, 0.9967259, -0.0052632, 0.0052132
2: -0.0081927, -0.0040720, -0.0082059, -0.0040628, -0.0041299, 0.0041338
3: 0.0022208, 0.0052973, 0.0021879, 0.0053007, -0.0030799, 0.0031094
4: 0.0016353, 0.0067420, 0.0016280, 0.0067513, -0.0051160, 0.0051140
5: 0.0024773, 0.0083045, 0.0024149, 0.0083109, -0.0058336, 0.0058896
6: -0.0040913, 0.0015363, -0.0040985, 0.0015940, -0.0056853, 0.0056349
7: -0.0088544, -0.0063617, -0.0088571, -0.0063350, -0.0025193, 0.0024954
8: 0.0036018, 0.0082405, 0.0035896, 0.0082440, -0.0046165, 0.0046236
9: -0.0050792, -0.0015198, -0.0050831, -0.0014817, -0.0035975, 0.0035633

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030511, upper bound: 0.0031657
time: 1.23 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030511, upper bound: 0.0031765
time: 1.28 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0004478, 0.0021180, -0.0024177, 0.0025335
1: 0.9914624, 0.9965141, 0.9913940, 0.9968277, -0.0053653, 0.0051201
2: -0.0081559, -0.0040717, -0.0082299, -0.0039603, -0.0041956, 0.0041582
3: 0.0023130, 0.0052974, 0.0021278, 0.0053378, -0.0030248, 0.0031696
4: 0.0016351, 0.0067424, 0.0015470, 0.0068536, -0.0052185, 0.0051954
5: 0.0026519, 0.0083048, 0.0023011, 0.0083812, -0.0057293, 0.0060037
6: -0.0040915, 0.0013748, -0.0041789, 0.0016992, -0.0057908, 0.0055538
7: -0.0088545, -0.0064364, -0.0088872, -0.0062864, -0.0025681, 0.0024507
8: 0.0036013, 0.0082306, 0.0034549, 0.0082504, -0.0046228, 0.0047586
9: -0.0050794, -0.0016265, -0.0051261, -0.0014122, -0.0036672, 0.0034996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0030829
time: 1.42 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0030829
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0004623, 0.0021186, -0.0024921, 0.0025479
1: 0.9914626, 0.9966701, 0.9913927, 0.9968583, -0.0053957, 0.0052775
2: -0.0081927, -0.0040720, -0.0082372, -0.0039580, -0.0042347, 0.0041651
3: 0.0022208, 0.0052973, 0.0021097, 0.0053386, -0.0031178, 0.0031877
4: 0.0016353, 0.0067420, 0.0015452, 0.0068558, -0.0052204, 0.0051968
5: 0.0024773, 0.0083045, 0.0022667, 0.0083827, -0.0059055, 0.0060378
6: -0.0040913, 0.0015363, -0.0041807, 0.0017310, -0.0058223, 0.0057170
7: -0.0088544, -0.0063617, -0.0088878, -0.0062717, -0.0025827, 0.0025261
8: 0.0036018, 0.0082405, 0.0034520, 0.0082523, -0.0046252, 0.0047639
9: -0.0050792, -0.0015198, -0.0051270, -0.0013912, -0.0036880, 0.0036072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0031835
time: 1.36 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0031934
time: 1.38 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0003860, 0.0020876, -0.0024453, 0.0024975
1: 0.9914078, 0.9966366, 0.9914583, 0.9966967, -0.0052890, 0.0051783
2: -0.0081848, -0.0039827, -0.0081990, -0.0040650, -0.0041198, 0.0042163
3: 0.0022406, 0.0053297, 0.0022051, 0.0052999, -0.0030592, 0.0031246
4: 0.0015647, 0.0068312, 0.0016298, 0.0067490, -0.0051843, 0.0052014
5: 0.0025148, 0.0083658, 0.0024476, 0.0083094, -0.0057945, 0.0059183
6: -0.0041613, 0.0015016, -0.0040968, 0.0015638, -0.0057252, 0.0055984
7: -0.0088806, -0.0063778, -0.0088564, -0.0063490, -0.0025316, 0.0024786
8: 0.0034844, 0.0082383, 0.0035925, 0.0082421, -0.0047335, 0.0046252
9: -0.0051167, -0.0015428, -0.0050822, -0.0015017, -0.0036150, 0.0035394

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0030781
time: 1.38 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0030781
time: 1.49 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0003998, 0.0020883, -0.0025271, 0.0025155
1: 0.9913990, 0.9968084, 0.9914569, 0.9967259, -0.0053269, 0.0053515
2: -0.0082254, -0.0039682, -0.0082059, -0.0040628, -0.0041626, 0.0042376
3: 0.0021390, 0.0053349, 0.0021879, 0.0053007, -0.0031617, 0.0031470
4: 0.0015533, 0.0068456, 0.0016280, 0.0067513, -0.0051980, 0.0052176
5: 0.0023224, 0.0083758, 0.0024149, 0.0083109, -0.0059885, 0.0059609
6: -0.0041727, 0.0016796, -0.0040985, 0.0015940, -0.0057666, 0.0057781
7: -0.0088848, -0.0062955, -0.0088571, -0.0063350, -0.0025498, 0.0025616
8: 0.0034654, 0.0082492, 0.0035896, 0.0082440, -0.0047562, 0.0046319
9: -0.0051227, -0.0014252, -0.0050831, -0.0014817, -0.0036410, 0.0036579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031559
time: 1.31 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031654
time: 1.18 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0004478, 0.0021180, -0.0024756, 0.0025593
1: 0.9914078, 0.9966366, 0.9913940, 0.9968277, -0.0054199, 0.0052426
2: -0.0081848, -0.0039827, -0.0082299, -0.0039603, -0.0042245, 0.0042472
3: 0.0022406, 0.0053297, 0.0021278, 0.0053378, -0.0030972, 0.0032019
4: 0.0015647, 0.0068312, 0.0015470, 0.0068536, -0.0052888, 0.0052842
5: 0.0025148, 0.0083658, 0.0023011, 0.0083812, -0.0058664, 0.0060647
6: -0.0041613, 0.0015016, -0.0041789, 0.0016992, -0.0058606, 0.0056805
7: -0.0088806, -0.0063778, -0.0088872, -0.0062864, -0.0025942, 0.0025094
8: 0.0034844, 0.0082383, 0.0034549, 0.0082504, -0.0047380, 0.0047619
9: -0.0051167, -0.0015428, -0.0051261, -0.0014122, -0.0037045, 0.0035833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0030781
time: 1.35 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0030781
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0004623, 0.0021186, -0.0025575, 0.0025780
1: 0.9913990, 0.9968084, 0.9913927, 0.9968583, -0.0054593, 0.0054157
2: -0.0082254, -0.0039682, -0.0082372, -0.0039580, -0.0042674, 0.0042689
3: 0.0021390, 0.0053349, 0.0021097, 0.0053386, -0.0031996, 0.0032253
4: 0.0015533, 0.0068456, 0.0015452, 0.0068558, -0.0053025, 0.0053004
5: 0.0023224, 0.0083758, 0.0022667, 0.0083827, -0.0060604, 0.0061090
6: -0.0041727, 0.0016796, -0.0041807, 0.0017310, -0.0059037, 0.0058602
7: -0.0088848, -0.0062955, -0.0088878, -0.0062717, -0.0026132, 0.0025923
8: 0.0034654, 0.0082492, 0.0034520, 0.0082523, -0.0047601, 0.0047686
9: -0.0051227, -0.0014252, -0.0051270, -0.0013912, -0.0037315, 0.0037018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 120

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0031559
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0031654
time: 1.24 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.65 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0029003
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0029003
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030700
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031019
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0029003
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0029003
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030687
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030962
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0028871
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0028871
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030400
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030616
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0028871
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0028871
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030399
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030616
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030474, upper bound: 0.0029003
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030474, upper bound: 0.0029003
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030478, upper bound: 0.0030701
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030478, upper bound: 0.0031019
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030422, upper bound: 0.0029003
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030422, upper bound: 0.0029003
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030424, upper bound: 0.0030689
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030424, upper bound: 0.0030962
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0028871
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0028871
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0030402
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030446, upper bound: 0.0030616
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0028871
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0028871
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0030402
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030425, upper bound: 0.0030615
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030474
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030478
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031393
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031635
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030446
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030446
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031400
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031662
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030422
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0030424
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031247
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0029003, upper bound: 0.0031490
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030421
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0030422
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031247
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0028871, upper bound: 0.0031490
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030512, upper bound: 0.0030823
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030512, upper bound: 0.0030823
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030511, upper bound: 0.0031657
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030511, upper bound: 0.0031765
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0030829
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0030829
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0031835
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030460, upper bound: 0.0031934
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0030781
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0030781
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031559
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030595, upper bound: 0.0031654
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0030781
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0030781
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0031559
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.65
Output dim: 1, lower bound: -0.0030592, upper bound: 0.0031654

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0001785, 0.0021020, -0.0022805, 0.0022805
1: 0.9914280, 0.9962574, 0.9914280, 0.9962574, -0.0048293, 0.0048293
2: -0.0080952, -0.0040155, -0.0080952, -0.0040155, -0.0040797, 0.0040797
3: 0.0024647, 0.0053178, 0.0024647, 0.0053178, -0.0028531, 0.0028531
4: 0.0015907, 0.0067984, 0.0015907, 0.0067984, -0.0052077, 0.0052077
5: 0.0029392, 0.0083433, 0.0029392, 0.0083433, -0.0054041, 0.0054041
6: -0.0041356, 0.0011092, -0.0041356, 0.0011092, -0.0052447, 0.0052447
7: -0.0088709, -0.0065593, -0.0088709, -0.0065593, -0.0023116, 0.0023116
8: 0.0035275, 0.0082144, 0.0035275, 0.0082144, -0.0046756, 0.0046756
9: -0.0051029, -0.0018020, -0.0051029, -0.0018020, -0.0033009, 0.0033009

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028611
time: 1.13 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028047
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0002602, 0.0021095, -0.0022880, 0.0023622
1: 0.9914280, 0.9962574, 0.9914120, 0.9964303, -0.0050023, 0.0048453
2: -0.0080952, -0.0040155, -0.0081361, -0.0039896, -0.0041056, 0.0041205
3: 0.0024647, 0.0053178, 0.0023625, 0.0053272, -0.0028624, 0.0029553
4: 0.0015907, 0.0067984, 0.0015702, 0.0068243, -0.0052336, 0.0052282
5: 0.0029392, 0.0083433, 0.0027457, 0.0083611, -0.0054218, 0.0055976
6: -0.0041356, 0.0011092, -0.0041559, 0.0012881, -0.0054237, 0.0052650
7: -0.0088709, -0.0065593, -0.0088785, -0.0064765, -0.0023944, 0.0023192
8: 0.0035275, 0.0082144, 0.0034935, 0.0082253, -0.0046793, 0.0047098
9: -0.0051029, -0.0018020, -0.0051138, -0.0016838, -0.0034191, 0.0033118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028611
time: 1.25 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028047
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0001785, 0.0021020, -0.0023622, 0.0022880
1: 0.9914120, 0.9964303, 0.9914280, 0.9962574, -0.0048453, 0.0050023
2: -0.0081361, -0.0039896, -0.0080952, -0.0040155, -0.0041205, 0.0041056
3: 0.0023625, 0.0053272, 0.0024647, 0.0053178, -0.0029553, 0.0028624
4: 0.0015702, 0.0068243, 0.0015907, 0.0067984, -0.0052282, 0.0052336
5: 0.0027457, 0.0083611, 0.0029392, 0.0083433, -0.0055976, 0.0054218
6: -0.0041559, 0.0012881, -0.0041356, 0.0011092, -0.0052650, 0.0054237
7: -0.0088785, -0.0064765, -0.0088709, -0.0065593, -0.0023192, 0.0023944
8: 0.0034935, 0.0082253, 0.0035275, 0.0082144, -0.0047098, 0.0046793
9: -0.0051138, -0.0016838, -0.0051029, -0.0018020, -0.0033118, 0.0034191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030008
time: 1.09 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029894
time: 1.10 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0002602, 0.0021095, -0.0023697, 0.0023697
1: 0.9914120, 0.9964303, 0.9914120, 0.9964303, -0.0050182, 0.0050182
2: -0.0081361, -0.0039896, -0.0081361, -0.0039896, -0.0041464, 0.0041464
3: 0.0023625, 0.0053272, 0.0023625, 0.0053272, -0.0029647, 0.0029647
4: 0.0015702, 0.0068243, 0.0015702, 0.0068243, -0.0052540, 0.0052540
5: 0.0027457, 0.0083611, 0.0027457, 0.0083611, -0.0056154, 0.0056154
6: -0.0041559, 0.0012881, -0.0041559, 0.0012881, -0.0054440, 0.0054440
7: -0.0088785, -0.0064765, -0.0088785, -0.0064765, -0.0024020, 0.0024020
8: 0.0034935, 0.0082253, 0.0034935, 0.0082253, -0.0047141, 0.0047141
9: -0.0051138, -0.0016838, -0.0051138, -0.0016838, -0.0034300, 0.0034300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030387
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030268
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0002436, 0.0021335, -0.0023120, 0.0023456
1: 0.9914280, 0.9962574, 0.9913613, 0.9963951, -0.0049670, 0.0048960
2: -0.0080952, -0.0040155, -0.0081277, -0.0039068, -0.0041884, 0.0041122
3: 0.0024647, 0.0053178, 0.0023833, 0.0053572, -0.0028925, 0.0029345
4: 0.0015907, 0.0067984, 0.0015047, 0.0069069, -0.0053163, 0.0052937
5: 0.0029392, 0.0083433, 0.0027850, 0.0084179, -0.0054787, 0.0055583
6: -0.0041356, 0.0011092, -0.0042209, 0.0012518, -0.0053874, 0.0053300
7: -0.0088709, -0.0065593, -0.0089029, -0.0064934, -0.0023776, 0.0023435
8: 0.0035275, 0.0082144, 0.0033846, 0.0082231, -0.0046838, 0.0048219
9: -0.0051029, -0.0018020, -0.0051485, -0.0017078, -0.0033952, 0.0033465

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028568
time: 1.35 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028047
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0003315, 0.0021473, -0.0023258, 0.0024335
1: 0.9914280, 0.9962574, 0.9913321, 0.9965813, -0.0051533, 0.0049253
2: -0.0080952, -0.0040155, -0.0081717, -0.0038591, -0.0042361, 0.0041562
3: 0.0024647, 0.0053178, 0.0022733, 0.0053745, -0.0029097, 0.0030445
4: 0.0015907, 0.0067984, 0.0014670, 0.0069545, -0.0053638, 0.0053314
5: 0.0029392, 0.0083433, 0.0025768, 0.0084506, -0.0055114, 0.0057665
6: -0.0041356, 0.0011092, -0.0042583, 0.0014443, -0.0055799, 0.0053674
7: -0.0088709, -0.0065593, -0.0089169, -0.0064043, -0.0024667, 0.0023575
8: 0.0035275, 0.0082144, 0.0033220, 0.0082349, -0.0046890, 0.0048865
9: -0.0051029, -0.0018020, -0.0051685, -0.0015806, -0.0035223, 0.0033665

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028568
time: 1.22 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028047
time: 0.96 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0002436, 0.0021335, -0.0023937, 0.0023531
1: 0.9914120, 0.9964303, 0.9913613, 0.9963951, -0.0049830, 0.0050690
2: -0.0081361, -0.0039896, -0.0081277, -0.0039068, -0.0042293, 0.0041381
3: 0.0023625, 0.0053272, 0.0023833, 0.0053572, -0.0029947, 0.0029439
4: 0.0015702, 0.0068243, 0.0015047, 0.0069069, -0.0053367, 0.0053196
5: 0.0027457, 0.0083611, 0.0027850, 0.0084179, -0.0056723, 0.0055761
6: -0.0041559, 0.0012881, -0.0042209, 0.0012518, -0.0054077, 0.0055090
7: -0.0088785, -0.0064765, -0.0089029, -0.0064934, -0.0023852, 0.0024263
8: 0.0034935, 0.0082253, 0.0033846, 0.0082231, -0.0047180, 0.0048256
9: -0.0051138, -0.0016838, -0.0051485, -0.0017078, -0.0034060, 0.0034647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029960
time: 1.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029853
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0003315, 0.0021473, -0.0024075, 0.0024410
1: 0.9914120, 0.9964303, 0.9913321, 0.9965813, -0.0051693, 0.0050982
2: -0.0081361, -0.0039896, -0.0081717, -0.0038591, -0.0042770, 0.0041821
3: 0.0023625, 0.0053272, 0.0022733, 0.0053745, -0.0030120, 0.0030539
4: 0.0015702, 0.0068243, 0.0014670, 0.0069545, -0.0053843, 0.0053572
5: 0.0027457, 0.0083611, 0.0025768, 0.0084506, -0.0057050, 0.0057843
6: -0.0041559, 0.0012881, -0.0042583, 0.0014443, -0.0056002, 0.0055464
7: -0.0088785, -0.0064765, -0.0089169, -0.0064043, -0.0024743, 0.0024403
8: 0.0034935, 0.0082253, 0.0033220, 0.0082349, -0.0047232, 0.0048896
9: -0.0051138, -0.0016838, -0.0051685, -0.0015806, -0.0035332, 0.0034847

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.11 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030274
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030176
time: 1.26 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0001785, 0.0021020, -0.0023456, 0.0023120
1: 0.9913613, 0.9963951, 0.9914280, 0.9962574, -0.0048960, 0.0049670
2: -0.0081277, -0.0039068, -0.0080952, -0.0040155, -0.0041122, 0.0041884
3: 0.0023833, 0.0053572, 0.0024647, 0.0053178, -0.0029345, 0.0028925
4: 0.0015047, 0.0069069, 0.0015907, 0.0067984, -0.0052937, 0.0053163
5: 0.0027850, 0.0084179, 0.0029392, 0.0083433, -0.0055583, 0.0054787
6: -0.0042209, 0.0012518, -0.0041356, 0.0011092, -0.0053300, 0.0053874
7: -0.0089029, -0.0064934, -0.0088709, -0.0065593, -0.0023435, 0.0023776
8: 0.0033846, 0.0082231, 0.0035275, 0.0082144, -0.0048219, 0.0046838
9: -0.0051485, -0.0017078, -0.0051029, -0.0018020, -0.0033465, 0.0033952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028501
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0027917
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0002602, 0.0021095, -0.0023531, 0.0023937
1: 0.9913613, 0.9963951, 0.9914120, 0.9964303, -0.0050690, 0.0049830
2: -0.0081277, -0.0039068, -0.0081361, -0.0039896, -0.0041381, 0.0042293
3: 0.0023833, 0.0053572, 0.0023625, 0.0053272, -0.0029439, 0.0029947
4: 0.0015047, 0.0069069, 0.0015702, 0.0068243, -0.0053196, 0.0053367
5: 0.0027850, 0.0084179, 0.0027457, 0.0083611, -0.0055761, 0.0056723
6: -0.0042209, 0.0012518, -0.0041559, 0.0012881, -0.0055090, 0.0054077
7: -0.0089029, -0.0064934, -0.0088785, -0.0064765, -0.0024263, 0.0023852
8: 0.0033846, 0.0082231, 0.0034935, 0.0082253, -0.0048256, 0.0047180
9: -0.0051485, -0.0017078, -0.0051138, -0.0016838, -0.0034647, 0.0034060

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028501
time: 1.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0027917
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0001785, 0.0021020, -0.0024335, 0.0023258
1: 0.9913321, 0.9965813, 0.9914280, 0.9962574, -0.0049253, 0.0051533
2: -0.0081717, -0.0038591, -0.0080952, -0.0040155, -0.0041562, 0.0042361
3: 0.0022733, 0.0053745, 0.0024647, 0.0053178, -0.0030445, 0.0029097
4: 0.0014670, 0.0069545, 0.0015907, 0.0067984, -0.0053314, 0.0053638
5: 0.0025768, 0.0084506, 0.0029392, 0.0083433, -0.0057665, 0.0055114
6: -0.0042583, 0.0014443, -0.0041356, 0.0011092, -0.0053674, 0.0055799
7: -0.0089169, -0.0064043, -0.0088709, -0.0065593, -0.0023575, 0.0024667
8: 0.0033220, 0.0082349, 0.0035275, 0.0082144, -0.0048865, 0.0046890
9: -0.0051685, -0.0015806, -0.0051029, -0.0018020, -0.0033665, 0.0035223

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.12 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029736
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029531
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0002602, 0.0021095, -0.0024410, 0.0024075
1: 0.9913321, 0.9965813, 0.9914120, 0.9964303, -0.0050982, 0.0051693
2: -0.0081717, -0.0038591, -0.0081361, -0.0039896, -0.0041821, 0.0042770
3: 0.0022733, 0.0053745, 0.0023625, 0.0053272, -0.0030539, 0.0030120
4: 0.0014670, 0.0069545, 0.0015702, 0.0068243, -0.0053572, 0.0053843
5: 0.0025768, 0.0084506, 0.0027457, 0.0083611, -0.0057843, 0.0057050
6: -0.0042583, 0.0014443, -0.0041559, 0.0012881, -0.0055464, 0.0056002
7: -0.0089169, -0.0064043, -0.0088785, -0.0064765, -0.0024403, 0.0024743
8: 0.0033220, 0.0082349, 0.0034935, 0.0082253, -0.0048896, 0.0047232
9: -0.0051685, -0.0015806, -0.0051138, -0.0016838, -0.0034847, 0.0035332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030013
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029771
time: 1.15 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0002436, 0.0021335, -0.0023771, 0.0023771
1: 0.9913613, 0.9963951, 0.9913613, 0.9963951, -0.0050337, 0.0050337
2: -0.0081277, -0.0039068, -0.0081277, -0.0039068, -0.0042210, 0.0042210
3: 0.0023833, 0.0053572, 0.0023833, 0.0053572, -0.0029739, 0.0029739
4: 0.0015047, 0.0069069, 0.0015047, 0.0069069, -0.0054022, 0.0054022
5: 0.0027850, 0.0084179, 0.0027850, 0.0084179, -0.0056329, 0.0056329
6: -0.0042209, 0.0012518, -0.0042209, 0.0012518, -0.0054726, 0.0054726
7: -0.0089029, -0.0064934, -0.0089029, -0.0064934, -0.0024095, 0.0024095
8: 0.0033846, 0.0082231, 0.0033846, 0.0082231, -0.0048266, 0.0048266
9: -0.0051485, -0.0017078, -0.0051485, -0.0017078, -0.0034407, 0.0034407

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028497
time: 1.41 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0027917
time: 1.18 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0003315, 0.0021473, -0.0023909, 0.0024650
1: 0.9913613, 0.9963951, 0.9913321, 0.9965813, -0.0052200, 0.0050630
2: -0.0081277, -0.0039068, -0.0081717, -0.0038591, -0.0042687, 0.0042650
3: 0.0023833, 0.0053572, 0.0022733, 0.0053745, -0.0029912, 0.0030839
4: 0.0015047, 0.0069069, 0.0014670, 0.0069545, -0.0054498, 0.0054399
5: 0.0027850, 0.0084179, 0.0025768, 0.0084506, -0.0056656, 0.0058412
6: -0.0042209, 0.0012518, -0.0042583, 0.0014443, -0.0056652, 0.0055100
7: -0.0089029, -0.0064934, -0.0089169, -0.0064043, -0.0024986, 0.0024235
8: 0.0033846, 0.0082231, 0.0033220, 0.0082349, -0.0048306, 0.0048904
9: -0.0051485, -0.0017078, -0.0051685, -0.0015806, -0.0035679, 0.0034607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028497
time: 1.16 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0027917
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0002436, 0.0021335, -0.0024650, 0.0023909
1: 0.9913321, 0.9965813, 0.9913613, 0.9963951, -0.0050630, 0.0052200
2: -0.0081717, -0.0038591, -0.0081277, -0.0039068, -0.0042650, 0.0042687
3: 0.0022733, 0.0053745, 0.0023833, 0.0053572, -0.0030839, 0.0029912
4: 0.0014670, 0.0069545, 0.0015047, 0.0069069, -0.0054399, 0.0054498
5: 0.0025768, 0.0084506, 0.0027850, 0.0084179, -0.0058412, 0.0056656
6: -0.0042583, 0.0014443, -0.0042209, 0.0012518, -0.0055100, 0.0056652
7: -0.0089169, -0.0064043, -0.0089029, -0.0064934, -0.0024235, 0.0024986
8: 0.0033220, 0.0082349, 0.0033846, 0.0082231, -0.0048904, 0.0048306
9: -0.0051685, -0.0015806, -0.0051485, -0.0017078, -0.0034607, 0.0035679

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029736
time: 1.23 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029531
time: 1.13 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0003315, 0.0021473, -0.0024788, 0.0024788
1: 0.9913321, 0.9965813, 0.9913321, 0.9965813, -0.0052493, 0.0052493
2: -0.0081717, -0.0038591, -0.0081717, -0.0038591, -0.0043126, 0.0043126
3: 0.0022733, 0.0053745, 0.0022733, 0.0053745, -0.0031011, 0.0031011
4: 0.0014670, 0.0069545, 0.0014670, 0.0069545, -0.0054875, 0.0054875
5: 0.0025768, 0.0084506, 0.0025768, 0.0084506, -0.0058739, 0.0058739
6: -0.0042583, 0.0014443, -0.0042583, 0.0014443, -0.0057026, 0.0057026
7: -0.0089169, -0.0064043, -0.0089169, -0.0064043, -0.0025126, 0.0025126
8: 0.0033220, 0.0082349, 0.0033220, 0.0082349, -0.0048951, 0.0048951
9: -0.0051685, -0.0015806, -0.0051685, -0.0015806, -0.0035879, 0.0035879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030013
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029770
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0002998, 0.0020857, -0.0022642, 0.0024017
1: 0.9914280, 0.9962574, 0.9914624, 0.9965141, -0.0050861, 0.0047950
2: -0.0080952, -0.0040155, -0.0081559, -0.0040717, -0.0040235, 0.0041403
3: 0.0024647, 0.0053178, 0.0023130, 0.0052974, -0.0028327, 0.0030048
4: 0.0015907, 0.0067984, 0.0016351, 0.0067424, -0.0051517, 0.0051634
5: 0.0029392, 0.0083433, 0.0026519, 0.0083048, -0.0053655, 0.0056914
6: -0.0041356, 0.0011092, -0.0040915, 0.0013748, -0.0055104, 0.0052007
7: -0.0088709, -0.0065593, -0.0088545, -0.0064364, -0.0024345, 0.0022951
8: 0.0035275, 0.0082144, 0.0036013, 0.0082306, -0.0046811, 0.0046020
9: -0.0051029, -0.0018020, -0.0050794, -0.0016265, -0.0034764, 0.0032774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030123, upper bound: 0.0028655
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029775, upper bound: 0.0028047
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0003735, 0.0020856, -0.0022641, 0.0024754
1: 0.9914280, 0.9962574, 0.9914626, 0.9966701, -0.0052421, 0.0047947
2: -0.0080952, -0.0040155, -0.0081927, -0.0040720, -0.0040231, 0.0041772
3: 0.0024647, 0.0053178, 0.0022208, 0.0052973, -0.0028326, 0.0030970
4: 0.0015907, 0.0067984, 0.0016353, 0.0067420, -0.0051514, 0.0051631
5: 0.0029392, 0.0083433, 0.0024773, 0.0083045, -0.0053653, 0.0058660
6: -0.0041356, 0.0011092, -0.0040913, 0.0015363, -0.0056719, 0.0052004
7: -0.0088709, -0.0065593, -0.0088544, -0.0063617, -0.0025092, 0.0022950
8: 0.0035275, 0.0082144, 0.0036018, 0.0082405, -0.0046845, 0.0046025
9: -0.0051029, -0.0018020, -0.0050792, -0.0015198, -0.0035831, 0.0032772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030123, upper bound: 0.0028655
time: 1.14 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029775, upper bound: 0.0028047
time: 1.29 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0002998, 0.0020857, -0.0023459, 0.0024092
1: 0.9914120, 0.9964303, 0.9914624, 0.9965141, -0.0051020, 0.0049679
2: -0.0081361, -0.0039896, -0.0081559, -0.0040717, -0.0040644, 0.0041662
3: 0.0023625, 0.0053272, 0.0023130, 0.0052974, -0.0029349, 0.0030142
4: 0.0015702, 0.0068243, 0.0016351, 0.0067424, -0.0051722, 0.0051892
5: 0.0027457, 0.0083611, 0.0026519, 0.0083048, -0.0055591, 0.0057092
6: -0.0041559, 0.0012881, -0.0040915, 0.0013748, -0.0055307, 0.0053797
7: -0.0088785, -0.0064765, -0.0088545, -0.0064364, -0.0024421, 0.0023779
8: 0.0034935, 0.0082253, 0.0036013, 0.0082306, -0.0047154, 0.0046057
9: -0.0051138, -0.0016838, -0.0050794, -0.0016265, -0.0034873, 0.0033956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030128, upper bound: 0.0030055
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029779, upper bound: 0.0029938
time: 1.32 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0003735, 0.0020856, -0.0023458, 0.0024829
1: 0.9914120, 0.9964303, 0.9914626, 0.9966701, -0.0052581, 0.0049676
2: -0.0081361, -0.0039896, -0.0081927, -0.0040720, -0.0040640, 0.0042031
3: 0.0023625, 0.0053272, 0.0022208, 0.0052973, -0.0029348, 0.0031063
4: 0.0015702, 0.0068243, 0.0016353, 0.0067420, -0.0051718, 0.0051889
5: 0.0027457, 0.0083611, 0.0024773, 0.0083045, -0.0055589, 0.0058838
6: -0.0041559, 0.0012881, -0.0040913, 0.0015363, -0.0056922, 0.0053794
7: -0.0088785, -0.0064765, -0.0088544, -0.0063617, -0.0025168, 0.0023778
8: 0.0034935, 0.0082253, 0.0036018, 0.0082405, -0.0047186, 0.0046077
9: -0.0051138, -0.0016838, -0.0050792, -0.0015198, -0.0035940, 0.0033955

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.13 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030128, upper bound: 0.0030393
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029779, upper bound: 0.0030275
time: 1.26 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0003576, 0.0021115, -0.0022900, 0.0024596
1: 0.9914280, 0.9962574, 0.9914078, 0.9966366, -0.0052086, 0.0048496
2: -0.0080952, -0.0040155, -0.0081848, -0.0039827, -0.0041125, 0.0041693
3: 0.0024647, 0.0053178, 0.0022406, 0.0053297, -0.0028650, 0.0030771
4: 0.0015907, 0.0067984, 0.0015647, 0.0068312, -0.0052405, 0.0052337
5: 0.0029392, 0.0083433, 0.0025148, 0.0083658, -0.0054266, 0.0058285
6: -0.0041356, 0.0011092, -0.0041613, 0.0015016, -0.0056372, 0.0052705
7: -0.0088709, -0.0065593, -0.0088806, -0.0063778, -0.0024931, 0.0023212
8: 0.0035275, 0.0082144, 0.0034844, 0.0082383, -0.0046882, 0.0047209
9: -0.0051029, -0.0018020, -0.0051167, -0.0015428, -0.0035602, 0.0033147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030037, upper bound: 0.0028632
time: 1.26 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0028047
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0001785, 0.0021020, -0.0004388, 0.0021157, -0.0022942, 0.0025408
1: 0.9914280, 0.9962574, 0.9913990, 0.9968084, -0.0053804, 0.0048584
2: -0.0080952, -0.0040155, -0.0082254, -0.0039682, -0.0041269, 0.0042099
3: 0.0024647, 0.0053178, 0.0021390, 0.0053349, -0.0028702, 0.0031788
4: 0.0015907, 0.0067984, 0.0015533, 0.0068456, -0.0052549, 0.0052451
5: 0.0029392, 0.0083433, 0.0023224, 0.0083758, -0.0054365, 0.0060209
6: -0.0041356, 0.0011092, -0.0041727, 0.0016796, -0.0058152, 0.0052818
7: -0.0088709, -0.0065593, -0.0088848, -0.0062955, -0.0025755, 0.0023255
8: 0.0035275, 0.0082144, 0.0034654, 0.0082492, -0.0046931, 0.0047417
9: -0.0051029, -0.0018020, -0.0051227, -0.0014252, -0.0036777, 0.0033207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030037, upper bound: 0.0028632
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0028047
time: 1.41 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0003576, 0.0021115, -0.0023717, 0.0024671
1: 0.9914120, 0.9964303, 0.9914078, 0.9966366, -0.0052245, 0.0050225
2: -0.0081361, -0.0039896, -0.0081848, -0.0039827, -0.0041534, 0.0041951
3: 0.0023625, 0.0053272, 0.0022406, 0.0053297, -0.0029672, 0.0030865
4: 0.0015702, 0.0068243, 0.0015647, 0.0068312, -0.0052610, 0.0052595
5: 0.0027457, 0.0083611, 0.0025148, 0.0083658, -0.0056202, 0.0058462
6: -0.0041559, 0.0012881, -0.0041613, 0.0015016, -0.0056575, 0.0054495
7: -0.0088785, -0.0064765, -0.0088806, -0.0063778, -0.0025007, 0.0024041
8: 0.0034935, 0.0082253, 0.0034844, 0.0082383, -0.0047224, 0.0047246
9: -0.0051138, -0.0016838, -0.0051167, -0.0015428, -0.0035710, 0.0034329

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030041, upper bound: 0.0030041
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029689, upper bound: 0.0029929
time: 1.19 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0002602, 0.0021095, -0.0004388, 0.0021157, -0.0023759, 0.0025483
1: 0.9914120, 0.9964303, 0.9913990, 0.9968084, -0.0053964, 0.0050313
2: -0.0081361, -0.0039896, -0.0082254, -0.0039682, -0.0041678, 0.0042358
3: 0.0023625, 0.0053272, 0.0021390, 0.0053349, -0.0029724, 0.0031881
4: 0.0015702, 0.0068243, 0.0015533, 0.0068456, -0.0052754, 0.0052710
5: 0.0027457, 0.0083611, 0.0023224, 0.0083758, -0.0056301, 0.0060387
6: -0.0041559, 0.0012881, -0.0041727, 0.0016796, -0.0058355, 0.0054608
7: -0.0088785, -0.0064765, -0.0088848, -0.0062955, -0.0025831, 0.0024083
8: 0.0034935, 0.0082253, 0.0034654, 0.0082492, -0.0047262, 0.0047462
9: -0.0051138, -0.0016838, -0.0051227, -0.0014252, -0.0036886, 0.0034390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030041, upper bound: 0.0030298
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029689, upper bound: 0.0030199
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0002998, 0.0020857, -0.0023293, 0.0024332
1: 0.9913613, 0.9963951, 0.9914624, 0.9965141, -0.0051528, 0.0049326
2: -0.0081277, -0.0039068, -0.0081559, -0.0040717, -0.0040560, 0.0042491
3: 0.0023833, 0.0053572, 0.0023130, 0.0052974, -0.0029142, 0.0030442
4: 0.0015047, 0.0069069, 0.0016351, 0.0067424, -0.0052377, 0.0052719
5: 0.0027850, 0.0084179, 0.0026519, 0.0083048, -0.0055198, 0.0057660
6: -0.0042209, 0.0012518, -0.0040915, 0.0013748, -0.0055957, 0.0053433
7: -0.0089029, -0.0064934, -0.0088545, -0.0064364, -0.0024664, 0.0023611
8: 0.0033846, 0.0082231, 0.0036013, 0.0082306, -0.0048275, 0.0046102
9: -0.0051485, -0.0017078, -0.0050794, -0.0016265, -0.0035220, 0.0033716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0028543
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029702, upper bound: 0.0027917
time: 1.06 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0003735, 0.0020856, -0.0023292, 0.0025069
1: 0.9913613, 0.9963951, 0.9914626, 0.9966701, -0.0053088, 0.0049324
2: -0.0081277, -0.0039068, -0.0081927, -0.0040720, -0.0040557, 0.0042860
3: 0.0023833, 0.0053572, 0.0022208, 0.0052973, -0.0029141, 0.0031364
4: 0.0015047, 0.0069069, 0.0016353, 0.0067420, -0.0052373, 0.0052716
5: 0.0027850, 0.0084179, 0.0024773, 0.0083045, -0.0055195, 0.0059407
6: -0.0042209, 0.0012518, -0.0040913, 0.0015363, -0.0057572, 0.0053430
7: -0.0089029, -0.0064934, -0.0088544, -0.0063617, -0.0025412, 0.0023610
8: 0.0033846, 0.0082231, 0.0036018, 0.0082405, -0.0048309, 0.0046106
9: -0.0051485, -0.0017078, -0.0050792, -0.0015198, -0.0036287, 0.0033715

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0028543
time: 1.21 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029702, upper bound: 0.0027917
time: 1.17 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0002998, 0.0020857, -0.0024172, 0.0024470
1: 0.9913321, 0.9965813, 0.9914624, 0.9965141, -0.0051820, 0.0051189
2: -0.0081717, -0.0038591, -0.0081559, -0.0040717, -0.0041000, 0.0042968
3: 0.0022733, 0.0053745, 0.0023130, 0.0052974, -0.0030241, 0.0030614
4: 0.0014670, 0.0069545, 0.0016351, 0.0067424, -0.0052754, 0.0053195
5: 0.0025768, 0.0084506, 0.0026519, 0.0083048, -0.0057280, 0.0057987
6: -0.0042583, 0.0014443, -0.0040915, 0.0013748, -0.0056331, 0.0055359
7: -0.0089169, -0.0064043, -0.0088545, -0.0064364, -0.0024804, 0.0024502
8: 0.0033220, 0.0082349, 0.0036013, 0.0082306, -0.0048921, 0.0046154
9: -0.0051685, -0.0015806, -0.0050794, -0.0016265, -0.0035420, 0.0034988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0029790
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029704, upper bound: 0.0029568
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0003735, 0.0020856, -0.0024171, 0.0025207
1: 0.9913321, 0.9965813, 0.9914626, 0.9966701, -0.0053381, 0.0051187
2: -0.0081717, -0.0038591, -0.0081927, -0.0040720, -0.0040997, 0.0043336
3: 0.0022733, 0.0053745, 0.0022208, 0.0052973, -0.0030240, 0.0031536
4: 0.0014670, 0.0069545, 0.0016353, 0.0067420, -0.0052750, 0.0053192
5: 0.0025768, 0.0084506, 0.0024773, 0.0083045, -0.0057278, 0.0059734
6: -0.0042583, 0.0014443, -0.0040913, 0.0015363, -0.0057946, 0.0055356
7: -0.0089169, -0.0064043, -0.0088544, -0.0063617, -0.0025552, 0.0024501
8: 0.0033220, 0.0082349, 0.0036018, 0.0082405, -0.0048941, 0.0046168
9: -0.0051685, -0.0015806, -0.0050792, -0.0015198, -0.0036487, 0.0034987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.14 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0030030
time: 1.18 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029704, upper bound: 0.0029777
time: 1.50 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0003576, 0.0021115, -0.0023551, 0.0024911
1: 0.9913613, 0.9963951, 0.9914078, 0.9966366, -0.0052752, 0.0049873
2: -0.0081277, -0.0039068, -0.0081848, -0.0039827, -0.0041450, 0.0042780
3: 0.0023833, 0.0053572, 0.0022406, 0.0053297, -0.0029464, 0.0031165
4: 0.0015047, 0.0069069, 0.0015647, 0.0068312, -0.0053265, 0.0053422
5: 0.0027850, 0.0084179, 0.0025148, 0.0083658, -0.0055808, 0.0059031
6: -0.0042209, 0.0012518, -0.0041613, 0.0015016, -0.0057225, 0.0054131
7: -0.0089029, -0.0064934, -0.0088806, -0.0063778, -0.0025251, 0.0023872
8: 0.0033846, 0.0082231, 0.0034844, 0.0082383, -0.0048315, 0.0047273
9: -0.0051485, -0.0017078, -0.0051167, -0.0015428, -0.0036057, 0.0034089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030040, upper bound: 0.0028542
time: 1.57 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029667, upper bound: 0.0027917
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002436, 0.0021335, -0.0004388, 0.0021157, -0.0023593, 0.0025723
1: 0.9913613, 0.9963951, 0.9913990, 0.9968084, -0.0054471, 0.0049961
2: -0.0081277, -0.0039068, -0.0082254, -0.0039682, -0.0041595, 0.0043186
3: 0.0023833, 0.0053572, 0.0021390, 0.0053349, -0.0029516, 0.0032182
4: 0.0015047, 0.0069069, 0.0015533, 0.0068456, -0.0053409, 0.0053536
5: 0.0027850, 0.0084179, 0.0023224, 0.0083758, -0.0055907, 0.0060956
6: -0.0042209, 0.0012518, -0.0041727, 0.0016796, -0.0059005, 0.0054244
7: -0.0089029, -0.0064934, -0.0088848, -0.0062955, -0.0026074, 0.0023915
8: 0.0033846, 0.0082231, 0.0034654, 0.0082492, -0.0048345, 0.0047478
9: -0.0051485, -0.0017078, -0.0051227, -0.0014252, -0.0037233, 0.0034150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030040, upper bound: 0.0028542
time: 1.51 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029667, upper bound: 0.0027917
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0003576, 0.0021115, -0.0024430, 0.0025049
1: 0.9913321, 0.9965813, 0.9914078, 0.9966366, -0.0053045, 0.0051736
2: -0.0081717, -0.0038591, -0.0081848, -0.0039827, -0.0041890, 0.0043257
3: 0.0022733, 0.0053745, 0.0022406, 0.0053297, -0.0030564, 0.0031338
4: 0.0014670, 0.0069545, 0.0015647, 0.0068312, -0.0053642, 0.0053898
5: 0.0025768, 0.0084506, 0.0025148, 0.0083658, -0.0057891, 0.0059358
6: -0.0042583, 0.0014443, -0.0041613, 0.0015016, -0.0057599, 0.0056057
7: -0.0089169, -0.0064043, -0.0088806, -0.0063778, -0.0025391, 0.0024763
8: 0.0033220, 0.0082349, 0.0034844, 0.0082383, -0.0048952, 0.0047313
9: -0.0051685, -0.0015806, -0.0051167, -0.0015428, -0.0036257, 0.0035361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030042, upper bound: 0.0029790
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029672, upper bound: 0.0029568
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003315, 0.0021473, -0.0004388, 0.0021157, -0.0024472, 0.0025861
1: 0.9913321, 0.9965813, 0.9913990, 0.9968084, -0.0054764, 0.0051823
2: -0.0081717, -0.0038591, -0.0082254, -0.0039682, -0.0042035, 0.0043663
3: 0.0022733, 0.0053745, 0.0021390, 0.0053349, -0.0030616, 0.0032354
4: 0.0014670, 0.0069545, 0.0015533, 0.0068456, -0.0053786, 0.0054012
5: 0.0025768, 0.0084506, 0.0023224, 0.0083758, -0.0057990, 0.0061283
6: -0.0042583, 0.0014443, -0.0041727, 0.0016796, -0.0059378, 0.0056170
7: -0.0089169, -0.0064043, -0.0088848, -0.0062955, -0.0026214, 0.0024806
8: 0.0033220, 0.0082349, 0.0034654, 0.0082492, -0.0048985, 0.0047539
9: -0.0051685, -0.0015806, -0.0051227, -0.0014252, -0.0037433, 0.0035422

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030042, upper bound: 0.0030030
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029672, upper bound: 0.0029777
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0001785, 0.0021020, -0.0024017, 0.0022642
1: 0.9914624, 0.9965141, 0.9914280, 0.9962574, -0.0047950, 0.0050861
2: -0.0081559, -0.0040717, -0.0080952, -0.0040155, -0.0041403, 0.0040235
3: 0.0023130, 0.0052974, 0.0024647, 0.0053178, -0.0030048, 0.0028327
4: 0.0016351, 0.0067424, 0.0015907, 0.0067984, -0.0051634, 0.0051517
5: 0.0026519, 0.0083048, 0.0029392, 0.0083433, -0.0056914, 0.0053655
6: -0.0040915, 0.0013748, -0.0041356, 0.0011092, -0.0052007, 0.0055104
7: -0.0088545, -0.0064364, -0.0088709, -0.0065593, -0.0022951, 0.0024345
8: 0.0036013, 0.0082306, 0.0035275, 0.0082144, -0.0046020, 0.0046811
9: -0.0050794, -0.0016265, -0.0051029, -0.0018020, -0.0032774, 0.0034764

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029747
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029775
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0002602, 0.0021095, -0.0024092, 0.0023459
1: 0.9914624, 0.9965141, 0.9914120, 0.9964303, -0.0049679, 0.0051020
2: -0.0081559, -0.0040717, -0.0081361, -0.0039896, -0.0041662, 0.0040644
3: 0.0023130, 0.0052974, 0.0023625, 0.0053272, -0.0030142, 0.0029349
4: 0.0016351, 0.0067424, 0.0015702, 0.0068243, -0.0051892, 0.0051722
5: 0.0026519, 0.0083048, 0.0027457, 0.0083611, -0.0057092, 0.0055591
6: -0.0040915, 0.0013748, -0.0041559, 0.0012881, -0.0053797, 0.0055307
7: -0.0088545, -0.0064364, -0.0088785, -0.0064765, -0.0023779, 0.0024421
8: 0.0036013, 0.0082306, 0.0034935, 0.0082253, -0.0046057, 0.0047154
9: -0.0050794, -0.0016265, -0.0051138, -0.0016838, -0.0033956, 0.0034873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029747
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029779
time: 1.13 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0001785, 0.0021020, -0.0024754, 0.0022641
1: 0.9914626, 0.9966701, 0.9914280, 0.9962574, -0.0047947, 0.0052421
2: -0.0081927, -0.0040720, -0.0080952, -0.0040155, -0.0041772, 0.0040231
3: 0.0022208, 0.0052973, 0.0024647, 0.0053178, -0.0030970, 0.0028326
4: 0.0016353, 0.0067420, 0.0015907, 0.0067984, -0.0051631, 0.0051514
5: 0.0024773, 0.0083045, 0.0029392, 0.0083433, -0.0058660, 0.0053653
6: -0.0040913, 0.0015363, -0.0041356, 0.0011092, -0.0052004, 0.0056719
7: -0.0088544, -0.0063617, -0.0088709, -0.0065593, -0.0022950, 0.0025092
8: 0.0036018, 0.0082405, 0.0035275, 0.0082144, -0.0046025, 0.0046845
9: -0.0050792, -0.0015198, -0.0051029, -0.0018020, -0.0032772, 0.0035831

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030704
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030749
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0002602, 0.0021095, -0.0024829, 0.0023458
1: 0.9914626, 0.9966701, 0.9914120, 0.9964303, -0.0049676, 0.0052581
2: -0.0081927, -0.0040720, -0.0081361, -0.0039896, -0.0042031, 0.0040640
3: 0.0022208, 0.0052973, 0.0023625, 0.0053272, -0.0031063, 0.0029348
4: 0.0016353, 0.0067420, 0.0015702, 0.0068243, -0.0051889, 0.0051718
5: 0.0024773, 0.0083045, 0.0027457, 0.0083611, -0.0058838, 0.0055589
6: -0.0040913, 0.0015363, -0.0041559, 0.0012881, -0.0053794, 0.0056922
7: -0.0088544, -0.0063617, -0.0088785, -0.0064765, -0.0023778, 0.0025168
8: 0.0036018, 0.0082405, 0.0034935, 0.0082253, -0.0046077, 0.0047186
9: -0.0050792, -0.0015198, -0.0051138, -0.0016838, -0.0033955, 0.0035940

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0031058
time: 1.23 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0031141
time: 1.30 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0002436, 0.0021335, -0.0024332, 0.0023293
1: 0.9914624, 0.9965141, 0.9913613, 0.9963951, -0.0049326, 0.0051528
2: -0.0081559, -0.0040717, -0.0081277, -0.0039068, -0.0042491, 0.0040560
3: 0.0023130, 0.0052974, 0.0023833, 0.0053572, -0.0030442, 0.0029142
4: 0.0016351, 0.0067424, 0.0015047, 0.0069069, -0.0052719, 0.0052377
5: 0.0026519, 0.0083048, 0.0027850, 0.0084179, -0.0057660, 0.0055198
6: -0.0040915, 0.0013748, -0.0042209, 0.0012518, -0.0053433, 0.0055957
7: -0.0088545, -0.0064364, -0.0089029, -0.0064934, -0.0023611, 0.0024664
8: 0.0036013, 0.0082306, 0.0033846, 0.0082231, -0.0046102, 0.0048275
9: -0.0050794, -0.0016265, -0.0051485, -0.0017078, -0.0033716, 0.0035220

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029677
time: 1.12 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029702
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0003315, 0.0021473, -0.0024470, 0.0024172
1: 0.9914624, 0.9965141, 0.9913321, 0.9965813, -0.0051189, 0.0051820
2: -0.0081559, -0.0040717, -0.0081717, -0.0038591, -0.0042968, 0.0041000
3: 0.0023130, 0.0052974, 0.0022733, 0.0053745, -0.0030614, 0.0030241
4: 0.0016351, 0.0067424, 0.0014670, 0.0069545, -0.0053195, 0.0052754
5: 0.0026519, 0.0083048, 0.0025768, 0.0084506, -0.0057987, 0.0057280
6: -0.0040915, 0.0013748, -0.0042583, 0.0014443, -0.0055359, 0.0056331
7: -0.0088545, -0.0064364, -0.0089169, -0.0064043, -0.0024502, 0.0024804
8: 0.0036013, 0.0082306, 0.0033220, 0.0082349, -0.0046154, 0.0048921
9: -0.0050794, -0.0016265, -0.0051685, -0.0015806, -0.0034988, 0.0035420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029678
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029704
time: 1.19 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0002436, 0.0021335, -0.0025069, 0.0023292
1: 0.9914626, 0.9966701, 0.9913613, 0.9963951, -0.0049324, 0.0053088
2: -0.0081927, -0.0040720, -0.0081277, -0.0039068, -0.0042860, 0.0040557
3: 0.0022208, 0.0052973, 0.0023833, 0.0053572, -0.0031364, 0.0029141
4: 0.0016353, 0.0067420, 0.0015047, 0.0069069, -0.0052716, 0.0052373
5: 0.0024773, 0.0083045, 0.0027850, 0.0084179, -0.0059407, 0.0055195
6: -0.0040913, 0.0015363, -0.0042209, 0.0012518, -0.0053430, 0.0057572
7: -0.0088544, -0.0063617, -0.0089029, -0.0064934, -0.0023610, 0.0025412
8: 0.0036018, 0.0082405, 0.0033846, 0.0082231, -0.0046106, 0.0048309
9: -0.0050792, -0.0015198, -0.0051485, -0.0017078, -0.0033715, 0.0036287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030630
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030662
time: 1.27 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0003315, 0.0021473, -0.0025207, 0.0024171
1: 0.9914626, 0.9966701, 0.9913321, 0.9965813, -0.0051187, 0.0053381
2: -0.0081927, -0.0040720, -0.0081717, -0.0038591, -0.0043336, 0.0040997
3: 0.0022208, 0.0052973, 0.0022733, 0.0053745, -0.0031536, 0.0030240
4: 0.0016353, 0.0067420, 0.0014670, 0.0069545, -0.0053192, 0.0052750
5: 0.0024773, 0.0083045, 0.0025768, 0.0084506, -0.0059734, 0.0057278
6: -0.0040913, 0.0015363, -0.0042583, 0.0014443, -0.0055356, 0.0057946
7: -0.0088544, -0.0063617, -0.0089169, -0.0064043, -0.0024501, 0.0025552
8: 0.0036018, 0.0082405, 0.0033220, 0.0082349, -0.0046168, 0.0048941
9: -0.0050792, -0.0015198, -0.0051685, -0.0015806, -0.0034987, 0.0036487

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030976
time: 1.09 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0031059
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0001785, 0.0021020, -0.0024596, 0.0022900
1: 0.9914078, 0.9966366, 0.9914280, 0.9962574, -0.0048496, 0.0052086
2: -0.0081848, -0.0039827, -0.0080952, -0.0040155, -0.0041693, 0.0041125
3: 0.0022406, 0.0053297, 0.0024647, 0.0053178, -0.0030771, 0.0028650
4: 0.0015647, 0.0068312, 0.0015907, 0.0067984, -0.0052337, 0.0052405
5: 0.0025148, 0.0083658, 0.0029392, 0.0083433, -0.0058285, 0.0054266
6: -0.0041613, 0.0015016, -0.0041356, 0.0011092, -0.0052705, 0.0056372
7: -0.0088806, -0.0063778, -0.0088709, -0.0065593, -0.0023212, 0.0024931
8: 0.0034844, 0.0082383, 0.0035275, 0.0082144, -0.0047209, 0.0046882
9: -0.0051167, -0.0015428, -0.0051029, -0.0018020, -0.0033147, 0.0035602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029679
time: 1.15 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029687
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0002602, 0.0021095, -0.0024671, 0.0023717
1: 0.9914078, 0.9966366, 0.9914120, 0.9964303, -0.0050225, 0.0052245
2: -0.0081848, -0.0039827, -0.0081361, -0.0039896, -0.0041951, 0.0041534
3: 0.0022406, 0.0053297, 0.0023625, 0.0053272, -0.0030865, 0.0029672
4: 0.0015647, 0.0068312, 0.0015702, 0.0068243, -0.0052595, 0.0052610
5: 0.0025148, 0.0083658, 0.0027457, 0.0083611, -0.0058462, 0.0056202
6: -0.0041613, 0.0015016, -0.0041559, 0.0012881, -0.0054495, 0.0056575
7: -0.0088806, -0.0063778, -0.0088785, -0.0064765, -0.0024041, 0.0025007
8: 0.0034844, 0.0082383, 0.0034935, 0.0082253, -0.0047246, 0.0047224
9: -0.0051167, -0.0015428, -0.0051138, -0.0016838, -0.0034329, 0.0035710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029680
time: 1.29 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029689
time: 1.15 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0001785, 0.0021020, -0.0025408, 0.0022942
1: 0.9913990, 0.9968084, 0.9914280, 0.9962574, -0.0048584, 0.0053804
2: -0.0082254, -0.0039682, -0.0080952, -0.0040155, -0.0042099, 0.0041269
3: 0.0021390, 0.0053349, 0.0024647, 0.0053178, -0.0031788, 0.0028702
4: 0.0015533, 0.0068456, 0.0015907, 0.0067984, -0.0052451, 0.0052549
5: 0.0023224, 0.0083758, 0.0029392, 0.0083433, -0.0060209, 0.0054365
6: -0.0041727, 0.0016796, -0.0041356, 0.0011092, -0.0052818, 0.0058152
7: -0.0088848, -0.0062955, -0.0088709, -0.0065593, -0.0023255, 0.0025755
8: 0.0034654, 0.0082492, 0.0035275, 0.0082144, -0.0047417, 0.0046931
9: -0.0051227, -0.0014252, -0.0051029, -0.0018020, -0.0033207, 0.0036777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030538
time: 1.07 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030569
time: 1.26 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0002602, 0.0021095, -0.0025483, 0.0023759
1: 0.9913990, 0.9968084, 0.9914120, 0.9964303, -0.0050313, 0.0053964
2: -0.0082254, -0.0039682, -0.0081361, -0.0039896, -0.0042358, 0.0041678
3: 0.0021390, 0.0053349, 0.0023625, 0.0053272, -0.0031881, 0.0029724
4: 0.0015533, 0.0068456, 0.0015702, 0.0068243, -0.0052710, 0.0052754
5: 0.0023224, 0.0083758, 0.0027457, 0.0083611, -0.0060387, 0.0056301
6: -0.0041727, 0.0016796, -0.0041559, 0.0012881, -0.0054608, 0.0058355
7: -0.0088848, -0.0062955, -0.0088785, -0.0064765, -0.0024083, 0.0025831
8: 0.0034654, 0.0082492, 0.0034935, 0.0082253, -0.0047462, 0.0047262
9: -0.0051227, -0.0014252, -0.0051138, -0.0016838, -0.0034390, 0.0036886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030850
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030569
time: 1.20 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0002436, 0.0021335, -0.0024911, 0.0023551
1: 0.9914078, 0.9966366, 0.9913613, 0.9963951, -0.0049873, 0.0052752
2: -0.0081848, -0.0039827, -0.0081277, -0.0039068, -0.0042780, 0.0041450
3: 0.0022406, 0.0053297, 0.0023833, 0.0053572, -0.0031165, 0.0029464
4: 0.0015647, 0.0068312, 0.0015047, 0.0069069, -0.0053422, 0.0053265
5: 0.0025148, 0.0083658, 0.0027850, 0.0084179, -0.0059031, 0.0055808
6: -0.0041613, 0.0015016, -0.0042209, 0.0012518, -0.0054131, 0.0057225
7: -0.0088806, -0.0063778, -0.0089029, -0.0064934, -0.0023872, 0.0025251
8: 0.0034844, 0.0082383, 0.0033846, 0.0082231, -0.0047273, 0.0048315
9: -0.0051167, -0.0015428, -0.0051485, -0.0017078, -0.0034089, 0.0036057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029656
time: 1.27 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029667
time: 1.22 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0003315, 0.0021473, -0.0025049, 0.0024430
1: 0.9914078, 0.9966366, 0.9913321, 0.9965813, -0.0051736, 0.0053045
2: -0.0081848, -0.0039827, -0.0081717, -0.0038591, -0.0043257, 0.0041890
3: 0.0022406, 0.0053297, 0.0022733, 0.0053745, -0.0031338, 0.0030564
4: 0.0015647, 0.0068312, 0.0014670, 0.0069545, -0.0053898, 0.0053642
5: 0.0025148, 0.0083658, 0.0025768, 0.0084506, -0.0059358, 0.0057891
6: -0.0041613, 0.0015016, -0.0042583, 0.0014443, -0.0056057, 0.0057599
7: -0.0088806, -0.0063778, -0.0089169, -0.0064043, -0.0024763, 0.0025391
8: 0.0034844, 0.0082383, 0.0033220, 0.0082349, -0.0047313, 0.0048952
9: -0.0051167, -0.0015428, -0.0051685, -0.0015806, -0.0035361, 0.0036257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029660
time: 1.46 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029672
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0002436, 0.0021335, -0.0025723, 0.0023593
1: 0.9913990, 0.9968084, 0.9913613, 0.9963951, -0.0049961, 0.0054471
2: -0.0082254, -0.0039682, -0.0081277, -0.0039068, -0.0043186, 0.0041595
3: 0.0021390, 0.0053349, 0.0023833, 0.0053572, -0.0032182, 0.0029516
4: 0.0015533, 0.0068456, 0.0015047, 0.0069069, -0.0053536, 0.0053409
5: 0.0023224, 0.0083758, 0.0027850, 0.0084179, -0.0060956, 0.0055907
6: -0.0041727, 0.0016796, -0.0042209, 0.0012518, -0.0054244, 0.0059005
7: -0.0088848, -0.0062955, -0.0089029, -0.0064934, -0.0023915, 0.0026074
8: 0.0034654, 0.0082492, 0.0033846, 0.0082231, -0.0047478, 0.0048345
9: -0.0051227, -0.0014252, -0.0051485, -0.0017078, -0.0034150, 0.0037233

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030538
time: 1.33 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030567
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0003315, 0.0021473, -0.0025861, 0.0024472
1: 0.9913990, 0.9968084, 0.9913321, 0.9965813, -0.0051823, 0.0054764
2: -0.0082254, -0.0039682, -0.0081717, -0.0038591, -0.0043663, 0.0042035
3: 0.0021390, 0.0053349, 0.0022733, 0.0053745, -0.0032354, 0.0030616
4: 0.0015533, 0.0068456, 0.0014670, 0.0069545, -0.0054012, 0.0053786
5: 0.0023224, 0.0083758, 0.0025768, 0.0084506, -0.0061283, 0.0057990
6: -0.0041727, 0.0016796, -0.0042583, 0.0014443, -0.0056170, 0.0059378
7: -0.0088848, -0.0062955, -0.0089169, -0.0064043, -0.0024806, 0.0026214
8: 0.0034654, 0.0082492, 0.0033220, 0.0082349, -0.0047539, 0.0048985
9: -0.0051227, -0.0014252, -0.0051685, -0.0015806, -0.0035422, 0.0037433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030850
time: 1.11 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030919
time: 1.08 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0002998, 0.0020857, -0.0023855, 0.0023855
1: 0.9914624, 0.9965141, 0.9914624, 0.9965141, -0.0050517, 0.0050517
2: -0.0081559, -0.0040717, -0.0081559, -0.0040717, -0.0040842, 0.0040842
3: 0.0023130, 0.0052974, 0.0023130, 0.0052974, -0.0029844, 0.0029844
4: 0.0016351, 0.0067424, 0.0016351, 0.0067424, -0.0051073, 0.0051073
5: 0.0026519, 0.0083048, 0.0026519, 0.0083048, -0.0056529, 0.0056529
6: -0.0040915, 0.0013748, -0.0040915, 0.0013748, -0.0054664, 0.0054664
7: -0.0088545, -0.0064364, -0.0088545, -0.0064364, -0.0024180, 0.0024180
8: 0.0036013, 0.0082306, 0.0036013, 0.0082306, -0.0046073, 0.0046073
9: -0.0050794, -0.0016265, -0.0050794, -0.0016265, -0.0034529, 0.0034529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030254, upper bound: 0.0030401
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030206, upper bound: 0.0030440
time: 1.25 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0003735, 0.0020856, -0.0023854, 0.0024592
1: 0.9914624, 0.9965141, 0.9914626, 0.9966701, -0.0052077, 0.0050514
2: -0.0081559, -0.0040717, -0.0081927, -0.0040720, -0.0040838, 0.0041210
3: 0.0023130, 0.0052974, 0.0022208, 0.0052973, -0.0029843, 0.0030766
4: 0.0016351, 0.0067424, 0.0016353, 0.0067420, -0.0051070, 0.0051071
5: 0.0026519, 0.0083048, 0.0024773, 0.0083045, -0.0056526, 0.0058275
6: -0.0040915, 0.0013748, -0.0040913, 0.0015363, -0.0056279, 0.0054661
7: -0.0088545, -0.0064364, -0.0088544, -0.0063617, -0.0024928, 0.0024179
8: 0.0036013, 0.0082306, 0.0036018, 0.0082405, -0.0046100, 0.0046084
9: -0.0050794, -0.0016265, -0.0050792, -0.0015198, -0.0035596, 0.0034527

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030254, upper bound: 0.0030401
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030206, upper bound: 0.0030440
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0002998, 0.0020857, -0.0024592, 0.0023854
1: 0.9914626, 0.9966701, 0.9914624, 0.9965141, -0.0050514, 0.0052077
2: -0.0081927, -0.0040720, -0.0081559, -0.0040717, -0.0041210, 0.0040838
3: 0.0022208, 0.0052973, 0.0023130, 0.0052974, -0.0030766, 0.0029843
4: 0.0016353, 0.0067420, 0.0016351, 0.0067424, -0.0051071, 0.0051070
5: 0.0024773, 0.0083045, 0.0026519, 0.0083048, -0.0058275, 0.0056526
6: -0.0040913, 0.0015363, -0.0040915, 0.0013748, -0.0054661, 0.0056279
7: -0.0088544, -0.0063617, -0.0088545, -0.0064364, -0.0024179, 0.0024928
8: 0.0036018, 0.0082405, 0.0036013, 0.0082306, -0.0046084, 0.0046100
9: -0.0050792, -0.0015198, -0.0050794, -0.0016265, -0.0034527, 0.0035596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030223, upper bound: 0.0031240
time: 1.24 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0031369
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0003735, 0.0020856, -0.0024591, 0.0024591
1: 0.9914626, 0.9966701, 0.9914626, 0.9966701, -0.0052075, 0.0052075
2: -0.0081927, -0.0040720, -0.0081927, -0.0040720, -0.0041207, 0.0041207
3: 0.0022208, 0.0052973, 0.0022208, 0.0052973, -0.0030765, 0.0030765
4: 0.0016353, 0.0067420, 0.0016353, 0.0067420, -0.0051067, 0.0051067
5: 0.0024773, 0.0083045, 0.0024773, 0.0083045, -0.0058273, 0.0058273
6: -0.0040913, 0.0015363, -0.0040913, 0.0015363, -0.0056276, 0.0056276
7: -0.0088544, -0.0063617, -0.0088544, -0.0063617, -0.0024927, 0.0024927
8: 0.0036018, 0.0082405, 0.0036018, 0.0082405, -0.0046114, 0.0046114
9: -0.0050792, -0.0015198, -0.0050792, -0.0015198, -0.0035594, 0.0035594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030223, upper bound: 0.0031343
time: 1.32 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0031490
time: 1.36 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0003576, 0.0021115, -0.0024113, 0.0024433
1: 0.9914624, 0.9965141, 0.9914078, 0.9966366, -0.0051742, 0.0051063
2: -0.0081559, -0.0040717, -0.0081848, -0.0039827, -0.0041731, 0.0041131
3: 0.0023130, 0.0052974, 0.0022406, 0.0053297, -0.0030167, 0.0030568
4: 0.0016351, 0.0067424, 0.0015647, 0.0068312, -0.0051961, 0.0051777
5: 0.0026519, 0.0083048, 0.0025148, 0.0083658, -0.0057139, 0.0057899
6: -0.0040915, 0.0013748, -0.0041613, 0.0015016, -0.0055931, 0.0055362
7: -0.0088545, -0.0064364, -0.0088806, -0.0063778, -0.0024767, 0.0024442
8: 0.0036013, 0.0082306, 0.0034844, 0.0082383, -0.0046149, 0.0047266
9: -0.0050794, -0.0016265, -0.0051167, -0.0015428, -0.0035366, 0.0034902

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030403
time: 1.40 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030113, upper bound: 0.0030436
time: 1.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0002998, 0.0020857, -0.0004388, 0.0021157, -0.0024154, 0.0025246
1: 0.9914624, 0.9965141, 0.9913990, 0.9968084, -0.0053460, 0.0051151
2: -0.0081559, -0.0040717, -0.0082254, -0.0039682, -0.0041876, 0.0041537
3: 0.0023130, 0.0052974, 0.0021390, 0.0053349, -0.0030219, 0.0031584
4: 0.0016351, 0.0067424, 0.0015533, 0.0068456, -0.0052105, 0.0051891
5: 0.0026519, 0.0083048, 0.0023224, 0.0083758, -0.0057238, 0.0059824
6: -0.0040915, 0.0013748, -0.0041727, 0.0016796, -0.0057711, 0.0055475
7: -0.0088545, -0.0064364, -0.0088848, -0.0062955, -0.0025590, 0.0024484
8: 0.0036013, 0.0082306, 0.0034654, 0.0082492, -0.0046188, 0.0047481
9: -0.0050794, -0.0016265, -0.0051227, -0.0014252, -0.0036542, 0.0034962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030403
time: 1.35 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030113, upper bound: 0.0030435
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0003576, 0.0021115, -0.0024850, 0.0024432
1: 0.9914626, 0.9966701, 0.9914078, 0.9966366, -0.0051739, 0.0052624
2: -0.0081927, -0.0040720, -0.0081848, -0.0039827, -0.0042100, 0.0041127
3: 0.0022208, 0.0052973, 0.0022406, 0.0053297, -0.0031089, 0.0030567
4: 0.0016353, 0.0067420, 0.0015647, 0.0068312, -0.0051959, 0.0051773
5: 0.0024773, 0.0083045, 0.0025148, 0.0083658, -0.0058886, 0.0057897
6: -0.0040913, 0.0015363, -0.0041613, 0.0015016, -0.0055929, 0.0056977
7: -0.0088544, -0.0063617, -0.0088806, -0.0063778, -0.0024766, 0.0025189
8: 0.0036018, 0.0082405, 0.0034844, 0.0082383, -0.0046160, 0.0047293
9: -0.0050792, -0.0015198, -0.0051167, -0.0015428, -0.0035365, 0.0035969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030144, upper bound: 0.0031386
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030085, upper bound: 0.0031465
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0003735, 0.0020856, -0.0004388, 0.0021157, -0.0024891, 0.0025245
1: 0.9914626, 0.9966701, 0.9913990, 0.9968084, -0.0053458, 0.0052711
2: -0.0081927, -0.0040720, -0.0082254, -0.0039682, -0.0042245, 0.0041534
3: 0.0022208, 0.0052973, 0.0021390, 0.0053349, -0.0031141, 0.0031583
4: 0.0016353, 0.0067420, 0.0015533, 0.0068456, -0.0052103, 0.0051887
5: 0.0024773, 0.0083045, 0.0023224, 0.0083758, -0.0058985, 0.0059822
6: -0.0040913, 0.0015363, -0.0041727, 0.0016796, -0.0057709, 0.0057090
7: -0.0088544, -0.0063617, -0.0088848, -0.0062955, -0.0025589, 0.0025231
8: 0.0036018, 0.0082405, 0.0034654, 0.0082492, -0.0046197, 0.0047505
9: -0.0050792, -0.0015198, -0.0051227, -0.0014252, -0.0036540, 0.0036029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030144, upper bound: 0.0031482
time: 1.22 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030085, upper bound: 0.0031576
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0002998, 0.0020857, -0.0024433, 0.0024113
1: 0.9914078, 0.9966366, 0.9914624, 0.9965141, -0.0051063, 0.0051742
2: -0.0081848, -0.0039827, -0.0081559, -0.0040717, -0.0041131, 0.0041731
3: 0.0022406, 0.0053297, 0.0023130, 0.0052974, -0.0030568, 0.0030167
4: 0.0015647, 0.0068312, 0.0016351, 0.0067424, -0.0051777, 0.0051961
5: 0.0025148, 0.0083658, 0.0026519, 0.0083048, -0.0057899, 0.0057139
6: -0.0041613, 0.0015016, -0.0040915, 0.0013748, -0.0055362, 0.0055931
7: -0.0088806, -0.0063778, -0.0088545, -0.0064364, -0.0024442, 0.0024767
8: 0.0034844, 0.0082383, 0.0036013, 0.0082306, -0.0047266, 0.0046149
9: -0.0051167, -0.0015428, -0.0050794, -0.0016265, -0.0034902, 0.0035366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030306, upper bound: 0.0030363
time: 1.33 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030258, upper bound: 0.0030384
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0003735, 0.0020856, -0.0024432, 0.0024850
1: 0.9914078, 0.9966366, 0.9914626, 0.9966701, -0.0052624, 0.0051739
2: -0.0081848, -0.0039827, -0.0081927, -0.0040720, -0.0041127, 0.0042100
3: 0.0022406, 0.0053297, 0.0022208, 0.0052973, -0.0030567, 0.0031089
4: 0.0015647, 0.0068312, 0.0016353, 0.0067420, -0.0051773, 0.0051959
5: 0.0025148, 0.0083658, 0.0024773, 0.0083045, -0.0057897, 0.0058886
6: -0.0041613, 0.0015016, -0.0040913, 0.0015363, -0.0056977, 0.0055929
7: -0.0088806, -0.0063778, -0.0088544, -0.0063617, -0.0025189, 0.0024766
8: 0.0034844, 0.0082383, 0.0036018, 0.0082405, -0.0047293, 0.0046160
9: -0.0051167, -0.0015428, -0.0050792, -0.0015198, -0.0035969, 0.0035365

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030306, upper bound: 0.0030363
time: 1.44 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030258, upper bound: 0.0030385
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0002998, 0.0020857, -0.0025246, 0.0024154
1: 0.9913990, 0.9968084, 0.9914624, 0.9965141, -0.0051151, 0.0053460
2: -0.0082254, -0.0039682, -0.0081559, -0.0040717, -0.0041537, 0.0041876
3: 0.0021390, 0.0053349, 0.0023130, 0.0052974, -0.0031584, 0.0030219
4: 0.0015533, 0.0068456, 0.0016351, 0.0067424, -0.0051891, 0.0052105
5: 0.0023224, 0.0083758, 0.0026519, 0.0083048, -0.0059824, 0.0057238
6: -0.0041727, 0.0016796, -0.0040915, 0.0013748, -0.0055475, 0.0057711
7: -0.0088848, -0.0062955, -0.0088545, -0.0064364, -0.0024484, 0.0025590
8: 0.0034654, 0.0082492, 0.0036013, 0.0082306, -0.0047481, 0.0046188
9: -0.0051227, -0.0014252, -0.0050794, -0.0016265, -0.0034962, 0.0036542

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0031120
time: 1.41 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030232, upper bound: 0.0031236
time: 1.19 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0003735, 0.0020856, -0.0025245, 0.0024891
1: 0.9913990, 0.9968084, 0.9914626, 0.9966701, -0.0052711, 0.0053458
2: -0.0082254, -0.0039682, -0.0081927, -0.0040720, -0.0041534, 0.0042245
3: 0.0021390, 0.0053349, 0.0022208, 0.0052973, -0.0031583, 0.0031141
4: 0.0015533, 0.0068456, 0.0016353, 0.0067420, -0.0051887, 0.0052103
5: 0.0023224, 0.0083758, 0.0024773, 0.0083045, -0.0059822, 0.0058985
6: -0.0041727, 0.0016796, -0.0040913, 0.0015363, -0.0057090, 0.0057709
7: -0.0088848, -0.0062955, -0.0088544, -0.0063617, -0.0025231, 0.0025589
8: 0.0034654, 0.0082492, 0.0036018, 0.0082405, -0.0047505, 0.0046197
9: -0.0051227, -0.0014252, -0.0050792, -0.0015198, -0.0036029, 0.0036540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0031203
time: 1.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030232, upper bound: 0.0031358
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0003576, 0.0021115, -0.0024691, 0.0024691
1: 0.9914078, 0.9966366, 0.9914078, 0.9966366, -0.0052288, 0.0052288
2: -0.0081848, -0.0039827, -0.0081848, -0.0039827, -0.0042021, 0.0042021
3: 0.0022406, 0.0053297, 0.0022406, 0.0053297, -0.0030890, 0.0030890
4: 0.0015647, 0.0068312, 0.0015647, 0.0068312, -0.0052665, 0.0052665
5: 0.0025148, 0.0083658, 0.0025148, 0.0083658, -0.0058510, 0.0058510
6: -0.0041613, 0.0015016, -0.0041613, 0.0015016, -0.0056629, 0.0056629
7: -0.0088806, -0.0063778, -0.0088806, -0.0063778, -0.0025028, 0.0025028
8: 0.0034844, 0.0082383, 0.0034844, 0.0082383, -0.0047306, 0.0047306
9: -0.0051167, -0.0015428, -0.0051167, -0.0015428, -0.0035739, 0.0035739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030285, upper bound: 0.0030363
time: 1.49 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030240, upper bound: 0.0030384
time: 1.40 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003576, 0.0021115, -0.0004388, 0.0021157, -0.0024733, 0.0025503
1: 0.9914078, 0.9966366, 0.9913990, 0.9968084, -0.0054007, 0.0052376
2: -0.0081848, -0.0039827, -0.0082254, -0.0039682, -0.0042165, 0.0042427
3: 0.0022406, 0.0053297, 0.0021390, 0.0053349, -0.0030943, 0.0031907
4: 0.0015647, 0.0068312, 0.0015533, 0.0068456, -0.0052809, 0.0052779
5: 0.0025148, 0.0083658, 0.0023224, 0.0083758, -0.0058609, 0.0060435
6: -0.0041613, 0.0015016, -0.0041727, 0.0016796, -0.0058409, 0.0056743
7: -0.0088806, -0.0063778, -0.0088848, -0.0062955, -0.0025851, 0.0025070
8: 0.0034844, 0.0082383, 0.0034654, 0.0082492, -0.0047334, 0.0047515
9: -0.0051167, -0.0015428, -0.0051227, -0.0014252, -0.0036915, 0.0035800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030285, upper bound: 0.0030363
time: 1.37 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030240, upper bound: 0.0030385
time: 1.37 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0003576, 0.0021115, -0.0025503, 0.0024733
1: 0.9913990, 0.9968084, 0.9914078, 0.9966366, -0.0052376, 0.0054007
2: -0.0082254, -0.0039682, -0.0081848, -0.0039827, -0.0042427, 0.0042165
3: 0.0021390, 0.0053349, 0.0022406, 0.0053297, -0.0031907, 0.0030943
4: 0.0015533, 0.0068456, 0.0015647, 0.0068312, -0.0052779, 0.0052809
5: 0.0023224, 0.0083758, 0.0025148, 0.0083658, -0.0060435, 0.0058609
6: -0.0041727, 0.0016796, -0.0041613, 0.0015016, -0.0056743, 0.0058409
7: -0.0088848, -0.0062955, -0.0088806, -0.0063778, -0.0025070, 0.0025851
8: 0.0034654, 0.0082492, 0.0034844, 0.0082383, -0.0047515, 0.0047334
9: -0.0051227, -0.0014252, -0.0051167, -0.0015428, -0.0035800, 0.0036915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030263, upper bound: 0.0031120
time: 1.55 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0031236
time: 1.23 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0004388, 0.0021157, -0.0004388, 0.0021157, -0.0025545, 0.0025545
1: 0.9913990, 0.9968084, 0.9913990, 0.9968084, -0.0054094, 0.0054094
2: -0.0082254, -0.0039682, -0.0082254, -0.0039682, -0.0042572, 0.0042572
3: 0.0021390, 0.0053349, 0.0021390, 0.0053349, -0.0031959, 0.0031959
4: 0.0015533, 0.0068456, 0.0015533, 0.0068456, -0.0052923, 0.0052923
5: 0.0023224, 0.0083758, 0.0023224, 0.0083758, -0.0060534, 0.0060534
6: -0.0041727, 0.0016796, -0.0041727, 0.0016796, -0.0058522, 0.0058522
7: -0.0088848, -0.0062955, -0.0088848, -0.0062955, -0.0025894, 0.0025894
8: 0.0034654, 0.0082492, 0.0034654, 0.0082492, -0.0047553, 0.0047553
9: -0.0051227, -0.0014252, -0.0051227, -0.0014252, -0.0036975, 0.0036975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 97
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 196
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 137
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030263, upper bound: 0.0031120
time: 1.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0031358
time: 1.24 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.47 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028611
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028047
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028611
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028047
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030008
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029894
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030387
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030268
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028568
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028047
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028568
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028047
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029960
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029853
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030274
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030176
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028501
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0027917
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028501
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0027917
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029736
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029531
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030013
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029771
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028497
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0027917
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028497
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0027917
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029736
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029531
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030013
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029770
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030123, upper bound: 0.0028655
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029775, upper bound: 0.0028047
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030123, upper bound: 0.0028655
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029775, upper bound: 0.0028047
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030128, upper bound: 0.0030055
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029779, upper bound: 0.0029938
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030128, upper bound: 0.0030393
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029779, upper bound: 0.0030275
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030037, upper bound: 0.0028632
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0028047
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030037, upper bound: 0.0028632
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029687, upper bound: 0.0028047
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030041, upper bound: 0.0030041
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029689, upper bound: 0.0029929
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030041, upper bound: 0.0030298
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029689, upper bound: 0.0030199
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0028543
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029702, upper bound: 0.0027917
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0028543
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029702, upper bound: 0.0027917
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0029790
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029704, upper bound: 0.0029568
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030089, upper bound: 0.0030030
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029704, upper bound: 0.0029777
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030040, upper bound: 0.0028542
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029667, upper bound: 0.0027917
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030040, upper bound: 0.0028542
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029667, upper bound: 0.0027917
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030042, upper bound: 0.0029790
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029672, upper bound: 0.0029568
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030042, upper bound: 0.0030030
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0029672, upper bound: 0.0029777
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029747
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029775
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029747
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029779
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030704
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030749
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0031058
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0031141
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029677
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029702
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029678
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029704
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030630
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030662
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030976
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0031059
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029679
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029687
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029680
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029689
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030538
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030569
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030850
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0030569
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029656
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029667
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029660
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029672
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030538
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030567
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030850
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0030919
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030254, upper bound: 0.0030401
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030206, upper bound: 0.0030440
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030254, upper bound: 0.0030401
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030206, upper bound: 0.0030440
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030223, upper bound: 0.0031240
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0031369
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030223, upper bound: 0.0031343
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030171, upper bound: 0.0031490
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030403
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030113, upper bound: 0.0030436
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030166, upper bound: 0.0030403
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030113, upper bound: 0.0030435
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030144, upper bound: 0.0031386
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030085, upper bound: 0.0031465
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030144, upper bound: 0.0031482
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030085, upper bound: 0.0031576
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030306, upper bound: 0.0030363
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030258, upper bound: 0.0030384
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030306, upper bound: 0.0030363
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030258, upper bound: 0.0030385
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0031120
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030232, upper bound: 0.0031236
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030276, upper bound: 0.0031203
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030232, upper bound: 0.0031358
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030285, upper bound: 0.0030363
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030240, upper bound: 0.0030384
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030285, upper bound: 0.0030363
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030240, upper bound: 0.0030385
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030263, upper bound: 0.0031120
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0031236
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030263, upper bound: 0.0031120
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.47
Output dim: 1, lower bound: -0.0030218, upper bound: 0.0031358

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001496, 0.0020977, -0.0001785, 0.0021020, -0.0022515, 0.0022763
1: 0.9914370, 0.9961960, 0.9914280, 0.9962574, -0.0048204, 0.0047680
2: -0.0080807, -0.0040301, -0.0080952, -0.0040155, -0.0040652, 0.0040650
3: 0.0025010, 0.0053125, 0.0024647, 0.0053178, -0.0028168, 0.0028478
4: 0.0016022, 0.0067838, 0.0015907, 0.0067984, -0.0051962, 0.0051932
5: 0.0030079, 0.0083333, 0.0029392, 0.0083433, -0.0053354, 0.0053940
6: -0.0041241, 0.0010457, -0.0041356, 0.0011092, -0.0052333, 0.0051812
7: -0.0088667, -0.0065887, -0.0088709, -0.0065593, -0.0023073, 0.0022822
8: 0.0035467, 0.0082106, 0.0035275, 0.0082144, -0.0046562, 0.0046703
9: -0.0050968, -0.0018440, -0.0051029, -0.0018020, -0.0032948, 0.0032590

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028047
time: 1.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0028047
time: 1.04 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001496, 0.0020977, -0.0002602, 0.0021095, -0.0022590, 0.0023579
1: 0.9914370, 0.9961960, 0.9914120, 0.9964303, -0.0049933, 0.0047840
2: -0.0080807, -0.0040301, -0.0081361, -0.0039896, -0.0040911, 0.0041059
3: 0.0025010, 0.0053125, 0.0023625, 0.0053272, -0.0028262, 0.0029500
4: 0.0016022, 0.0067838, 0.0015702, 0.0068243, -0.0052220, 0.0052136
5: 0.0030079, 0.0083333, 0.0027457, 0.0083611, -0.0053532, 0.0055876
6: -0.0041241, 0.0010457, -0.0041559, 0.0012881, -0.0054123, 0.0052015
7: -0.0088667, -0.0065887, -0.0088785, -0.0064765, -0.0023901, 0.0022898
8: 0.0035467, 0.0082106, 0.0034935, 0.0082253, -0.0046599, 0.0047045
9: -0.0050968, -0.0018440, -0.0051138, -0.0016838, -0.0034130, 0.0032698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029894, upper bound: 0.0028047
time: 1.34 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029894, upper bound: 0.0028047
time: 1.12 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002304, 0.0021051, -0.0001785, 0.0021020, -0.0023323, 0.0022836
1: 0.9914213, 0.9963670, 0.9914280, 0.9962574, -0.0048360, 0.0049390
2: -0.0081211, -0.0040047, -0.0080952, -0.0040155, -0.0041056, 0.0040905
3: 0.0023999, 0.0053217, 0.0024647, 0.0053178, -0.0029179, 0.0028570
4: 0.0015821, 0.0068092, 0.0015907, 0.0067984, -0.0052163, 0.0052185
5: 0.0028164, 0.0083507, 0.0029392, 0.0083433, -0.0055269, 0.0054115
6: -0.0041441, 0.0012227, -0.0041356, 0.0011092, -0.0052532, 0.0053583
7: -0.0088741, -0.0065068, -0.0088709, -0.0065593, -0.0023148, 0.0023641
8: 0.0035133, 0.0082213, 0.0035275, 0.0082144, -0.0046900, 0.0046740
9: -0.0051074, -0.0017270, -0.0051029, -0.0018020, -0.0033054, 0.0033759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029894
time: 0.97 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029894
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002215, 0.0021554, -0.0001724, 0.0021011, -0.0023226, 0.0023279
1: 0.9913147, 0.9963483, 0.9914299, 0.9962444, -0.0049297, 0.0049184
2: -0.0081167, -0.0038309, -0.0080921, -0.0040187, -0.0040980, 0.0042613
3: 0.0024110, 0.0053847, 0.0024723, 0.0053167, -0.0029057, 0.0029123
4: 0.0014448, 0.0069826, 0.0015932, 0.0067953, -0.0053505, 0.0053895
5: 0.0028375, 0.0084700, 0.0029537, 0.0083412, -0.0055037, 0.0055163
6: -0.0042804, 0.0012033, -0.0041331, 0.0010958, -0.0053762, 0.0053364
7: -0.0089251, -0.0065158, -0.0088700, -0.0065655, -0.0023596, 0.0023542
8: 0.0032850, 0.0082202, 0.0035317, 0.0082136, -0.0049161, 0.0046885
9: -0.0051803, -0.0017398, -0.0051016, -0.0018108, -0.0033695, 0.0033618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029894
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029894
time: 1.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002304, 0.0021051, -0.0002602, 0.0021095, -0.0023398, 0.0023653
1: 0.9914213, 0.9963670, 0.9914120, 0.9964303, -0.0050089, 0.0049550
2: -0.0081211, -0.0040047, -0.0081361, -0.0039896, -0.0041315, 0.0041313
3: 0.0023999, 0.0053217, 0.0023625, 0.0053272, -0.0029273, 0.0029592
4: 0.0015821, 0.0068092, 0.0015702, 0.0068243, -0.0052421, 0.0052390
5: 0.0028164, 0.0083507, 0.0027457, 0.0083611, -0.0055447, 0.0056050
6: -0.0041441, 0.0012227, -0.0041559, 0.0012881, -0.0054322, 0.0053786
7: -0.0088741, -0.0065068, -0.0088785, -0.0064765, -0.0023976, 0.0023717
8: 0.0035133, 0.0082213, 0.0034935, 0.0082253, -0.0046942, 0.0047087
9: -0.0051074, -0.0017270, -0.0051138, -0.0016838, -0.0034237, 0.0033868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029910, upper bound: 0.0030268
time: 1.33 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029910, upper bound: 0.0030268
time: 1.27 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002215, 0.0021554, -0.0002541, 0.0021085, -0.0023300, 0.0024096
1: 0.9913147, 0.9963483, 0.9914141, 0.9964173, -0.0051026, 0.0049342
2: -0.0081167, -0.0038309, -0.0081330, -0.0039930, -0.0041237, 0.0043021
3: 0.0024110, 0.0053847, 0.0023701, 0.0053260, -0.0029150, 0.0030145
4: 0.0014448, 0.0069826, 0.0015729, 0.0068209, -0.0053762, 0.0054098
5: 0.0028375, 0.0084700, 0.0027601, 0.0083588, -0.0055213, 0.0057099
6: -0.0042804, 0.0012033, -0.0041533, 0.0012748, -0.0055552, 0.0053565
7: -0.0089251, -0.0065158, -0.0088776, -0.0064827, -0.0024424, 0.0023618
8: 0.0032850, 0.0082202, 0.0034979, 0.0082245, -0.0049207, 0.0047223
9: -0.0051803, -0.0017398, -0.0051124, -0.0016926, -0.0034877, 0.0033725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029910, upper bound: 0.0030268
time: 1.26 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029910, upper bound: 0.0030268
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001496, 0.0020977, -0.0002436, 0.0021335, -0.0022830, 0.0023414
1: 0.9914370, 0.9961960, 0.9913613, 0.9963951, -0.0049581, 0.0048347
2: -0.0080807, -0.0040301, -0.0081277, -0.0039068, -0.0041740, 0.0040976
3: 0.0025010, 0.0053125, 0.0023833, 0.0053572, -0.0028562, 0.0029292
4: 0.0016022, 0.0067838, 0.0015047, 0.0069069, -0.0053047, 0.0052791
5: 0.0030079, 0.0083333, 0.0027850, 0.0084179, -0.0054100, 0.0055483
6: -0.0041241, 0.0010457, -0.0042209, 0.0012518, -0.0053759, 0.0052665
7: -0.0088667, -0.0065887, -0.0089029, -0.0064934, -0.0023733, 0.0023141
8: 0.0035467, 0.0082106, 0.0033846, 0.0082231, -0.0046644, 0.0048166
9: -0.0050968, -0.0018440, -0.0051485, -0.0017078, -0.0033890, 0.0033046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028047
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0028047
time: 1.08 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001496, 0.0020977, -0.0003315, 0.0021473, -0.0022968, 0.0024292
1: 0.9914370, 0.9961960, 0.9913321, 0.9965813, -0.0051444, 0.0048640
2: -0.0080807, -0.0040301, -0.0081717, -0.0038591, -0.0042216, 0.0041416
3: 0.0025010, 0.0053125, 0.0022733, 0.0053745, -0.0028735, 0.0030392
4: 0.0016022, 0.0067838, 0.0014670, 0.0069545, -0.0053523, 0.0053168
5: 0.0030079, 0.0083333, 0.0025768, 0.0084506, -0.0054427, 0.0057565
6: -0.0041241, 0.0010457, -0.0042583, 0.0014443, -0.0055685, 0.0053039
7: -0.0088667, -0.0065887, -0.0089169, -0.0064043, -0.0024624, 0.0023281
8: 0.0035467, 0.0082106, 0.0033220, 0.0082349, -0.0046696, 0.0048812
9: -0.0050968, -0.0018440, -0.0051685, -0.0015806, -0.0035162, 0.0033245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029531, upper bound: 0.0028047
time: 1.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029531, upper bound: 0.0028047
time: 1.23 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002304, 0.0021051, -0.0002436, 0.0021335, -0.0023638, 0.0023487
1: 0.9914213, 0.9963670, 0.9913613, 0.9963951, -0.0049737, 0.0050057
2: -0.0081211, -0.0040047, -0.0081277, -0.0039068, -0.0042144, 0.0041230
3: 0.0023999, 0.0053217, 0.0023833, 0.0053572, -0.0029573, 0.0029384
4: 0.0015821, 0.0068092, 0.0015047, 0.0069069, -0.0053248, 0.0053045
5: 0.0028164, 0.0083507, 0.0027850, 0.0084179, -0.0056015, 0.0055657
6: -0.0041441, 0.0012227, -0.0042209, 0.0012518, -0.0053958, 0.0054436
7: -0.0088741, -0.0065068, -0.0089029, -0.0064934, -0.0023807, 0.0023961
8: 0.0035133, 0.0082213, 0.0033846, 0.0082231, -0.0046982, 0.0048203
9: -0.0051074, -0.0017270, -0.0051485, -0.0017078, -0.0033997, 0.0034215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029853
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029853
time: 1.37 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002215, 0.0021554, -0.0002379, 0.0021327, -0.0023542, 0.0023933
1: 0.9913147, 0.9963483, 0.9913630, 0.9963829, -0.0050682, 0.0049853
2: -0.0081167, -0.0038309, -0.0081249, -0.0039095, -0.0042071, 0.0042940
3: 0.0024110, 0.0053847, 0.0023905, 0.0053562, -0.0029452, 0.0029942
4: 0.0014448, 0.0069826, 0.0015069, 0.0069042, -0.0054594, 0.0054757
5: 0.0028375, 0.0084700, 0.0027987, 0.0084160, -0.0055786, 0.0056713
6: -0.0042804, 0.0012033, -0.0042187, 0.0012391, -0.0055195, 0.0054220
7: -0.0089251, -0.0065158, -0.0089021, -0.0064992, -0.0024259, 0.0023862
8: 0.0032850, 0.0082202, 0.0033883, 0.0082224, -0.0049243, 0.0048319
9: -0.0051803, -0.0017398, -0.0051473, -0.0017161, -0.0034641, 0.0034075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029853
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029853
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002304, 0.0021051, -0.0003315, 0.0021473, -0.0023776, 0.0024366
1: 0.9914213, 0.9963670, 0.9913321, 0.9965813, -0.0051600, 0.0050350
2: -0.0081211, -0.0040047, -0.0081717, -0.0038591, -0.0042621, 0.0041670
3: 0.0023999, 0.0053217, 0.0022733, 0.0053745, -0.0029746, 0.0030484
4: 0.0015821, 0.0068092, 0.0014670, 0.0069545, -0.0053724, 0.0053422
5: 0.0028164, 0.0083507, 0.0025768, 0.0084506, -0.0056342, 0.0057739
6: -0.0041441, 0.0012227, -0.0042583, 0.0014443, -0.0055884, 0.0054810
7: -0.0088741, -0.0065068, -0.0089169, -0.0064043, -0.0024698, 0.0024100
8: 0.0035133, 0.0082213, 0.0033220, 0.0082349, -0.0047034, 0.0048842
9: -0.0051074, -0.0017270, -0.0051685, -0.0015806, -0.0035269, 0.0034415

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029577, upper bound: 0.0030176
time: 1.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029577, upper bound: 0.0030176
time: 1.29 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002215, 0.0021554, -0.0003256, 0.0021464, -0.0023679, 0.0024810
1: 0.9913147, 0.9963483, 0.9913338, 0.9965687, -0.0052540, 0.0050145
2: -0.0081167, -0.0038309, -0.0081687, -0.0038619, -0.0042547, 0.0043379
3: 0.0024110, 0.0053847, 0.0022807, 0.0053734, -0.0029624, 0.0031039
4: 0.0014448, 0.0069826, 0.0014693, 0.0069517, -0.0055069, 0.0055133
5: 0.0028375, 0.0084700, 0.0025908, 0.0084487, -0.0056112, 0.0058792
6: -0.0042804, 0.0012033, -0.0042560, 0.0014314, -0.0057117, 0.0054593
7: -0.0089251, -0.0065158, -0.0089160, -0.0064103, -0.0025148, 0.0024002
8: 0.0032850, 0.0082202, 0.0033257, 0.0082341, -0.0049298, 0.0048944
9: -0.0051803, -0.0017398, -0.0051673, -0.0015892, -0.0035911, 0.0034275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029577, upper bound: 0.0030176
time: 1.07 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029577, upper bound: 0.0030176
time: 1.05 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003015, 0.0021431, -0.0001785, 0.0021020, -0.0024035, 0.0023216
1: 0.9913409, 0.9965178, 0.9914280, 0.9962574, -0.0049165, 0.0050898
2: -0.0081567, -0.0038735, -0.0080952, -0.0040155, -0.0041412, 0.0042217
3: 0.0023109, 0.0053693, 0.0024647, 0.0053178, -0.0030069, 0.0029045
4: 0.0014784, 0.0069402, 0.0015907, 0.0067984, -0.0053200, 0.0053495
5: 0.0026478, 0.0084408, 0.0029392, 0.0083433, -0.0056955, 0.0055015
6: -0.0042470, 0.0013786, -0.0041356, 0.0011092, -0.0053561, 0.0055142
7: -0.0089126, -0.0064347, -0.0088709, -0.0065593, -0.0023533, 0.0024363
8: 0.0033409, 0.0082308, 0.0035275, 0.0082144, -0.0048675, 0.0046835
9: -0.0051625, -0.0016240, -0.0051029, -0.0018020, -0.0033604, 0.0034789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029531
time: 1.24 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029531
time: 1.17 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0021944, -0.0001724, 0.0021011, -0.0023949, 0.0023668
1: 0.9912323, 0.9965016, 0.9914299, 0.9962444, -0.0050121, 0.0050716
2: -0.0081529, -0.0036964, -0.0080921, -0.0040187, -0.0041342, 0.0043957
3: 0.0023204, 0.0054334, 0.0024723, 0.0053167, -0.0029962, 0.0029611
4: 0.0013385, 0.0071168, 0.0015932, 0.0067953, -0.0054568, 0.0055237
5: 0.0026659, 0.0085623, 0.0029537, 0.0083412, -0.0056752, 0.0056086
6: -0.0043858, 0.0013619, -0.0041331, 0.0010958, -0.0054816, 0.0054950
7: -0.0089646, -0.0064424, -0.0088700, -0.0065655, -0.0023991, 0.0024276
8: 0.0031082, 0.0082298, 0.0035317, 0.0082136, -0.0050980, 0.0046982
9: -0.0052367, -0.0016351, -0.0051016, -0.0018108, -0.0034258, 0.0034666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029531
time: 1.18 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0028047, upper bound: 0.0029531
time: 1.11 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003015, 0.0021431, -0.0002602, 0.0021095, -0.0024110, 0.0024033
1: 0.9913409, 0.9965178, 0.9914120, 0.9964303, -0.0050894, 0.0051057
2: -0.0081567, -0.0038735, -0.0081361, -0.0039896, -0.0041671, 0.0042626
3: 0.0023109, 0.0053693, 0.0023625, 0.0053272, -0.0030163, 0.0030068
4: 0.0014784, 0.0069402, 0.0015702, 0.0068243, -0.0053459, 0.0053699
5: 0.0026478, 0.0084408, 0.0027457, 0.0083611, -0.0057133, 0.0056951
6: -0.0042470, 0.0013786, -0.0041559, 0.0012881, -0.0055351, 0.0055345
7: -0.0089126, -0.0064347, -0.0088785, -0.0064765, -0.0024361, 0.0024439
8: 0.0033409, 0.0082308, 0.0034935, 0.0082253, -0.0048707, 0.0047179
9: -0.0051625, -0.0016240, -0.0051138, -0.0016838, -0.0034787, 0.0034898

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029781, upper bound: 0.0029771
time: 1.62 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029781, upper bound: 0.0029771
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0021944, -0.0002541, 0.0021085, -0.0024024, 0.0024485
1: 0.9912323, 0.9965016, 0.9914141, 0.9964173, -0.0051850, 0.0050874
2: -0.0081529, -0.0036964, -0.0081330, -0.0039930, -0.0041599, 0.0044366
3: 0.0023204, 0.0054334, 0.0023701, 0.0053260, -0.0030055, 0.0030633
4: 0.0013385, 0.0071168, 0.0015729, 0.0068209, -0.0054825, 0.0055440
5: 0.0026659, 0.0085623, 0.0027601, 0.0083588, -0.0056929, 0.0058022
6: -0.0043858, 0.0013619, -0.0041533, 0.0012748, -0.0056606, 0.0055152
7: -0.0089646, -0.0064424, -0.0088776, -0.0064827, -0.0024819, 0.0024352
8: 0.0031082, 0.0082298, 0.0034979, 0.0082245, -0.0051015, 0.0047319
9: -0.0052367, -0.0016351, -0.0051124, -0.0016926, -0.0035441, 0.0034773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029781, upper bound: 0.0029771
time: 1.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029781, upper bound: 0.0029771
time: 2.30 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0003015, 0.0021431, -0.0002436, 0.0021335, -0.0024350, 0.0023867
1: 0.9913409, 0.9965178, 0.9913613, 0.9963951, -0.0050542, 0.0051565
2: -0.0081567, -0.0038735, -0.0081277, -0.0039068, -0.0042500, 0.0042543
3: 0.0023109, 0.0053693, 0.0023833, 0.0053572, -0.0030463, 0.0029860
4: 0.0014784, 0.0069402, 0.0015047, 0.0069069, -0.0054285, 0.0054355
5: 0.0026478, 0.0084408, 0.0027850, 0.0084179, -0.0057701, 0.0056558
6: -0.0042470, 0.0013786, -0.0042209, 0.0012518, -0.0054988, 0.0055995
7: -0.0089126, -0.0064347, -0.0089029, -0.0064934, -0.0024193, 0.0024682
8: 0.0033409, 0.0082308, 0.0033846, 0.0082231, -0.0048714, 0.0048254
9: -0.0051625, -0.0016240, -0.0051485, -0.0017078, -0.0034547, 0.0035245

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029531
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029531
time: 1.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0021944, -0.0002379, 0.0021327, -0.0024265, 0.0024322
1: 0.9912323, 0.9965016, 0.9913630, 0.9963829, -0.0051506, 0.0051385
2: -0.0081529, -0.0036964, -0.0081249, -0.0039095, -0.0042434, 0.0044284
3: 0.0023204, 0.0054334, 0.0023905, 0.0053562, -0.0030358, 0.0030429
4: 0.0013385, 0.0071168, 0.0015069, 0.0069042, -0.0055657, 0.0056099
5: 0.0026659, 0.0085623, 0.0027987, 0.0084160, -0.0057501, 0.0057636
6: -0.0043858, 0.0013619, -0.0042187, 0.0012391, -0.0056249, 0.0055806
7: -0.0089646, -0.0064424, -0.0089021, -0.0064992, -0.0024654, 0.0024596
8: 0.0031082, 0.0082298, 0.0033883, 0.0082224, -0.0051014, 0.0048416
9: -0.0052367, -0.0016351, -0.0051473, -0.0017161, -0.0035205, 0.0035123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029531
time: 1.50 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0027917, upper bound: 0.0029531
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003015, 0.0021431, -0.0003315, 0.0021473, -0.0024488, 0.0024746
1: 0.9913409, 0.9965178, 0.9913321, 0.9965813, -0.0052404, 0.0051857
2: -0.0081567, -0.0038735, -0.0081717, -0.0038591, -0.0042976, 0.0042982
3: 0.0023109, 0.0053693, 0.0022733, 0.0053745, -0.0030636, 0.0030959
4: 0.0014784, 0.0069402, 0.0014670, 0.0069545, -0.0054761, 0.0054731
5: 0.0026478, 0.0084408, 0.0025768, 0.0084506, -0.0058028, 0.0058640
6: -0.0042470, 0.0013786, -0.0042583, 0.0014443, -0.0056913, 0.0056369
7: -0.0089126, -0.0064347, -0.0089169, -0.0064043, -0.0025084, 0.0024822
8: 0.0033409, 0.0082308, 0.0033220, 0.0082349, -0.0048761, 0.0048899
9: -0.0051625, -0.0016240, -0.0051685, -0.0015806, -0.0035819, 0.0035445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029563, upper bound: 0.0029770
time: 1.36 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029563, upper bound: 0.0029770
time: 3.37 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002939, 0.0021944, -0.0003256, 0.0021464, -0.0024403, 0.0025199
1: 0.9912323, 0.9965016, 0.9913338, 0.9965687, -0.0053363, 0.0051677
2: -0.0081529, -0.0036964, -0.0081687, -0.0038619, -0.0042910, 0.0044723
3: 0.0023204, 0.0054334, 0.0022807, 0.0053734, -0.0030530, 0.0031526
4: 0.0013385, 0.0071168, 0.0014693, 0.0069517, -0.0056132, 0.0056475
5: 0.0026659, 0.0085623, 0.0025908, 0.0084487, -0.0057828, 0.0059715
6: -0.0043858, 0.0013619, -0.0042560, 0.0014314, -0.0058172, 0.0056179
7: -0.0089646, -0.0064424, -0.0089160, -0.0064103, -0.0025543, 0.0024736
8: 0.0031082, 0.0082298, 0.0033257, 0.0082341, -0.0051066, 0.0049041
9: -0.0052367, -0.0016351, -0.0051673, -0.0015892, -0.0036475, 0.0035322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029563, upper bound: 0.0029770
time: 1.14 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029563, upper bound: 0.0029770
time: 1.14 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001496, 0.0020977, -0.0002998, 0.0020857, -0.0022353, 0.0023975
1: 0.9914370, 0.9961960, 0.9914624, 0.9965141, -0.0050771, 0.0047336
2: -0.0080807, -0.0040301, -0.0081559, -0.0040717, -0.0040090, 0.0041257
3: 0.0025010, 0.0053125, 0.0023130, 0.0052974, -0.0027965, 0.0029995
4: 0.0016022, 0.0067838, 0.0016351, 0.0067424, -0.0051402, 0.0051488
5: 0.0030079, 0.0083333, 0.0026519, 0.0083048, -0.0052969, 0.0056814
6: -0.0041241, 0.0010457, -0.0040915, 0.0013748, -0.0054990, 0.0051372
7: -0.0088667, -0.0065887, -0.0088545, -0.0064364, -0.0024302, 0.0022657
8: 0.0035467, 0.0082106, 0.0036013, 0.0082306, -0.0046617, 0.0045967
9: -0.0050968, -0.0018440, -0.0050794, -0.0016265, -0.0034703, 0.0032354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029747, upper bound: 0.0028047
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029747, upper bound: 0.0028047
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001405, 0.0021509, -0.0002945, 0.0020849, -0.0022254, 0.0024455
1: 0.9913242, 0.9961767, 0.9914642, 0.9965029, -0.0051787, 0.0047125
2: -0.0080762, -0.0038464, -0.0081532, -0.0040746, -0.0040016, 0.0043068
3: 0.0025123, 0.0053791, 0.0023196, 0.0052964, -0.0027841, 0.0030594
4: 0.0014570, 0.0069672, 0.0016374, 0.0067395, -0.0052825, 0.0053298
5: 0.0030294, 0.0084594, 0.0026643, 0.0083028, -0.0052734, 0.0057950
6: -0.0042682, 0.0010258, -0.0040893, 0.0013633, -0.0056315, 0.0051151
7: -0.0089206, -0.0065979, -0.0088536, -0.0064418, -0.0024788, 0.0022557
8: 0.0033053, 0.0082094, 0.0036052, 0.0082299, -0.0049002, 0.0046042
9: -0.0051738, -0.0018571, -0.0050782, -0.0016341, -0.0035397, 0.0032211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029747, upper bound: 0.0028047
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029747, upper bound: 0.0028047
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0001496, 0.0020977, -0.0003735, 0.0020856, -0.0022352, 0.0024712
1: 0.9914370, 0.9961960, 0.9914626, 0.9966701, -0.0052332, 0.0047334
2: -0.0080807, -0.0040301, -0.0081927, -0.0040720, -0.0040087, 0.0041626
3: 0.0025010, 0.0053125, 0.0022208, 0.0052973, -0.0027964, 0.0030917
4: 0.0016022, 0.0067838, 0.0016353, 0.0067420, -0.0051398, 0.0051485
5: 0.0030079, 0.0083333, 0.0024773, 0.0083045, -0.0052966, 0.0058560
6: -0.0041241, 0.0010457, -0.0040913, 0.0015363, -0.0056605, 0.0051369
7: -0.0088667, -0.0065887, -0.0088544, -0.0063617, -0.0025050, 0.0022656
8: 0.0035467, 0.0082106, 0.0036018, 0.0082405, -0.0046651, 0.0045971
9: -0.0050968, -0.0018440, -0.0050792, -0.0015198, -0.0035770, 0.0032353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030704, upper bound: 0.0028047
time: 1.45 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030704, upper bound: 0.0028047
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0001405, 0.0021509, -0.0003679, 0.0020847, -0.0022252, 0.0025189
1: 0.9913242, 0.9961767, 0.9914646, 0.9966583, -0.0053341, 0.0047122
2: -0.0080762, -0.0038464, -0.0081899, -0.0040752, -0.0040010, 0.0043436
3: 0.0025123, 0.0053791, 0.0022278, 0.0052962, -0.0027839, 0.0031513
4: 0.0014570, 0.0069672, 0.0016378, 0.0067389, -0.0052819, 0.0053294
5: 0.0030294, 0.0084594, 0.0024904, 0.0083024, -0.0052730, 0.0059690
6: -0.0042682, 0.0010258, -0.0040888, 0.0015242, -0.0057924, 0.0051146
7: -0.0089206, -0.0065979, -0.0088534, -0.0063674, -0.0025532, 0.0022556
8: 0.0033053, 0.0082094, 0.0036059, 0.0082397, -0.0049036, 0.0046034
9: -0.0051738, -0.0018571, -0.0050779, -0.0015278, -0.0036460, 0.0032209

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030704, upper bound: 0.0028047
time: 1.25 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030704, upper bound: 0.0028047
time: 1.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0002304, 0.0021051, -0.0002998, 0.0020857, -0.0023161, 0.0024049
1: 0.9914213, 0.9963670, 0.9914624, 0.9965141, -0.0050927, 0.0049046
2: -0.0081211, -0.0040047, -0.0081559, -0.0040717, -0.0040494, 0.0041511
3: 0.0023999, 0.0053217, 0.0023130, 0.0052974, -0.0028976, 0.0030087
4: 0.0015821, 0.0068092, 0.0016351, 0.0067424, -0.0051602, 0.0051741
5: 0.0028164, 0.0083507, 0.0026519, 0.0083048, -0.0054884, 0.0056988
6: -0.0041441, 0.0012227, -0.0040915, 0.0013748, -0.0055189, 0.0053143
7: -0.0088741, -0.0065068, -0.0088545, -0.0064364, -0.0024377, 0.0023476
8: 0.0035133, 0.0082213, 0.0036013, 0.0082306, -0.0046955, 0.0046004
9: -0.0051074, -0.0017270, -0.0050794, -0.0016265, -0.0034809, 0.0033524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029747, upper bound: 0.0029938
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029747, upper bound: 0.0029938
time: 1.27 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0002215, 0.0021554, -0.0002945, 0.0020849, -0.0023064, 0.0024500
1: 0.9913147, 0.9963483, 0.9914642, 0.9965029, -0.0051882, 0.0048841
2: -0.0081167, -0.0038309, -0.0081532, -0.0040746, -0.0040421, 0.0043223
3: 0.0024110, 0.0053847, 0.0023196, 0.0052964, -0.0028854, 0.0030651
4: 0.0014448, 0.0069826, 0.0016374, 0.0067395, -0.0052947, 0.0053452
5: 0.0028375, 0.0084700, 0.0026643, 0.0083028, -0.0054653, 0.0058056
6: -0.0042804, 0.0012033, -0.0040893, 0.0013633, -0.0056437, 0.0052925
7: -0.0089251, -0.0065158, -0.0088536, -0.0064418, -0.0024834, 0.0023378
8: 0.0032850, 0.0082202, 0.0036052, 0.0082299, -0.0049219, 0.0046150
9: -0.0051803, -0.0017398, -0.0050782, -0.0016341, -0.0035462, 0.0033383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029748, upper bound: 0.0029938
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029748, upper bound: 0.0029938
time: 1.97 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0002304, 0.0021051, -0.0003735, 0.0020856, -0.0023160, 0.0024786
1: 0.9914213, 0.9963670, 0.9914626, 0.9966701, -0.0052488, 0.0049044
2: -0.0081211, -0.0040047, -0.0081927, -0.0040720, -0.0040491, 0.0041880
3: 0.0023999, 0.0053217, 0.0022208, 0.0052973, -0.0028975, 0.0031009
4: 0.0015821, 0.0068092, 0.0016353, 0.0067420, -0.0051599, 0.0051739
5: 0.0028164, 0.0083507, 0.0024773, 0.0083045, -0.0054881, 0.0058734
6: -0.0041441, 0.0012227, -0.0040913, 0.0015363, -0.0056804, 0.0053140
7: -0.0088741, -0.0065068, -0.0088544, -0.0063617, -0.0025124, 0.0023475
8: 0.0035133, 0.0082213, 0.0036018, 0.0082405, -0.0046987, 0.0046023
9: -0.0051074, -0.0017270, -0.0050792, -0.0015198, -0.0035876, 0.0033523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030657, upper bound: 0.0030275
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030657, upper bound: 0.0030275
time: 1.34 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0002215, 0.0021554, -0.0003679, 0.0020847, -0.0023062, 0.0025234
1: 0.9913147, 0.9963483, 0.9914646, 0.9966583, -0.0053436, 0.0048838
2: -0.0081167, -0.0038309, -0.0081899, -0.0040752, -0.0040415, 0.0043590
3: 0.0024110, 0.0053847, 0.0022278, 0.0052962, -0.0028852, 0.0031569
4: 0.0014448, 0.0069826, 0.0016378, 0.0067389, -0.0052942, 0.0053448
5: 0.0028375, 0.0084700, 0.0024904, 0.0083024, -0.0054649, 0.0059796
6: -0.0042804, 0.0012033, -0.0040888, 0.0015242, -0.0058045, 0.0052921
7: -0.0089251, -0.0065158, -0.0088534, -0.0063674, -0.0025578, 0.0023376
8: 0.0032850, 0.0082202, 0.0036059, 0.0082397, -0.0049254, 0.0046142
9: -0.0051803, -0.0017398, -0.0050779, -0.0015278, -0.0036525, 0.0033381

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030657, upper bound: 0.0030275
time: 1.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0030657, upper bound: 0.0030275
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0001496, 0.0020977, -0.0003576, 0.0021115, -0.0022610, 0.0024554
1: 0.9914370, 0.9961960, 0.9914078, 0.9966366, -0.0051996, 0.0047883
2: -0.0080807, -0.0040301, -0.0081848, -0.0039827, -0.0040980, 0.0041546
3: 0.0025010, 0.0053125, 0.0022406, 0.0053297, -0.0028287, 0.0030719
4: 0.0016022, 0.0067838, 0.0015647, 0.0068312, -0.0052290, 0.0052191
5: 0.0030079, 0.0083333, 0.0025148, 0.0083658, -0.0053579, 0.0058184
6: -0.0041241, 0.0010457, -0.0041613, 0.0015016, -0.0056257, 0.0052070
7: -0.0088667, -0.0065887, -0.0088806, -0.0063778, -0.0024889, 0.0022919
8: 0.0035467, 0.0082106, 0.0034844, 0.0082383, -0.0046688, 0.0047156
9: -0.0050968, -0.0018440, -0.0051167, -0.0015428, -0.0035540, 0.0032727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 97
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 196
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 137
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 239

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029677, upper bound: 0.0028047
time: 1.33 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -0.0029677, upper bound: 0.0028047
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0001405, 0.0021509, -0.0003522, 0.0021107, -0.0022512, 0.0025031
1: 0.9913242, 0.9961767, 0.9914094, 0.9966252, -0.0053009, 0.0047673
2: -0.0080762, -0.0038464, -0.0081821, -0.0039853, -0.0040908, 0.0043357
3: 0.0025123, 0.0053791, 0.0022474, 0.0053287, -0.0028165, 0.0031316
4: 0.0014570, 0.0069672, 0.0015668, 0.0068285, -0.0053715, 0.0054004
5: 0.0030294, 0.0084594, 0.0025276, 0.0083640, -0.0053347, 0.0059317
6: -0.0042682, 0.0010258, -0.0041593, 0.0014898, -0.0057580, 0.0051851
7: -0.0089206, -0.0065979, -0.0088798, -0.0063833, -0.0025373, 0.0022819
8: 0.0033053, 0.0082094, 0.0034879, 0.0082376, -0.0049072, 0.0047215
9: -0.0051738, -0.0018571, -0.0051156, -0.0015506, -0.0036232, 0.0032585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=9, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.24 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.13 + 597.17 = 600.30 seconds
