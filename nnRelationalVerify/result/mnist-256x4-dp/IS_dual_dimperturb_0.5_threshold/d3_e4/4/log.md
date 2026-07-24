## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 4)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00913976


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=19, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=27, inp2_unstable=27, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0028527, 0.0215212, 0.0028527, 0.0215212, -0.0186686, 0.0186686)
1: (-0.0046181, 0.0039611, -0.0046181, 0.0039611, -0.0085791, 0.0085791)
2: (0.0000760, 0.0127827, 0.0000760, 0.0127827, -0.0127067, 0.0127067)
3: (-0.0065579, 0.0040404, -0.0065579, 0.0040404, -0.0105983, 0.0105983)
4: (-0.0032518, 0.0020554, -0.0032518, 0.0020554, -0.0053072, 0.0053072)
5: (-0.0022887, 0.0064380, -0.0022887, 0.0064380, -0.0087267, 0.0087267)
6: (-0.0169698, 0.0032436, -0.0169698, 0.0032436, -0.0202134, 0.0202134)
7: (-0.0136335, 0.0160977, -0.0136335, 0.0160977, -0.0297312, 0.0297312)
8: (0.9820704, 1.0015050, 0.9820704, 1.0015050, -0.0194346, 0.0194346)
9: (-0.0163897, 0.0016074, -0.0163897, 0.0016074, -0.0179971, 0.0179971)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.72 + 2.29 = 4.01 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0117634, upper bound: 0.0117633

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 247

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0113746, upper bound: 0.0114387
time: 1.41 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0115459, upper bound: 0.0115460
time: 1.44 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 3.03 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 8, lower bound: -0.0113746, upper bound: 0.0114387
IS_A2, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 8, lower bound: -0.0115459, upper bound: 0.0115460

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0032198, 0.0179315, 0.0030026, 0.0200589, -0.0168392, 0.0149290
1: -0.0028645, 0.0033814, -0.0039119, 0.0037501, -0.0066146, 0.0072932
2: 0.0027210, 0.0125797, 0.0011535, 0.0126998, -0.0099788, 0.0114262
3: -0.0060847, 0.0023517, -0.0063854, 0.0033577, -0.0094424, 0.0087371
4: -0.0027110, 0.0020368, -0.0030288, 0.0020481, -0.0047591, 0.0050655
5: -0.0008425, 0.0062229, -0.0016911, 0.0063501, -0.0071927, 0.0079140
6: -0.0160568, 0.0023903, -0.0166023, 0.0028951, -0.0189520, 0.0189926
7: -0.0104196, 0.0160026, -0.0123215, 0.0160606, -0.0264802, 0.0283240
8: 0.9848322, 1.0012064, 0.9832236, 1.0013866, -0.0165544, 0.0179828
9: -0.0163288, -0.0004719, -0.0163659, 0.0007522, -0.0170810, 0.0158941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108075, upper bound: 0.0106436
time: 1.21 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0111156, upper bound: 0.0112005
time: 1.30 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0030302, 0.0198553, 0.0029626, 0.0207148, -0.0176846, 0.0168926
1: -0.0037487, 0.0037480, -0.0042165, 0.0038476, -0.0075963, 0.0079644
2: 0.0013153, 0.0126846, 0.0006623, 0.0127219, -0.0114066, 0.0120222
3: -0.0064387, 0.0032527, -0.0064647, 0.0036444, -0.0100832, 0.0097173
4: -0.0029972, 0.0021226, -0.0031117, 0.0020508, -0.0050480, 0.0052344
5: -0.0016396, 0.0063340, -0.0019561, 0.0063735, -0.0080131, 0.0082900
6: -0.0168734, 0.0028310, -0.0167607, 0.0029880, -0.0198613, 0.0195916
7: -0.0121227, 0.0164417, -0.0128114, 0.0160746, -0.0281973, 0.0292530
8: 0.9833894, 1.0016363, 0.9827573, 1.0014373, -0.0180479, 0.0188789
9: -0.0166096, 0.0006219, -0.0163749, 0.0010793, -0.0176889, 0.0169968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 94

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0109836, upper bound: 0.0107112
time: 1.29 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0112927, upper bound: 0.0112927
time: 1.29 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.34 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 8, lower bound: -0.0108075, upper bound: 0.0106436
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 8, lower bound: -0.0111156, upper bound: 0.0112005
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 8, lower bound: -0.0109836, upper bound: 0.0107112
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.34
Output dim: 8, lower bound: -0.0112927, upper bound: 0.0112927

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0035726, 0.0153808, 0.0038137, 0.0140840, -0.0105114, 0.0115671
1: -0.0014319, 0.0027814, -0.0005601, 0.0028167, -0.0042486, 0.0033415
2: 0.0046915, 0.0123847, 0.0057546, 0.0122513, -0.0075598, 0.0066301
3: -0.0056040, 0.0009449, -0.0057435, 0.0000929, -0.0056970, 0.0066884
4: -0.0022639, 0.0020297, -0.0020520, 0.0021807, -0.0042908, 0.0039071
5: 0.0001197, 0.0060162, 0.0005924, 0.0058750, -0.0057553, 0.0054239
6: -0.0153786, 0.0015703, -0.0155283, 0.0010098, -0.0161804, 0.0168708
7: -0.0076633, 0.0159666, -0.0061464, 0.0167388, -0.0239098, 0.0214286
8: 0.9859064, 1.0009955, 0.9864441, 1.0014154, -0.0149416, 0.0139187
9: -0.0163059, -0.0021439, -0.0167996, -0.0028677, -0.0128817, 0.0141756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1

### Relational analysis result of IS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105112, upper bound: 0.0102946
time: 1.37 seconds

## Relational analysis of IS_A1_B1_B2

### Relational analysis result of IS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0105112, upper bound: 0.0103547
time: 1.29 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0032448, 0.0178258, 0.0031620, 0.0192962, -0.0160513, 0.0146638
1: -0.0028094, 0.0033593, -0.0035293, 0.0036228, -0.0064322, 0.0068886
2: 0.0028007, 0.0125659, 0.0017103, 0.0126117, -0.0098110, 0.0108556
3: -0.0060668, 0.0022940, -0.0062809, 0.0029589, -0.0090257, 0.0085749
4: -0.0026857, 0.0020363, -0.0028635, 0.0020433, -0.0047290, 0.0046295
5: -0.0008003, 0.0062082, -0.0013834, 0.0062567, -0.0070570, 0.0075917
6: -0.0160302, 0.0023321, -0.0164061, 0.0025246, -0.0185548, 0.0187382
7: -0.0102730, 0.0160001, -0.0113767, 0.0160358, -0.0263088, 0.0262675
8: 0.9849423, 1.0011976, 0.9839782, 1.0013200, -0.0163777, 0.0172194
9: -0.0163272, -0.0005637, -0.0163501, 0.0001437, -0.0156931, 0.0157864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106800, upper bound: 0.0109443
time: 1.99 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0108937, upper bound: 0.0109950
time: 1.21 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0034011, 0.0171139, 0.0037828, 0.0145621, -0.0111610, 0.0133311
1: -0.0022852, 0.0032056, -0.0008478, 0.0028169, -0.0051022, 0.0040534
2: 0.0033979, 0.0124795, 0.0053710, 0.0122685, -0.0088706, 0.0071085
3: -0.0060005, 0.0017953, -0.0057444, 0.0003558, -0.0063563, 0.0075397
4: -0.0025127, 0.0021112, -0.0020892, 0.0021817, -0.0045792, 0.0042005
5: -0.0005713, 0.0061167, 0.0004072, 0.0058931, -0.0064644, 0.0057095
6: -0.0161080, 0.0019690, -0.0156673, 0.0010818, -0.0171898, 0.0176363
7: -0.0092263, 0.0163834, -0.0064691, 0.0167437, -0.0256705, 0.0228525
8: 0.9855239, 1.0013990, 0.9863750, 1.0014606, -0.0159366, 0.0150240
9: -0.0165723, -0.0012054, -0.0168027, -0.0027371, -0.0138353, 0.0152853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106757, upper bound: 0.0103536
time: 1.14 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106757, upper bound: 0.0104101
time: 1.22 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0030533, 0.0197241, 0.0031182, 0.0199250, -0.0168716, 0.0166059
1: -0.0036834, 0.0037260, -0.0038356, 0.0037249, -0.0074083, 0.0075616
2: 0.0014130, 0.0126718, 0.0012357, 0.0126359, -0.0112229, 0.0114361
3: -0.0064208, 0.0031867, -0.0063639, 0.0032483, -0.0096691, 0.0095506
4: -0.0029714, 0.0021219, -0.0029489, 0.0020461, -0.0050174, 0.0049098
5: -0.0015868, 0.0063204, -0.0016361, 0.0062824, -0.0078692, 0.0079565
6: -0.0168380, 0.0027772, -0.0165636, 0.0026264, -0.0194644, 0.0193408
7: -0.0119735, 0.0164378, -0.0118834, 0.0160501, -0.0280236, 0.0278056
8: 0.9835119, 1.0016248, 0.9835194, 1.0013705, -0.0178587, 0.0181054
9: -0.0166071, 0.0005258, -0.0163592, 0.0004811, -0.0166855, 0.0168850

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=7, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A2_B2_B1

### Relational analysis result of IS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110455, upper bound: 0.0108409
time: 1.35 seconds

## Relational analysis of IS_A2_B2_B2

### Relational analysis result of IS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0110783, upper bound: 0.0110783
time: 1.34 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.43 seconds
IS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0105112, upper bound: 0.0102946
IS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0105112, upper bound: 0.0103547
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0106800, upper bound: 0.0109443
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0108937, upper bound: 0.0109950
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0106757, upper bound: 0.0103536
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0106757, upper bound: 0.0104101
IS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0110455, upper bound: 0.0108409
IS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 4.43
Output dim: 8, lower bound: -0.0110783, upper bound: 0.0110783

## BFS IS instance: IS_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0037803, 0.0152965, 0.0043131, 0.0138685, -0.0100882, 0.0109833
1: -0.0014085, 0.0027798, -0.0005019, 0.0028126, -0.0042211, 0.0032817
2: 0.0047410, 0.0122698, 0.0058909, 0.0119752, -0.0072342, 0.0063789
3: -0.0055976, 0.0008564, -0.0057273, -0.0001482, -0.0054494, 0.0065837
4: -0.0021360, 0.0020228, -0.0017412, 0.0021632, -0.0041258, 0.0035285
5: 0.0001635, 0.0058946, 0.0006933, 0.0055824, -0.0054189, 0.0052012
6: -0.0152883, 0.0010875, -0.0153286, -0.0001509, -0.0146802, 0.0161213
7: -0.0070265, 0.0159310, -0.0045708, 0.0166492, -0.0230978, 0.0195593
8: 0.9863696, 1.0009497, 0.9875577, 1.0013121, -0.0142858, 0.0125113
9: -0.0162831, -0.0025618, -0.0167423, -0.0038839, -0.0116493, 0.0136386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102007, upper bound: 0.0098611
time: 1.27 seconds

## Relational analysis of IS_A1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102565, upper bound: 0.0100036
time: 1.49 seconds

## BFS IS instance: IS_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0037721, 0.0153148, 0.0042238, 0.0150405, -0.0112684, 0.0110909
1: -0.0014185, 0.0027796, -0.0008339, 0.0028528, -0.0042712, 0.0036135
2: 0.0047266, 0.0122744, 0.0050714, 0.0120246, -0.0072980, 0.0072030
3: -0.0055967, 0.0008556, -0.0058863, 0.0001160, -0.0057126, 0.0067419
4: -0.0021389, 0.0020217, -0.0018113, 0.0023353, -0.0044742, 0.0035999
5: 0.0001567, 0.0058994, 0.0001695, 0.0056347, -0.0054780, 0.0057299
6: -0.0152905, 0.0011066, -0.0162916, 0.0000566, -0.0148438, 0.0173982
7: -0.0070312, 0.0159258, -0.0050297, 0.0175291, -0.0245603, 0.0201319
8: 0.9863513, 1.0009475, 0.9873586, 1.0020379, -0.0156866, 0.0126627
9: -0.0162797, -0.0025529, -0.0173049, -0.0036470, -0.0118973, 0.0147520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of IS_A1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102007, upper bound: 0.0099286
time: 1.29 seconds

## Relational analysis of IS_A1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102565, upper bound: 0.0100539
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0036839, 0.0152324, 0.0033497, 0.0181812, -0.0144973, 0.0118827
1: -0.0013737, 0.0027802, -0.0029360, 0.0034158, -0.0047896, 0.0057162
2: 0.0047849, 0.0123231, 0.0025487, 0.0125079, -0.0077230, 0.0097745
3: -0.0055989, 0.0008834, -0.0061134, 0.0023674, -0.0079663, 0.0069968
4: -0.0021963, 0.0020242, -0.0026478, 0.0020386, -0.0041125, 0.0043498
5: 0.0001874, 0.0059510, -0.0009465, 0.0061468, -0.0059594, 0.0068975
6: -0.0152994, 0.0013116, -0.0161074, 0.0020885, -0.0173878, 0.0174190
7: -0.0073215, 0.0159384, -0.0101173, 0.0160119, -0.0229394, 0.0246498
8: 0.9861545, 1.0009562, 0.9849942, 1.0012240, -0.0146675, 0.0159620
9: -0.0162878, -0.0023648, -0.0163348, -0.0006608, -0.0146513, 0.0135888

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103194, upper bound: 0.0106482
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103902, upper bound: 0.0106482
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0035065, 0.0167823, 0.0032913, 0.0187262, -0.0152197, 0.0134909
1: -0.0022209, 0.0031553, -0.0032402, 0.0035195, -0.0057404, 0.0063955
2: 0.0036200, 0.0124212, 0.0021326, 0.0125402, -0.0089202, 0.0102886
3: -0.0060019, 0.0017198, -0.0061969, 0.0026580, -0.0086599, 0.0079166
4: -0.0024315, 0.0021672, -0.0027324, 0.0020404, -0.0043644, 0.0047252
5: -0.0004340, 0.0060550, -0.0011568, 0.0061810, -0.0066150, 0.0072118
6: -0.0161348, 0.0017240, -0.0162542, 0.0022240, -0.0183588, 0.0179782
7: -0.0088176, 0.0166696, -0.0106290, 0.0160211, -0.0245735, 0.0266892
8: 0.9857590, 1.0015651, 0.9845694, 1.0012698, -0.0155108, 0.0169957
9: -0.0167554, -0.0014884, -0.0163407, -0.0003360, -0.0159462, 0.0145568

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0102105, upper bound: 0.0099717
time: 1.23 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0106621, upper bound: 0.0107699
time: 1.25 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0036011, 0.0170288, 0.0042749, 0.0143417, -0.0107406, 0.0127538
1: -0.0022591, 0.0031815, -0.0007870, 0.0028128, -0.0050719, 0.0039685
2: 0.0034504, 0.0123689, 0.0055047, 0.0119964, -0.0085460, 0.0068642
3: -0.0059757, 0.0017091, -0.0057281, 0.0001225, -0.0060982, 0.0074372
4: -0.0023879, 0.0021036, -0.0017877, 0.0021640, -0.0044130, 0.0038290
5: -0.0005287, 0.0059995, 0.0005126, 0.0056048, -0.0061335, 0.0054870
6: -0.0160137, 0.0015040, -0.0154565, -0.0000621, -0.0159516, 0.0169605
7: -0.0086027, 0.0163446, -0.0049505, 0.0166535, -0.0248676, 0.0212893
8: 0.9859700, 1.0013494, 0.9874724, 1.0013528, -0.0153828, 0.0137160
9: -0.0165475, -0.0016161, -0.0167450, -0.0037205, -0.0126454, 0.0147427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101967, upper bound: 0.0100295
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104200, upper bound: 0.0100478
time: 1.32 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0035962, 0.0170464, 0.0041781, 0.0157223, -0.0121261, 0.0128683
1: -0.0022683, 0.0031856, -0.0012204, 0.0028529, -0.0051213, 0.0044061
2: 0.0034366, 0.0123716, 0.0045449, 0.0120499, -0.0086133, 0.0078267
3: -0.0059786, 0.0017057, -0.0058870, 0.0004796, -0.0064582, 0.0075927
4: -0.0023857, 0.0021031, -0.0018650, 0.0023360, -0.0047217, 0.0039123
5: -0.0005352, 0.0060024, -0.0000915, 0.0056615, -0.0061967, 0.0060938
6: -0.0160156, 0.0015154, -0.0164713, 0.0001630, -0.0161786, 0.0179867
7: -0.0085839, 0.0163416, -0.0054762, 0.0175330, -0.0261169, 0.0218179
8: 0.9859591, 1.0013480, 0.9872565, 1.0020949, -0.0161358, 0.0138861
9: -0.0165456, -0.0016226, -0.0173075, -0.0034601, -0.0129345, 0.0156848

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101967, upper bound: 0.0100759
time: 1.23 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104200, upper bound: 0.0100957
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0032421, 0.0186468, 0.0035866, 0.0172375, -0.0139954, 0.0150601
1: -0.0031094, 0.0035288, -0.0024051, 0.0032160, -0.0063254, 0.0059339
2: 0.0022333, 0.0125674, 0.0032674, 0.0123769, -0.0101436, 0.0093000
3: -0.0062607, 0.0026136, -0.0059517, 0.0018185, -0.0080792, 0.0085652
4: -0.0027550, 0.0021168, -0.0024142, 0.0020341, -0.0047891, 0.0042717
5: -0.0011629, 0.0062098, -0.0005841, 0.0060080, -0.0071709, 0.0067939
6: -0.0165347, 0.0023384, -0.0158410, 0.0015377, -0.0180724, 0.0181794
7: -0.0107242, 0.0164118, -0.0087930, 0.0159891, -0.0267133, 0.0240660
8: 0.9845282, 1.0015287, 0.9859377, 1.0011359, -0.0166078, 0.0155911
9: -0.0165905, -0.0002730, -0.0163202, -0.0015025, -0.0142919, 0.0160473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103841, upper bound: 0.0104962
time: 1.40 seconds

## Relational analysis of IS_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0103841, upper bound: 0.0107088
time: 1.41 seconds

## BFS IS instance: IS_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0031822, 0.0191326, 0.0033670, 0.0188288, -0.0156466, 0.0157656
1: -0.0033776, 0.0036191, -0.0032331, 0.0035800, -0.0069576, 0.0068522
2: 0.0018582, 0.0126005, 0.0020816, 0.0124983, -0.0106401, 0.0105189
3: -0.0063337, 0.0028652, -0.0063469, 0.0026731, -0.0090069, 0.0092121
4: -0.0028387, 0.0021187, -0.0027010, 0.0021785, -0.0050172, 0.0045502
5: -0.0013482, 0.0062449, -0.0012306, 0.0061366, -0.0074849, 0.0074755
6: -0.0166707, 0.0024777, -0.0166478, 0.0020481, -0.0187188, 0.0191255
7: -0.0112135, 0.0164218, -0.0104957, 0.0167274, -0.0279409, 0.0258114
8: 0.9841193, 1.0015707, 0.9846769, 1.0017411, -0.0176218, 0.0168938
9: -0.0165969, 0.0000404, -0.0167923, -0.0004291, -0.0153839, 0.0168328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=7, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104003, upper bound: 0.0107242
time: 1.32 seconds

## Relational analysis of IS_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0104003, upper bound: 0.0108363
time: 1.37 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 4.43 seconds
IS_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0102007, upper bound: 0.0098611
IS_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0102565, upper bound: 0.0100036
IS_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0102007, upper bound: 0.0099286
IS_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0102565, upper bound: 0.0100539
IS_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0103194, upper bound: 0.0106482
IS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0103902, upper bound: 0.0106482
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0102105, upper bound: 0.0099717
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0106621, upper bound: 0.0107699
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0101967, upper bound: 0.0100295
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0104200, upper bound: 0.0100478
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0101967, upper bound: 0.0100759
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0104200, upper bound: 0.0100957
IS_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0103841, upper bound: 0.0104962
IS_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0103841, upper bound: 0.0107088
IS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0104003, upper bound: 0.0107242
IS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 4.43
Output dim: 8, lower bound: -0.0104003, upper bound: 0.0108363

## BFS IS instance: IS_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0039504, 0.0143239, 0.0046966, 0.0119910, -0.0079908, 0.0096273
1: -0.0008423, 0.0027787, 0.0007290, 0.0028098, -0.0036521, 0.0020497
2: 0.0054979, 0.0121758, 0.0073550, 0.0117632, -0.0062654, 0.0048207
3: -0.0055933, 0.0003036, -0.0057162, -0.0013718, -0.0040108, 0.0060197
4: -0.0019929, 0.0020181, -0.0014195, 0.0021511, -0.0037774, 0.0030688
5: 0.0005336, 0.0057949, 0.0014065, 0.0053578, -0.0048242, 0.0042597
6: -0.0150013, 0.0006921, -0.0147125, -0.0010422, -0.0129307, 0.0141036
7: -0.0060282, 0.0159071, -0.0023497, 0.0165873, -0.0209744, 0.0164631
8: 0.9867489, 1.0008525, 0.9884127, 1.0010918, -0.0129796, 0.0111716
9: -0.0162678, -0.0030476, -0.0167027, -0.0049771, -0.0100910, 0.0124734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094396, upper bound: 0.0087173
time: 1.18 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099575, upper bound: 0.0096002
time: 1.42 seconds

## BFS IS instance: IS_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0039145, 0.0147926, 0.0045777, 0.0130185, -0.0091039, 0.0102148
1: -0.0011304, 0.0027791, 0.0001112, 0.0028339, -0.0039643, 0.0026680
2: 0.0051317, 0.0121956, 0.0065909, 0.0118290, -0.0066972, 0.0056047
3: -0.0055948, 0.0005565, -0.0058116, -0.0007815, -0.0047027, 0.0063681
4: -0.0020331, 0.0020197, -0.0015375, 0.0022544, -0.0041127, 0.0031915
5: 0.0003537, 0.0058159, 0.0009868, 0.0054274, -0.0050737, 0.0048292
6: -0.0151435, 0.0007756, -0.0153577, -0.0007659, -0.0133533, 0.0155160
7: -0.0063598, 0.0159156, -0.0032410, 0.0171157, -0.0228738, 0.0174906
8: 0.9866688, 1.0008991, 0.9881476, 1.0015430, -0.0141599, 0.0114548
9: -0.0162732, -0.0029081, -0.0170406, -0.0045696, -0.0105211, 0.0135842

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094467, upper bound: 0.0087493
time: 1.29 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099942, upper bound: 0.0097223
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0039425, 0.0143412, 0.0045889, 0.0130198, -0.0090773, 0.0097523
1: -0.0008523, 0.0027785, 0.0003787, 0.0028497, -0.0037020, 0.0023997
2: 0.0054840, 0.0121802, 0.0066303, 0.0118228, -0.0063388, 0.0055499
3: -0.0055923, 0.0003029, -0.0058741, -0.0010657, -0.0043956, 0.0061770
4: -0.0019956, 0.0020171, -0.0015061, 0.0023221, -0.0041388, 0.0031339
5: 0.0005268, 0.0057995, 0.0009522, 0.0054209, -0.0048941, 0.0048473
6: -0.0150032, 0.0007105, -0.0156397, -0.0007918, -0.0130688, 0.0158887
7: -0.0060319, 0.0159019, -0.0029194, 0.0174618, -0.0227712, 0.0170107
8: 0.9867312, 1.0008503, 0.9881726, 1.0018005, -0.0144231, 0.0112982
9: -0.0162645, -0.0030410, -0.0172619, -0.0046843, -0.0103192, 0.0136510

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094396, upper bound: 0.0087295
time: 1.18 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099575, upper bound: 0.0096710
time: 2.44 seconds

## BFS IS instance: IS_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0039096, 0.0148096, 0.0045094, 0.0140433, -0.0101337, 0.0103002
1: -0.0011399, 0.0027789, -0.0001996, 0.0028698, -0.0040097, 0.0029785
2: 0.0051183, 0.0121983, 0.0058716, 0.0118667, -0.0067485, 0.0063267
3: -0.0055939, 0.0005579, -0.0059537, -0.0005272, -0.0050573, 0.0065116
4: -0.0020344, 0.0020187, -0.0015972, 0.0024083, -0.0044161, 0.0032522
5: 0.0003473, 0.0058188, 0.0005352, 0.0054674, -0.0051202, 0.0052836
6: -0.0151454, 0.0007870, -0.0162303, -0.0006071, -0.0134797, 0.0170172
7: -0.0063622, 0.0159104, -0.0036767, 0.0179025, -0.0242647, 0.0179949
8: 0.9866579, 1.0008968, 0.9879954, 1.0021985, -0.0154241, 0.0115706
9: -0.0162699, -0.0029038, -0.0175437, -0.0043657, -0.0107331, 0.0145733

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094467, upper bound: 0.0087559
time: 1.27 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099942, upper bound: 0.0097814
time: 1.25 seconds

## BFS IS instance: IS_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0041807, 0.0150149, 0.0035565, 0.0180998, -0.0139191, 0.0114584
1: -0.0013125, 0.0027759, -0.0029092, 0.0033884, -0.0047009, 0.0056851
2: 0.0049115, 0.0120485, 0.0026014, 0.0123936, -0.0074821, 0.0094471
3: -0.0055823, 0.0006630, -0.0060861, 0.0022901, -0.0078724, 0.0067491
4: -0.0018876, 0.0020062, -0.0025207, 0.0020312, -0.0037573, 0.0041843
5: 0.0003024, 0.0056600, -0.0009019, 0.0060256, -0.0057233, 0.0065619
6: -0.0150686, 0.0001570, -0.0159989, 0.0016076, -0.0166762, 0.0160414
7: -0.0057660, 0.0158461, -0.0094876, 0.0159740, -0.0212011, 0.0238695
8: 0.9872622, 1.0008383, 0.9854687, 1.0011691, -0.0133174, 0.0153696
9: -0.0162288, -0.0033733, -0.0163105, -0.0010766, -0.0141196, 0.0124332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096374, upper bound: 0.0096025
time: 1.19 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100673, upper bound: 0.0104168
time: 1.21 seconds

## BFS IS instance: IS_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0040833, 0.0164875, 0.0035535, 0.0181194, -0.0140361, 0.0129339
1: -0.0017581, 0.0028108, -0.0029189, 0.0033916, -0.0051497, 0.0057297
2: 0.0038984, 0.0121023, 0.0025869, 0.0123952, -0.0084968, 0.0095154
3: -0.0057201, 0.0010469, -0.0060879, 0.0022963, -0.0080164, 0.0071348
4: -0.0019727, 0.0021554, -0.0025221, 0.0020300, -0.0038367, 0.0045532
5: -0.0003460, 0.0057171, -0.0009093, 0.0060274, -0.0063734, 0.0066264
6: -0.0161336, 0.0003833, -0.0160008, 0.0016146, -0.0177481, 0.0162173
7: -0.0063821, 0.0166094, -0.0094943, 0.0159683, -0.0218579, 0.0256548
8: 0.9870451, 1.0015284, 0.9854530, 1.0011662, -0.0134814, 0.0160754
9: -0.0167169, -0.0030833, -0.0163069, -0.0010711, -0.0153084, 0.0127104

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098917, upper bound: 0.0103106
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098917, upper bound: 0.0104135
time: 1.55 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039425, 0.0153611, 0.0043686, 0.0152004, -0.0112579, 0.0109926
1: -0.0013391, 0.0028102, -0.0010237, 0.0028319, -0.0041710, 0.0038339
2: 0.0047304, 0.0121802, 0.0049056, 0.0119446, -0.0072142, 0.0072746
3: -0.0057177, 0.0007357, -0.0058039, 0.0002251, -0.0059429, 0.0065396
4: -0.0020327, 0.0021528, -0.0017379, 0.0022461, -0.0039422, 0.0034535
5: 0.0001225, 0.0057996, 0.0001320, 0.0055499, -0.0054275, 0.0056675
6: -0.0156896, 0.0007106, -0.0160844, -0.0002798, -0.0141399, 0.0155665
7: -0.0064595, 0.0165961, -0.0047562, 0.0170730, -0.0220975, 0.0194518
8: 0.9867311, 1.0013984, 0.9876813, 1.0017607, -0.0137193, 0.0121260
9: -0.0167084, -0.0029013, -0.0170133, -0.0038803, -0.0114210, 0.0130325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098270, upper bound: 0.0096192
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098931, upper bound: 0.0096192
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0035065, 0.0167823, 0.0034336, 0.0180855, -0.0145790, 0.0133487
1: -0.0022209, 0.0031553, -0.0028952, 0.0033921, -0.0056129, 0.0060504
2: 0.0036200, 0.0124212, 0.0026219, 0.0124615, -0.0088416, 0.0097993
3: -0.0060019, 0.0017198, -0.0060913, 0.0022956, -0.0082975, 0.0078110
4: -0.0024315, 0.0021672, -0.0025855, 0.0020341, -0.0043587, 0.0041131
5: -0.0004340, 0.0060550, -0.0009188, 0.0060977, -0.0065317, 0.0069738
6: -0.0161348, 0.0017240, -0.0160665, 0.0018934, -0.0173047, 0.0177905
7: -0.0088176, 0.0166696, -0.0097795, 0.0159892, -0.0245445, 0.0234340
8: 0.9857590, 1.0015651, 0.9852242, 1.0012020, -0.0154430, 0.0158252
9: -0.0167554, -0.0014884, -0.0163203, -0.0008733, -0.0138720, 0.0145382

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 126

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100073, upper bound: 0.0104080
time: 1.35 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100073, upper bound: 0.0105234
time: 1.50 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0040363, 0.0146205, 0.0044391, 0.0134955, -0.0094592, 0.0101814
1: -0.0008960, 0.0027957, -0.0002639, 0.0028117, -0.0037077, 0.0030596
2: 0.0053072, 0.0121283, 0.0061796, 0.0119056, -0.0065984, 0.0059487
3: -0.0056604, 0.0003482, -0.0057238, -0.0004013, -0.0052591, 0.0060687
4: -0.0019470, 0.0020907, -0.0016504, 0.0021594, -0.0037345, 0.0035030
5: 0.0003946, 0.0057446, 0.0008387, 0.0055086, -0.0051140, 0.0049059
6: -0.0152988, 0.0004925, -0.0152015, -0.0004437, -0.0141876, 0.0144918
7: -0.0058358, 0.0162785, -0.0040051, 0.0166295, -0.0208122, 0.0192739
8: 0.9869404, 1.0011092, 0.9878386, 1.0012631, -0.0129728, 0.0124313
9: -0.0165053, -0.0031945, -0.0167297, -0.0041873, -0.0115529, 0.0123380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093635, upper bound: 0.0087503
time: 1.20 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099481, upper bound: 0.0097574
time: 1.52 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0038607, 0.0159603, 0.0044093, 0.0138399, -0.0099792, 0.0115509
1: -0.0016224, 0.0029290, -0.0004845, 0.0028120, -0.0044344, 0.0034136
2: 0.0042953, 0.0122254, 0.0059053, 0.0119221, -0.0076268, 0.0063201
3: -0.0058842, 0.0010746, -0.0057251, -0.0002011, -0.0056832, 0.0067997
4: -0.0021306, 0.0022507, -0.0016826, 0.0021608, -0.0039678, 0.0038308
5: -0.0001466, 0.0058475, 0.0007036, 0.0055261, -0.0056727, 0.0051439
6: -0.0161441, 0.0009007, -0.0153079, -0.0003744, -0.0156603, 0.0153389
7: -0.0071178, 0.0170965, -0.0042682, 0.0166369, -0.0224209, 0.0211075
8: 0.9865488, 1.0017666, 0.9877721, 1.0012996, -0.0137900, 0.0136149
9: -0.0170283, -0.0025464, -0.0167345, -0.0040750, -0.0126378, 0.0131620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095635, upper bound: 0.0087503
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101600, upper bound: 0.0097676
time: 1.64 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0040265, 0.0146347, 0.0043378, 0.0147859, -0.0107594, 0.0102969
1: -0.0009050, 0.0027955, -0.0006899, 0.0028517, -0.0037567, 0.0034854
2: 0.0052953, 0.0121337, 0.0052562, 0.0119616, -0.0066664, 0.0068776
3: -0.0056598, 0.0003407, -0.0058823, -0.0000526, -0.0056072, 0.0062229
4: -0.0019484, 0.0020901, -0.0017302, 0.0023309, -0.0040985, 0.0037275
5: 0.0003895, 0.0057504, 0.0002722, 0.0055680, -0.0051785, 0.0054781
6: -0.0152993, 0.0005154, -0.0161963, -0.0002082, -0.0150157, 0.0164554
7: -0.0058280, 0.0162751, -0.0045438, 0.0175069, -0.0226191, 0.0206838
8: 0.9869184, 1.0011073, 0.9876126, 1.0019970, -0.0144749, 0.0131499
9: -0.0165031, -0.0031918, -0.0172907, -0.0039173, -0.0123069, 0.0135241

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0089829, upper bound: 0.0091721
time: 1.24 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099481, upper bound: 0.0098097
time: 1.28 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038613, 0.0159691, 0.0043204, 0.0151287, -0.0112674, 0.0116486
1: -0.0016294, 0.0029322, -0.0008920, 0.0028522, -0.0044815, 0.0038242
2: 0.0042868, 0.0122251, 0.0049937, 0.0119712, -0.0076844, 0.0072313
3: -0.0058859, 0.0010637, -0.0058839, 0.0001377, -0.0060236, 0.0069476
4: -0.0021269, 0.0022496, -0.0017540, 0.0023327, -0.0043321, 0.0040035
5: -0.0001491, 0.0058471, 0.0001364, 0.0055782, -0.0057273, 0.0057106
6: -0.0161376, 0.0008993, -0.0163005, -0.0001679, -0.0159698, 0.0171997
7: -0.0070947, 0.0170907, -0.0047575, 0.0175159, -0.0242101, 0.0218482
8: 0.9865501, 1.0017612, 0.9875739, 1.0020342, -0.0153184, 0.0141873
9: -0.0170246, -0.0025586, -0.0172965, -0.0038347, -0.0131900, 0.0143483

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095635, upper bound: 0.0087559
time: 1.22 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101600, upper bound: 0.0098231
time: 1.26 seconds

## BFS IS instance: IS_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0040056, 0.0132246, 0.0035866, 0.0172375, -0.0132319, 0.0096380
1: 0.0000457, 0.0028318, -0.0024051, 0.0032160, -0.0031703, 0.0052369
2: 0.0064406, 0.0121453, 0.0032674, 0.0123769, -0.0059363, 0.0088778
3: -0.0058034, -0.0004632, -0.0059517, 0.0018185, -0.0076219, 0.0054885
4: -0.0018935, 0.0022456, -0.0024142, 0.0020341, -0.0037127, 0.0046598
5: 0.0008751, 0.0057626, -0.0005841, 0.0060080, -0.0051329, 0.0063467
6: -0.0154765, 0.0005639, -0.0158410, 0.0015377, -0.0170142, 0.0160273
7: -0.0050691, 0.0170704, -0.0087930, 0.0159891, -0.0201140, 0.0258634
8: 0.9868719, 1.0015477, 0.9859377, 1.0011359, -0.0135416, 0.0156100
9: -0.0170116, -0.0034055, -0.0163202, -0.0015025, -0.0155091, 0.0122240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100295, upper bound: 0.0101967
time: 1.61 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100759, upper bound: 0.0101967
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0033737, 0.0179459, 0.0035866, 0.0172375, -0.0138639, 0.0143593
1: -0.0027620, 0.0033902, -0.0024051, 0.0032160, -0.0059780, 0.0057953
2: 0.0027584, 0.0124947, 0.0032674, 0.0123769, -0.0096185, 0.0092272
3: -0.0061477, 0.0022548, -0.0059517, 0.0018185, -0.0079662, 0.0082065
4: -0.0026139, 0.0021125, -0.0024142, 0.0020341, -0.0042828, 0.0042676
5: -0.0008895, 0.0061328, -0.0005841, 0.0060080, -0.0068975, 0.0067169
6: -0.0163391, 0.0020327, -0.0158410, 0.0015377, -0.0178768, 0.0178736
7: -0.0099114, 0.0163900, -0.0087930, 0.0159891, -0.0242573, 0.0240452
8: 0.9851890, 1.0014647, 0.9859377, 1.0011359, -0.0159470, 0.0155271
9: -0.0165766, -0.0007925, -0.0163202, -0.0015025, -0.0142786, 0.0144029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095217, upper bound: 0.0097291
time: 1.37 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101118, upper bound: 0.0104586
time: 1.34 seconds

## BFS IS instance: IS_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0039838, 0.0135400, 0.0033670, 0.0188288, -0.0148450, 0.0101730
1: -0.0001573, 0.0028323, -0.0032331, 0.0035800, -0.0037373, 0.0060654
2: 0.0061948, 0.0121573, 0.0020816, 0.0124983, -0.0063035, 0.0100757
3: -0.0058052, -0.0002808, -0.0063469, 0.0026731, -0.0084783, 0.0060661
4: -0.0019208, 0.0022475, -0.0027010, 0.0021785, -0.0040387, 0.0049484
5: 0.0007578, 0.0057754, -0.0012306, 0.0061366, -0.0053789, 0.0070059
6: -0.0155881, 0.0006146, -0.0166478, 0.0020481, -0.0176362, 0.0172624
7: -0.0053207, 0.0170801, -0.0104957, 0.0167274, -0.0219154, 0.0275758
8: 0.9868233, 1.0015869, 0.9846769, 1.0017411, -0.0147402, 0.0169100
9: -0.0170178, -0.0033086, -0.0167923, -0.0004291, -0.0165888, 0.0132990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100477, upper bound: 0.0104200
time: 1.85 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100957, upper bound: 0.0104200
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0033094, 0.0184461, 0.0033670, 0.0188288, -0.0155194, 0.0150791
1: -0.0030382, 0.0034916, -0.0032331, 0.0035800, -0.0066182, 0.0067247
2: 0.0023723, 0.0125302, 0.0020816, 0.0124983, -0.0101260, 0.0104486
3: -0.0062296, 0.0025193, -0.0063469, 0.0026731, -0.0089027, 0.0088662
4: -0.0027004, 0.0021146, -0.0027010, 0.0021785, -0.0046501, 0.0045462
5: -0.0010798, 0.0061704, -0.0012306, 0.0061366, -0.0072164, 0.0074010
6: -0.0164824, 0.0021820, -0.0166478, 0.0020481, -0.0185305, 0.0188298
7: -0.0104198, 0.0164008, -0.0104957, 0.0167274, -0.0262458, 0.0257913
8: 0.9847677, 1.0015093, 0.9846769, 1.0017411, -0.0169733, 0.0168324
9: -0.0165834, -0.0004660, -0.0167923, -0.0004291, -0.0153711, 0.0156682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095290, upper bound: 0.0098061
time: 1.32 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0101237, upper bound: 0.0105972
time: 1.23 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.37 seconds
IS_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0094396, upper bound: 0.0087173
IS_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0099575, upper bound: 0.0096002
IS_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0094467, upper bound: 0.0087493
IS_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0099942, upper bound: 0.0097223
IS_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0094396, upper bound: 0.0087295
IS_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0099575, upper bound: 0.0096710
IS_A1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0094467, upper bound: 0.0087559
IS_A1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0099942, upper bound: 0.0097814
IS_A1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0096374, upper bound: 0.0096025
IS_A1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0100673, upper bound: 0.0104168
IS_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0098917, upper bound: 0.0103106
IS_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0098917, upper bound: 0.0104135
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0098270, upper bound: 0.0096192
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0098931, upper bound: 0.0096192
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0100073, upper bound: 0.0104080
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0100073, upper bound: 0.0105234
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0093635, upper bound: 0.0087503
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0099481, upper bound: 0.0097574
IS_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0095635, upper bound: 0.0087503
IS_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0101600, upper bound: 0.0097676
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0089829, upper bound: 0.0091721
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0099481, upper bound: 0.0098097
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0095635, upper bound: 0.0087559
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0101600, upper bound: 0.0098231
IS_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0100295, upper bound: 0.0101967
IS_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0100759, upper bound: 0.0101967
IS_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0095217, upper bound: 0.0097291
IS_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0101118, upper bound: 0.0104586
IS_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0100477, upper bound: 0.0104200
IS_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0100957, upper bound: 0.0104200
IS_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0095290, upper bound: 0.0098061
IS_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 4.37
Output dim: 8, lower bound: -0.0101237, upper bound: 0.0105972

## BFS IS instance: IS_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0043709, 0.0130173, 0.0056435, 0.0107016, -0.0052092, 0.0071095
1: 0.0000223, 0.0027755, 0.0021376, 0.0028684, -0.0028461, 0.0005116
2: 0.0065141, 0.0119433, 0.0084432, 0.0112397, -0.0046712, 0.0028800
3: -0.0055806, -0.0006443, -0.0059481, -0.0030558, -0.0020248, 0.0049521
4: -0.0016683, 0.0020044, -0.0007289, 0.0024022, -0.0033996, 0.0021919
5: 0.0010632, 0.0055486, 0.0018401, 0.0048031, -0.0035158, 0.0030515
6: -0.0145531, -0.0002851, -0.0149994, -0.0032431, -0.0095331, 0.0121076
7: -0.0039352, 0.0158369, 0.0018601, 0.0178712, -0.0185363, 0.0112089
8: 0.9876863, 1.0006857, 0.9905241, 1.0018027, -0.0116155, 0.0082871
9: -0.0162229, -0.0041406, -0.0175237, -0.0072857, -0.0071673, 0.0112005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B1_B1_B1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090441, upper bound: 0.0084062
time: 1.31 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092528, upper bound: 0.0085450
time: 1.35 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0039504, 0.0143239, 0.0048407, 0.0115796, -0.0073830, 0.0094306
1: -0.0008423, 0.0027787, 0.0010269, 0.0028082, -0.0036505, 0.0016813
2: 0.0054979, 0.0121758, 0.0076772, 0.0116835, -0.0061857, 0.0044309
3: -0.0055933, 0.0003036, -0.0057099, -0.0016935, -0.0032691, 0.0060135
4: -0.0019929, 0.0020181, -0.0013097, 0.0021444, -0.0037716, 0.0025451
5: 0.0005336, 0.0057949, 0.0015645, 0.0052734, -0.0045442, 0.0040183
6: -0.0150013, 0.0006921, -0.0145513, -0.0013772, -0.0111556, 0.0139077
7: -0.0060282, 0.0159071, -0.0016487, 0.0165528, -0.0209447, 0.0136050
8: 0.9867489, 1.0008525, 0.9887341, 1.0010285, -0.0129046, 0.0093910
9: -0.0162678, -0.0030476, -0.0166806, -0.0053462, -0.0083648, 0.0124544

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B1_B1_B2_B1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096537, upper bound: 0.0093570
time: 1.26 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097794, upper bound: 0.0094243
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0043424, 0.0134531, 0.0055593, 0.0108432, -0.0057926, 0.0077301
1: -0.0002581, 0.0027759, 0.0021080, 0.0028853, -0.0031435, 0.0005514
2: 0.0061772, 0.0119591, 0.0083610, 0.0112862, -0.0051091, 0.0032074
3: -0.0055821, -0.0004092, -0.0060152, -0.0029902, -0.0021226, 0.0055571
4: -0.0017053, 0.0020060, -0.0007827, 0.0024748, -0.0037701, 0.0022780
5: 0.0008907, 0.0055653, 0.0017609, 0.0048524, -0.0037738, 0.0033887
6: -0.0146887, -0.0002189, -0.0152842, -0.0030474, -0.0100212, 0.0134097
7: -0.0042495, 0.0158451, 0.0015733, 0.0182428, -0.0206215, 0.0116620
8: 0.9876228, 1.0007312, 0.9903364, 1.0020684, -0.0128556, 0.0086361
9: -0.0162281, -0.0040103, -0.0177613, -0.0071087, -0.0074497, 0.0124266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B1_B2_B1_B1

### Relational analysis result of IS_A1_B1_B1_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090532, upper bound: 0.0084262
time: 1.65 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_B2

### Relational analysis result of IS_A1_B1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092600, upper bound: 0.0085773
time: 1.38 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0039145, 0.0147926, 0.0047234, 0.0125802, -0.0086657, 0.0100691
1: -0.0011304, 0.0027791, 0.0004087, 0.0028323, -0.0039627, 0.0023380
2: 0.0051317, 0.0121956, 0.0069248, 0.0117484, -0.0066167, 0.0052709
3: -0.0055948, 0.0005565, -0.0058055, -0.0011003, -0.0039764, 0.0063620
4: -0.0020331, 0.0020197, -0.0014265, 0.0022478, -0.0041071, 0.0026786
5: 0.0003537, 0.0058159, 0.0011541, 0.0053421, -0.0048335, 0.0046221
6: -0.0151435, 0.0007756, -0.0152018, -0.0011046, -0.0116345, 0.0153379
7: -0.0063598, 0.0159156, -0.0025509, 0.0170819, -0.0228450, 0.0146656
8: 0.9866688, 1.0008991, 0.9884726, 1.0014807, -0.0140907, 0.0097192
9: -0.0162732, -0.0029081, -0.0170190, -0.0049425, -0.0088289, 0.0135658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B1_B2_B2_B1

### Relational analysis result of IS_A1_B1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096802, upper bound: 0.0094803
time: 1.32 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_B2

### Relational analysis result of IS_A1_B1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098167, upper bound: 0.0095534
time: 1.39 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0043612, 0.0130345, 0.0055900, 0.0109090, -0.0059183, 0.0071876
1: 0.0000124, 0.0027753, 0.0021299, 0.0028983, -0.0028859, 0.0005199
2: 0.0065008, 0.0119487, 0.0083286, 0.0112693, -0.0047185, 0.0032721
3: -0.0055796, -0.0006456, -0.0060667, -0.0030252, -0.0020576, 0.0053352
4: -0.0016729, 0.0020033, -0.0007620, 0.0025305, -0.0038304, 0.0022275
5: 0.0010567, 0.0055542, 0.0017186, 0.0048344, -0.0035577, 0.0034669
6: -0.0145548, -0.0002627, -0.0154814, -0.0031187, -0.0096739, 0.0137557
7: -0.0039427, 0.0158316, 0.0016906, 0.0185277, -0.0207151, 0.0113909
8: 0.9876649, 1.0006831, 0.9904047, 1.0022651, -0.0131967, 0.0084174
9: -0.0162195, -0.0041264, -0.0179434, -0.0071774, -0.0072836, 0.0126072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B2_B1_B1_B1

### Relational analysis result of IS_A1_B1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090441, upper bound: 0.0083839
time: 1.24 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_B2

### Relational analysis result of IS_A1_B1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092527, upper bound: 0.0085452
time: 2.05 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0039425, 0.0143412, 0.0047282, 0.0125112, -0.0085687, 0.0095458
1: -0.0008523, 0.0027785, 0.0007016, 0.0028481, -0.0037004, 0.0020584
2: 0.0054840, 0.0121802, 0.0070120, 0.0117457, -0.0062618, 0.0051682
3: -0.0055923, 0.0003029, -0.0058679, -0.0013991, -0.0036591, 0.0061707
4: -0.0019956, 0.0020171, -0.0013971, 0.0023153, -0.0041330, 0.0026292
5: 0.0005268, 0.0057995, 0.0011442, 0.0053393, -0.0046079, 0.0046553
6: -0.0150032, 0.0007105, -0.0154540, -0.0011157, -0.0113803, 0.0156484
7: -0.0060319, 0.0159019, -0.0022209, 0.0174272, -0.0227417, 0.0142215
8: 0.9867312, 1.0008503, 0.9884832, 1.0017271, -0.0143364, 0.0095996
9: -0.0162645, -0.0030410, -0.0172397, -0.0050516, -0.0086545, 0.0136322

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B2_B1_B2_B1

### Relational analysis result of IS_A1_B1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096537, upper bound: 0.0093919
time: 2.09 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_B2

### Relational analysis result of IS_A1_B1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097795, upper bound: 0.0094799
time: 1.24 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0043394, 0.0134696, 0.0055284, 0.0114062, -0.0069020, 0.0078041
1: -0.0002675, 0.0027757, 0.0018452, 0.0029118, -0.0031793, 0.0008783
2: 0.0061647, 0.0119607, 0.0079825, 0.0113034, -0.0051387, 0.0039135
3: -0.0055812, -0.0004084, -0.0061199, -0.0027405, -0.0024429, 0.0057115
4: -0.0017042, 0.0020050, -0.0008248, 0.0025882, -0.0041184, 0.0023390
5: 0.0008842, 0.0055671, 0.0014924, 0.0048705, -0.0038132, 0.0039513
6: -0.0146903, -0.0002119, -0.0158826, -0.0029755, -0.0101532, 0.0149854
7: -0.0042265, 0.0158401, 0.0012126, 0.0188226, -0.0223620, 0.0121516
8: 0.9876162, 1.0007288, 0.9902674, 1.0025336, -0.0142088, 0.0087577
9: -0.0162249, -0.0040156, -0.0181320, -0.0069604, -0.0076622, 0.0135622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B2_B2_B1_B1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090532, upper bound: 0.0083966
time: 1.39 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092600, upper bound: 0.0085695
time: 1.33 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0039096, 0.0148096, 0.0046436, 0.0135084, -0.0095988, 0.0101659
1: -0.0011399, 0.0027789, 0.0001308, 0.0028682, -0.0040081, 0.0026481
2: 0.0051183, 0.0121983, 0.0062713, 0.0117925, -0.0066742, 0.0059270
3: -0.0055939, 0.0005579, -0.0059476, -0.0008705, -0.0043203, 0.0065055
4: -0.0020344, 0.0020187, -0.0014920, 0.0024016, -0.0044106, 0.0027581
5: 0.0003473, 0.0058188, 0.0007430, 0.0053888, -0.0048954, 0.0050758
6: -0.0151454, 0.0007870, -0.0160452, -0.0009191, -0.0118530, 0.0168322
7: -0.0063622, 0.0159104, -0.0029859, 0.0178684, -0.0242306, 0.0152466
8: 0.9866579, 1.0008968, 0.9882947, 1.0021240, -0.0153401, 0.0099223
9: -0.0162699, -0.0029038, -0.0175219, -0.0047206, -0.0091009, 0.0145553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_B2_B2_B2_B1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096802, upper bound: 0.0095006
time: 1.37 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098167, upper bound: 0.0095926
time: 3.23 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0045968, 0.0136889, 0.0046206, 0.0146418, -0.0100450, 0.0090684
1: -0.0004622, 0.0027726, -0.0007124, 0.0028297, -0.0032919, 0.0034850
2: 0.0059677, 0.0118184, 0.0053293, 0.0118053, -0.0058376, 0.0064891
3: -0.0055691, -0.0003046, -0.0057950, -0.0001432, -0.0052278, 0.0054076
4: -0.0015653, 0.0019919, -0.0015635, 0.0022364, -0.0034084, 0.0029337
5: 0.0008199, 0.0054162, 0.0003549, 0.0054023, -0.0045497, 0.0050260
6: -0.0146374, -0.0008103, -0.0158550, -0.0008655, -0.0118473, 0.0136027
7: -0.0036610, 0.0157733, -0.0037460, 0.0170237, -0.0189172, 0.0166165
8: 0.9881902, 1.0006807, 0.9882432, 1.0016673, -0.0119789, 0.0102043
9: -0.0161822, -0.0044597, -0.0169818, -0.0044589, -0.0097093, 0.0112537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092993, upper bound: 0.0093112
time: 3.66 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094492, upper bound: 0.0094072
time: 1.20 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0041807, 0.0150149, 0.0037006, 0.0174656, -0.0132850, 0.0113143
1: -0.0013125, 0.0027759, -0.0025655, 0.0032546, -0.0045671, 0.0053379
2: 0.0049115, 0.0120485, 0.0030880, 0.0123139, -0.0074024, 0.0089605
3: -0.0055823, 0.0006630, -0.0059754, 0.0019305, -0.0072256, 0.0066384
4: -0.0018876, 0.0020062, -0.0023699, 0.0020248, -0.0037516, 0.0035474
5: 0.0003024, 0.0056600, -0.0006688, 0.0059412, -0.0056389, 0.0063289
6: -0.0150686, 0.0001570, -0.0158141, 0.0012727, -0.0148220, 0.0156543
7: -0.0057660, 0.0158461, -0.0086246, 0.0159416, -0.0211721, 0.0204626
8: 0.9872622, 1.0008383, 0.9861289, 1.0011022, -0.0132019, 0.0133692
9: -0.0162288, -0.0033733, -0.0162898, -0.0016230, -0.0119549, 0.0124146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097967, upper bound: 0.0102050
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098957, upper bound: 0.0102218
time: 1.33 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0047652, 0.0117497, 0.0035535, 0.0181194, -0.0133541, 0.0081961
1: 0.0012037, 0.0028482, -0.0029189, 0.0033916, -0.0021879, 0.0057670
2: 0.0076349, 0.0117253, 0.0025869, 0.0123952, -0.0047603, 0.0091384
3: -0.0058681, -0.0018406, -0.0060879, 0.0022963, -0.0081644, 0.0042473
4: -0.0013390, 0.0023156, -0.0025221, 0.0020300, -0.0029438, 0.0048377
5: 0.0014319, 0.0053176, -0.0009093, 0.0060274, -0.0045955, 0.0062269
6: -0.0151837, -0.0012018, -0.0160008, 0.0016146, -0.0167983, 0.0137766
7: -0.0016913, 0.0174286, -0.0094943, 0.0159683, -0.0155591, 0.0269229
8: 0.9885658, 1.0016469, 0.9854530, 1.0011662, -0.0111784, 0.0161939
9: -0.0172406, -0.0052587, -0.0163069, -0.0010711, -0.0161696, 0.0096629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A2_A1_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0087490, upper bound: 0.0095515
time: 1.29 seconds

## Relational analysis of IS_A1_B2_A1_A2_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096249, upper bound: 0.0100768
time: 1.24 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0042336, 0.0158745, 0.0035535, 0.0181194, -0.0138858, 0.0123210
1: -0.0014460, 0.0028101, -0.0029189, 0.0033916, -0.0048376, 0.0057290
2: 0.0043636, 0.0120192, 0.0025869, 0.0123952, -0.0080316, 0.0094323
3: -0.0057174, 0.0007024, -0.0060879, 0.0022963, -0.0080137, 0.0067903
4: -0.0018577, 0.0021525, -0.0025221, 0.0020300, -0.0033373, 0.0045502
5: -0.0001125, 0.0056290, -0.0009093, 0.0060274, -0.0061399, 0.0065383
6: -0.0159702, 0.0000340, -0.0160008, 0.0016146, -0.0175847, 0.0150299
7: -0.0056258, 0.0165944, -0.0094943, 0.0159683, -0.0190746, 0.0256391
8: 0.9873802, 1.0014735, 0.9854530, 1.0011662, -0.0119381, 0.0160205
9: -0.0167073, -0.0034706, -0.0163069, -0.0010711, -0.0152984, 0.0110588

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=18, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090592, upper bound: 0.0094193
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096249, upper bound: 0.0101791
time: 1.39 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0044344, 0.0151764, 0.0045648, 0.0151170, -0.0106826, 0.0106116
1: -0.0012928, 0.0028058, -0.0010008, 0.0028302, -0.0041230, 0.0038065
2: 0.0048422, 0.0119082, 0.0049573, 0.0118361, -0.0069939, 0.0069509
3: -0.0057003, 0.0005159, -0.0057968, 0.0001347, -0.0058245, 0.0063127
4: -0.0017300, 0.0021339, -0.0016188, 0.0022384, -0.0035923, 0.0032885
5: 0.0002161, 0.0055114, 0.0001755, 0.0054350, -0.0052189, 0.0053359
6: -0.0154712, -0.0004328, -0.0159972, -0.0007358, -0.0133707, 0.0141524
7: -0.0049367, 0.0164993, -0.0041630, 0.0170339, -0.0203872, 0.0186366
8: 0.9878281, 1.0012847, 0.9881188, 1.0017157, -0.0123971, 0.0114552
9: -0.0166465, -0.0038896, -0.0169883, -0.0042679, -0.0108849, 0.0118958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094617, upper bound: 0.0093173
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096282, upper bound: 0.0094197
time: 1.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0043775, 0.0163130, 0.0045692, 0.0151346, -0.0107571, 0.0117438
1: -0.0015693, 0.0028379, -0.0010090, 0.0028300, -0.0043993, 0.0038469
2: 0.0040743, 0.0119397, 0.0049444, 0.0118337, -0.0077594, 0.0069952
3: -0.0058276, 0.0007638, -0.0057961, 0.0001359, -0.0059635, 0.0065599
4: -0.0017816, 0.0022718, -0.0016130, 0.0022376, -0.0036562, 0.0036083
5: -0.0003006, 0.0055447, 0.0001682, 0.0054324, -0.0057330, 0.0053765
6: -0.0163545, -0.0003005, -0.0160022, -0.0007461, -0.0151831, 0.0143099
7: -0.0053238, 0.0172043, -0.0041162, 0.0170298, -0.0208785, 0.0202302
8: 0.9877012, 1.0018805, 0.9881287, 1.0017149, -0.0125425, 0.0128193
9: -0.0170972, -0.0037123, -0.0169857, -0.0042883, -0.0119251, 0.0121165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095318, upper bound: 0.0093173
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096833, upper bound: 0.0094197
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042860, 0.0119145, 0.0034336, 0.0180855, -0.0137995, 0.0084809
1: 0.0009172, 0.0028361, -0.0028952, 0.0033921, -0.0024749, 0.0057313
2: 0.0074872, 0.0119903, 0.0026219, 0.0124615, -0.0049744, 0.0093684
3: -0.0058205, -0.0013505, -0.0060913, 0.0022956, -0.0081161, 0.0047407
4: -0.0016568, 0.0022641, -0.0025855, 0.0020341, -0.0033635, 0.0045692
5: 0.0013841, 0.0055983, -0.0009188, 0.0060977, -0.0047136, 0.0065171
6: -0.0150683, -0.0000878, -0.0160665, 0.0018934, -0.0167887, 0.0151911
7: -0.0034519, 0.0171649, -0.0097795, 0.0159892, -0.0178292, 0.0257794
8: 0.9874971, 1.0014842, 0.9852242, 1.0012020, -0.0126406, 0.0162600
9: -0.0170721, -0.0042091, -0.0163203, -0.0008733, -0.0153647, 0.0110443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096425, upper bound: 0.0101123
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097127, upper bound: 0.0101123
time: 1.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036401, 0.0162265, 0.0034336, 0.0180855, -0.0144455, 0.0127930
1: -0.0019186, 0.0030108, -0.0028952, 0.0033921, -0.0053107, 0.0059060
2: 0.0040408, 0.0123474, 0.0026219, 0.0124615, -0.0084207, 0.0097255
3: -0.0058851, 0.0014022, -0.0060913, 0.0022956, -0.0081807, 0.0074935
4: -0.0022968, 0.0021641, -0.0025855, 0.0020341, -0.0038259, 0.0041104
5: -0.0002208, 0.0059767, -0.0009188, 0.0060977, -0.0063185, 0.0068955
6: -0.0159938, 0.0014135, -0.0160665, 0.0018934, -0.0169840, 0.0166827
7: -0.0080363, 0.0166538, -0.0097795, 0.0159892, -0.0216796, 0.0234202
8: 0.9860568, 1.0015160, 0.9852242, 1.0012020, -0.0139770, 0.0157292
9: -0.0167453, -0.0019745, -0.0163203, -0.0008733, -0.0138631, 0.0127296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A2_B2_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091192, upper bound: 0.0100626
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091192, upper bound: 0.0103429
time: 1.42 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044514, 0.0133596, 0.0054057, 0.0111089, -0.0056543, 0.0076991
1: -0.0000802, 0.0027923, 0.0018011, 0.0028708, -0.0029510, 0.0009483
2: 0.0062749, 0.0118988, 0.0081517, 0.0113712, -0.0050227, 0.0032266
3: -0.0056471, -0.0005700, -0.0059577, -0.0026494, -0.0026784, 0.0050161
4: -0.0016265, 0.0020764, -0.0009029, 0.0024126, -0.0033566, 0.0025823
5: 0.0009013, 0.0055014, 0.0016641, 0.0049424, -0.0038461, 0.0032208
6: -0.0148633, -0.0004724, -0.0152190, -0.0026903, -0.0109161, 0.0121337
7: -0.0037746, 0.0162051, 0.0007988, 0.0179243, -0.0184087, 0.0134015
8: 0.9878660, 1.0009472, 0.9899939, 1.0018992, -0.0114922, 0.0096073
9: -0.0164583, -0.0042734, -0.0175577, -0.0067040, -0.0084584, 0.0110677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092977, upper bound: 0.0087503
time: 1.34 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092977, upper bound: 0.0087503
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0040363, 0.0146205, 0.0045760, 0.0130352, -0.0089989, 0.0099032
1: -0.0008960, 0.0027957, 0.0000388, 0.0028102, -0.0037062, 0.0027569
2: 0.0053072, 0.0121283, 0.0065340, 0.0118299, -0.0065227, 0.0055943
3: -0.0056604, 0.0003482, -0.0057177, -0.0007229, -0.0045693, 0.0060635
4: -0.0019470, 0.0020907, -0.0015449, 0.0021528, -0.0037288, 0.0029877
5: 0.0003946, 0.0057446, 0.0010178, 0.0054285, -0.0047999, 0.0046899
6: -0.0152988, 0.0004925, -0.0150433, -0.0007618, -0.0124254, 0.0142914
7: -0.0058358, 0.0162785, -0.0033146, 0.0165960, -0.0207834, 0.0164552
8: 0.9869404, 1.0011092, 0.9881437, 1.0011992, -0.0129003, 0.0106861
9: -0.0165053, -0.0031945, -0.0167083, -0.0045433, -0.0098558, 0.0123196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098653, upper bound: 0.0097574
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098653, upper bound: 0.0097574
time: 1.49 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042898, 0.0145840, 0.0053906, 0.0113135, -0.0061098, 0.0091230
1: -0.0007644, 0.0028296, 0.0016343, 0.0028713, -0.0036356, 0.0011953
2: 0.0053497, 0.0119881, 0.0079997, 0.0113795, -0.0060298, 0.0035490
3: -0.0057946, 0.0001070, -0.0059595, -0.0024997, -0.0031589, 0.0057897
4: -0.0017818, 0.0022361, -0.0009247, 0.0024145, -0.0035155, 0.0029181
5: 0.0003958, 0.0055961, 0.0015771, 0.0049512, -0.0044819, 0.0034330
6: -0.0157129, -0.0000968, -0.0153187, -0.0026553, -0.0124393, 0.0126006
7: -0.0049190, 0.0170218, 0.0006042, 0.0179344, -0.0196569, 0.0152403
8: 0.9875057, 1.0016031, 0.9899603, 1.0019355, -0.0118642, 0.0108497
9: -0.0169806, -0.0037395, -0.0175641, -0.0066267, -0.0095654, 0.0116181

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095046, upper bound: 0.0087503
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095046, upper bound: 0.0087503
time: 1.34 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0038607, 0.0159603, 0.0045483, 0.0133689, -0.0095082, 0.0114120
1: -0.0016224, 0.0029290, -0.0001764, 0.0028105, -0.0044329, 0.0031055
2: 0.0042953, 0.0122254, 0.0062764, 0.0118452, -0.0075500, 0.0059490
3: -0.0058842, 0.0010746, -0.0057192, -0.0005287, -0.0052298, 0.0067938
4: -0.0021306, 0.0022507, -0.0015760, 0.0021544, -0.0039624, 0.0033451
5: -0.0001466, 0.0058475, 0.0008914, 0.0054447, -0.0054743, 0.0049560
6: -0.0161441, 0.0009007, -0.0151481, -0.0006975, -0.0139711, 0.0150419
7: -0.0071178, 0.0170965, -0.0035739, 0.0166041, -0.0223933, 0.0184258
8: 0.9865488, 1.0017666, 0.9880820, 1.0012356, -0.0137076, 0.0119789
9: -0.0170283, -0.0025464, -0.0167134, -0.0044344, -0.0110353, 0.0131443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100910, upper bound: 0.0097676
time: 1.78 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100910, upper bound: 0.0097676
time: 1.31 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0050245, 0.0120231, 0.0047791, 0.0133374, -0.0083130, 0.0068332
1: 0.0010418, 0.0028375, 0.0002271, 0.0028484, -0.0018066, 0.0026104
2: 0.0074087, 0.0115820, 0.0063596, 0.0117176, -0.0041190, 0.0052223
3: -0.0058259, -0.0018378, -0.0058690, -0.0010522, -0.0047570, 0.0039081
4: -0.0011916, 0.0022699, -0.0013928, 0.0023166, -0.0032092, 0.0033361
5: 0.0013373, 0.0051657, 0.0008460, 0.0053095, -0.0037001, 0.0043198
6: -0.0151378, -0.0018042, -0.0157116, -0.0012339, -0.0126150, 0.0131001
7: -0.0010228, 0.0171949, -0.0023986, 0.0174336, -0.0170230, 0.0181483
8: 0.9891437, 1.0015181, 0.9885966, 1.0018138, -0.0116642, 0.0116459
9: -0.0170912, -0.0057339, -0.0172439, -0.0050519, -0.0109877, 0.0105383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0086480, upper bound: 0.0087499
time: 1.68 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0087997, upper bound: 0.0089800
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0041624, 0.0140926, 0.0043378, 0.0147859, -0.0106235, 0.0097548
1: -0.0005718, 0.0027939, -0.0006899, 0.0028517, -0.0034236, 0.0034838
2: 0.0057170, 0.0120586, 0.0052562, 0.0119616, -0.0062447, 0.0068024
3: -0.0056535, -0.0000060, -0.0058823, -0.0000526, -0.0056009, 0.0056113
4: -0.0018465, 0.0020833, -0.0017302, 0.0023309, -0.0035849, 0.0037178
5: 0.0006048, 0.0056707, 0.0002722, 0.0055680, -0.0049599, 0.0053985
6: -0.0151114, 0.0001995, -0.0161963, -0.0002082, -0.0146698, 0.0148413
7: -0.0051537, 0.0162403, -0.0045438, 0.0175069, -0.0197735, 0.0206338
8: 0.9872214, 1.0010359, 0.9876126, 1.0019970, -0.0128217, 0.0130163
9: -0.0164809, -0.0035350, -0.0172907, -0.0039173, -0.0122749, 0.0118286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096804, upper bound: 0.0094385
time: 1.37 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097647, upper bound: 0.0096196
time: 1.53 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042967, 0.0145940, 0.0053581, 0.0120504, -0.0076355, 0.0091836
1: -0.0007714, 0.0028293, 0.0013032, 0.0029012, -0.0036727, 0.0015262
2: 0.0053416, 0.0119843, 0.0074741, 0.0113975, -0.0060559, 0.0045102
3: -0.0057936, 0.0000980, -0.0060781, -0.0022229, -0.0035354, 0.0061668
4: -0.0017741, 0.0022350, -0.0009672, 0.0025430, -0.0039463, 0.0029800
5: 0.0003927, 0.0055920, 0.0012394, 0.0049703, -0.0045137, 0.0041914
6: -0.0157076, -0.0001127, -0.0160202, -0.0025798, -0.0125509, 0.0146410
7: -0.0048817, 0.0170162, 0.0002388, 0.0185912, -0.0218118, 0.0157673
8: 0.9875211, 1.0015982, 0.9898878, 1.0024654, -0.0135717, 0.0109556
9: -0.0169770, -0.0037657, -0.0179841, -0.0064764, -0.0097835, 0.0130237

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091494, upper bound: 0.0083966
time: 1.24 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093737, upper bound: 0.0085695
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0038613, 0.0159691, 0.0044596, 0.0145373, -0.0106760, 0.0115094
1: -0.0016294, 0.0029322, -0.0005317, 0.0028507, -0.0044800, 0.0034639
2: 0.0042868, 0.0122251, 0.0054595, 0.0118942, -0.0076074, 0.0067655
3: -0.0058859, 0.0010637, -0.0058779, -0.0002452, -0.0056407, 0.0069416
4: -0.0021269, 0.0022496, -0.0016438, 0.0023263, -0.0043268, 0.0034668
5: -0.0001491, 0.0058471, 0.0003764, 0.0054966, -0.0055679, 0.0054707
6: -0.0161376, 0.0008993, -0.0161065, -0.0004914, -0.0144106, 0.0169743
7: -0.0070947, 0.0170907, -0.0040294, 0.0174830, -0.0241832, 0.0192362
8: 0.9865501, 1.0017612, 0.9878843, 1.0019590, -0.0152145, 0.0123341
9: -0.0170246, -0.0025586, -0.0172754, -0.0042061, -0.0114468, 0.0143311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098318, upper bound: 0.0095238
time: 1.49 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099764, upper bound: 0.0096313
time: 1.46 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0044895, 0.0130242, 0.0037956, 0.0171519, -0.0126624, 0.0092286
1: 0.0001089, 0.0028280, -0.0023784, 0.0031915, -0.0030825, 0.0052064
2: 0.0065741, 0.0118778, 0.0033199, 0.0122614, -0.0056873, 0.0085579
3: -0.0057882, -0.0007294, -0.0059269, 0.0017366, -0.0075248, 0.0051975
4: -0.0015959, 0.0022291, -0.0022846, 0.0020269, -0.0032721, 0.0045138
5: 0.0009750, 0.0054791, -0.0005383, 0.0058856, -0.0049106, 0.0060175
6: -0.0153103, -0.0005607, -0.0157366, 0.0010520, -0.0163623, 0.0142664
7: -0.0035627, 0.0169863, -0.0081590, 0.0159521, -0.0178907, 0.0251453
8: 0.9879508, 1.0014580, 0.9864036, 1.0010835, -0.0119285, 0.0150545
9: -0.0169579, -0.0043770, -0.0162965, -0.0019254, -0.0150324, 0.0107840

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0087503, upper bound: 0.0093635
time: 1.32 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097574, upper bound: 0.0099481
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0044139, 0.0140498, 0.0037832, 0.0171706, -0.0127568, 0.0102665
1: -0.0002102, 0.0028622, -0.0023880, 0.0031956, -0.0034057, 0.0052502
2: 0.0058549, 0.0119195, 0.0033056, 0.0122682, -0.0064133, 0.0086139
3: -0.0059237, -0.0004753, -0.0059293, 0.0017397, -0.0076635, 0.0054540
4: -0.0016541, 0.0023758, -0.0022898, 0.0020257, -0.0033288, 0.0046656
5: 0.0005260, 0.0055234, -0.0005453, 0.0058928, -0.0053668, 0.0060687
6: -0.0162229, -0.0003850, -0.0157378, 0.0010807, -0.0173036, 0.0143739
7: -0.0039581, 0.0177365, -0.0081709, 0.0159460, -0.0183825, 0.0259074
8: 0.9877822, 1.0021088, 0.9863762, 1.0010803, -0.0120266, 0.0157326
9: -0.0174375, -0.0041802, -0.0162927, -0.0019103, -0.0155273, 0.0109838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0087559, upper bound: 0.0093635
time: 1.46 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098097, upper bound: 0.0099481
time: 1.55 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0038235, 0.0164146, 0.0046435, 0.0140800, -0.0102564, 0.0117710
1: -0.0018479, 0.0030158, -0.0003296, 0.0028301, -0.0046779, 0.0033454
2: 0.0039392, 0.0122459, 0.0057776, 0.0117926, -0.0078533, 0.0064683
3: -0.0058405, 0.0012226, -0.0057966, -0.0005132, -0.0053273, 0.0068159
4: -0.0021760, 0.0020983, -0.0015163, 0.0022382, -0.0036768, 0.0030796
5: -0.0002848, 0.0058692, 0.0005527, 0.0053889, -0.0056737, 0.0053166
6: -0.0158418, 0.0009871, -0.0157233, -0.0009188, -0.0133235, 0.0144585
7: -0.0073395, 0.0163174, -0.0032757, 0.0170325, -0.0209049, 0.0170969
8: 0.9864660, 1.0012903, 0.9882944, 1.0016296, -0.0130584, 0.0111246
9: -0.0165302, -0.0023701, -0.0169874, -0.0046295, -0.0101684, 0.0122494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099406, upper bound: 0.0093760
time: 1.46 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100065, upper bound: 0.0093760
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033737, 0.0179459, 0.0037284, 0.0166059, -0.0132322, 0.0142175
1: -0.0027620, 0.0033902, -0.0020477, 0.0030618, -0.0058238, 0.0054379
2: 0.0027584, 0.0124947, 0.0037578, 0.0122985, -0.0095401, 0.0087368
3: -0.0061477, 0.0022548, -0.0058247, 0.0014540, -0.0076018, 0.0080795
4: -0.0026139, 0.0021125, -0.0022691, 0.0020276, -0.0042723, 0.0036519
5: -0.0008895, 0.0061328, -0.0003494, 0.0059249, -0.0068145, 0.0064822
6: -0.0163391, 0.0020327, -0.0156481, 0.0012081, -0.0160989, 0.0176808
7: -0.0099114, 0.0163900, -0.0079409, 0.0159559, -0.0242036, 0.0207654
8: 0.9851890, 1.0014647, 0.9862539, 1.0010656, -0.0158767, 0.0135867
9: -0.0165766, -0.0007925, -0.0162990, -0.0020370, -0.0121918, 0.0143685

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099726, upper bound: 0.0099925
time: 1.20 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099726, upper bound: 0.0104586
time: 1.45 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0044582, 0.0133388, 0.0035688, 0.0187535, -0.0142953, 0.0097700
1: -0.0000955, 0.0028284, -0.0032103, 0.0035525, -0.0036480, 0.0060388
2: 0.0063287, 0.0118950, 0.0021279, 0.0123868, -0.0060581, 0.0097671
3: -0.0057899, -0.0005402, -0.0063196, 0.0025984, -0.0083882, 0.0057794
4: -0.0016292, 0.0022309, -0.0025755, 0.0021711, -0.0036053, 0.0048064
5: 0.0008582, 0.0054974, -0.0011914, 0.0060185, -0.0051602, 0.0066888
6: -0.0154157, -0.0004881, -0.0165430, 0.0015791, -0.0169948, 0.0156927
7: -0.0038296, 0.0169956, -0.0098733, 0.0166894, -0.0197444, 0.0268688
8: 0.9878811, 1.0014950, 0.9851367, 1.0016875, -0.0131443, 0.0163583
9: -0.0169638, -0.0042614, -0.0167680, -0.0008396, -0.0161242, 0.0118845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0091298, upper bound: 0.0091326
time: 1.31 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097676, upper bound: 0.0101600
time: 1.42 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0043972, 0.0143586, 0.0035741, 0.0187656, -0.0143684, 0.0107845
1: -0.0003976, 0.0028627, -0.0032173, 0.0035539, -0.0039514, 0.0060800
2: 0.0056148, 0.0119288, 0.0021180, 0.0123838, -0.0067690, 0.0098108
3: -0.0059256, -0.0003048, -0.0063197, 0.0026010, -0.0085266, 0.0060149
4: -0.0016781, 0.0023779, -0.0025754, 0.0021698, -0.0036618, 0.0049533
5: 0.0004106, 0.0055332, -0.0011955, 0.0060153, -0.0056048, 0.0067287
6: -0.0163300, -0.0003463, -0.0165378, 0.0015667, -0.0178966, 0.0158049
7: -0.0041619, 0.0177470, -0.0098825, 0.0166829, -0.0202204, 0.0276295
8: 0.9877451, 1.0021480, 0.9851380, 1.0016817, -0.0132506, 0.0170100
9: -0.0174443, -0.0040960, -0.0167639, -0.0008352, -0.0166091, 0.0120834

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_A1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091738, upper bound: 0.0091326
time: 1.27 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098231, upper bound: 0.0101600
time: 1.43 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037644, 0.0168760, 0.0044450, 0.0153511, -0.0115867, 0.0124310
1: -0.0021114, 0.0031333, -0.0010313, 0.0028598, -0.0049713, 0.0041646
2: 0.0035785, 0.0122786, 0.0048224, 0.0119023, -0.0083238, 0.0074562
3: -0.0059351, 0.0014740, -0.0059143, 0.0002254, -0.0061605, 0.0073883
4: -0.0022544, 0.0021004, -0.0016972, 0.0023657, -0.0040916, 0.0032600
5: -0.0004675, 0.0059039, 0.0000424, 0.0055052, -0.0059727, 0.0058614
6: -0.0159756, 0.0011244, -0.0164629, -0.0004574, -0.0139036, 0.0161908
7: -0.0078203, 0.0163279, -0.0045889, 0.0176844, -0.0231511, 0.0184703
8: 0.9863342, 1.0013341, 0.9878517, 1.0021622, -0.0148602, 0.0115742
9: -0.0165369, -0.0020759, -0.0174042, -0.0040093, -0.0107916, 0.0136611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099598, upper bound: 0.0094562
time: 1.44 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0100274, upper bound: 0.0094562
time: 1.82 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0033094, 0.0184461, 0.0035079, 0.0181734, -0.0148640, 0.0149382
1: -0.0030382, 0.0034916, -0.0028728, 0.0034460, -0.0064842, 0.0063644
2: 0.0023723, 0.0125302, 0.0025847, 0.0124204, -0.0100481, 0.0099455
3: -0.0062296, 0.0025193, -0.0062359, 0.0022981, -0.0085277, 0.0087551
4: -0.0027004, 0.0021146, -0.0025519, 0.0021720, -0.0046417, 0.0039281
5: -0.0010798, 0.0061704, -0.0009869, 0.0060541, -0.0071339, 0.0071573
6: -0.0164824, 0.0021820, -0.0164576, 0.0017206, -0.0173094, 0.0186396
7: -0.0104198, 0.0164008, -0.0096254, 0.0166941, -0.0262027, 0.0224799
8: 0.9847677, 1.0015093, 0.9853576, 1.0016712, -0.0169035, 0.0153516
9: -0.0165834, -0.0004660, -0.0167710, -0.0009750, -0.0132681, 0.0156406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099894, upper bound: 0.0100934
time: 1.46 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099894, upper bound: 0.0105972
time: 1.31 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 4.62 seconds
IS_A1_B1_B1_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0090441, upper bound: 0.0084062
IS_A1_B1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0092528, upper bound: 0.0085450
IS_A1_B1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096537, upper bound: 0.0093570
IS_A1_B1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0097794, upper bound: 0.0094243
IS_A1_B1_B1_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0090532, upper bound: 0.0084262
IS_A1_B1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0092600, upper bound: 0.0085773
IS_A1_B1_B1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096802, upper bound: 0.0094803
IS_A1_B1_B1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098167, upper bound: 0.0095534
IS_A1_B1_B2_B1_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0090441, upper bound: 0.0083839
IS_A1_B1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0092527, upper bound: 0.0085452
IS_A1_B1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096537, upper bound: 0.0093919
IS_A1_B1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0097795, upper bound: 0.0094799
IS_A1_B1_B2_B2_B1_B1, status: Status.VERIFIED, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0090532, upper bound: 0.0083966
IS_A1_B1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0092600, upper bound: 0.0085695
IS_A1_B1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096802, upper bound: 0.0095006
IS_A1_B1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098167, upper bound: 0.0095926
IS_A1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0092993, upper bound: 0.0093112
IS_A1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0094492, upper bound: 0.0094072
IS_A1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0097967, upper bound: 0.0102050
IS_A1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098957, upper bound: 0.0102218
IS_A1_B2_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0087490, upper bound: 0.0095515
IS_A1_B2_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096249, upper bound: 0.0100768
IS_A1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0090592, upper bound: 0.0094193
IS_A1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096249, upper bound: 0.0101791
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0094617, upper bound: 0.0093173
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096282, upper bound: 0.0094197
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0095318, upper bound: 0.0093173
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096833, upper bound: 0.0094197
IS_A1_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096425, upper bound: 0.0101123
IS_A1_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0097127, upper bound: 0.0101123
IS_A1_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0091192, upper bound: 0.0100626
IS_A1_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0091192, upper bound: 0.0103429
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0092977, upper bound: 0.0087503
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0092977, upper bound: 0.0087503
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098653, upper bound: 0.0097574
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098653, upper bound: 0.0097574
IS_A2_B1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0095046, upper bound: 0.0087503
IS_A2_B1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0095046, upper bound: 0.0087503
IS_A2_B1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0100910, upper bound: 0.0097676
IS_A2_B1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0100910, upper bound: 0.0097676
IS_A2_B1_B2_A1_A1_A1, status: Status.VERIFIED, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0086480, upper bound: 0.0087499
IS_A2_B1_B2_A1_A1_A2, status: Status.VERIFIED, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0087997, upper bound: 0.0089800
IS_A2_B1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0096804, upper bound: 0.0094385
IS_A2_B1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0097647, upper bound: 0.0096196
IS_A2_B1_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0091494, upper bound: 0.0083966
IS_A2_B1_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0093737, upper bound: 0.0085695
IS_A2_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098318, upper bound: 0.0095238
IS_A2_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0099764, upper bound: 0.0096313
IS_A2_B2_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0087503, upper bound: 0.0093635
IS_A2_B2_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0097574, upper bound: 0.0099481
IS_A2_B2_B1_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0087559, upper bound: 0.0093635
IS_A2_B2_B1_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098097, upper bound: 0.0099481
IS_A2_B2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0099406, upper bound: 0.0093760
IS_A2_B2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0100065, upper bound: 0.0093760
IS_A2_B2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0099726, upper bound: 0.0099925
IS_A2_B2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0099726, upper bound: 0.0104586
IS_A2_B2_B2_A1_A1_B1, status: Status.VERIFIED, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0091298, upper bound: 0.0091326
IS_A2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0097676, upper bound: 0.0101600
IS_A2_B2_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0091738, upper bound: 0.0091326
IS_A2_B2_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0098231, upper bound: 0.0101600
IS_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0099598, upper bound: 0.0094562
IS_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0100274, upper bound: 0.0094562
IS_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0099894, upper bound: 0.0100934
IS_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 4.62
Output dim: 8, lower bound: -0.0099894, upper bound: 0.0105972

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0043737, 0.0129901, 0.0056549, 0.0105722, -0.0048977, 0.0070461
1: 0.0000258, 0.0027711, 0.0021393, 0.0028497, -0.0028209, 0.0005048
2: 0.0065279, 0.0119417, 0.0085148, 0.0112334, -0.0046335, 0.0027078
3: -0.0055631, -0.0006498, -0.0058741, -0.0030623, -0.0019978, 0.0047533
4: -0.0016661, 0.0019855, -0.0007218, 0.0023221, -0.0032035, 0.0021628
5: 0.0010804, 0.0055469, 0.0019159, 0.0047964, -0.0034822, 0.0028691
6: -0.0144803, -0.0002917, -0.0146987, -0.0032695, -0.0094220, 0.0113837
7: -0.0039210, 0.0157403, 0.0018961, 0.0174617, -0.0175047, 0.0110598
8: 0.9876928, 1.0006180, 0.9905495, 1.0015142, -0.0109210, 0.0081838
9: -0.0161611, -0.0041480, -0.0172618, -0.0073087, -0.0070719, 0.0105584

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085450
time: 1.37 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085450
time: 1.47 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0039682, 0.0141937, 0.0047321, 0.0114691, -0.0072545, 0.0092882
1: -0.0008196, 0.0027511, 0.0008648, 0.0027455, -0.0035651, 0.0018148
2: 0.0055596, 0.0121659, 0.0076660, 0.0117436, -0.0061840, 0.0044404
3: -0.0054838, 0.0002683, -0.0054618, -0.0015188, -0.0033188, 0.0057301
4: -0.0019789, 0.0018996, -0.0013860, 0.0018758, -0.0034727, 0.0024556
5: 0.0006175, 0.0057845, 0.0016952, 0.0053370, -0.0044581, 0.0038738
6: -0.0145781, 0.0006508, -0.0136535, -0.0011247, -0.0107773, 0.0128756
7: -0.0059391, 0.0153013, -0.0020986, 0.0151793, -0.0193944, 0.0132144
8: 0.9867885, 1.0004362, 0.9884919, 1.0000950, -0.0118674, 0.0090164
9: -0.0158804, -0.0030949, -0.0158024, -0.0050924, -0.0080773, 0.0114756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093570
time: 1.52 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093570
time: 1.53 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0039532, 0.0143001, 0.0048520, 0.0114465, -0.0071304, 0.0093682
1: -0.0008388, 0.0027743, 0.0010427, 0.0027898, -0.0036287, 0.0016611
2: 0.0055085, 0.0121742, 0.0077519, 0.0116773, -0.0061688, 0.0043000
3: -0.0055758, 0.0002979, -0.0056372, -0.0017140, -0.0032329, 0.0058615
4: -0.0019907, 0.0019992, -0.0013014, 0.0020656, -0.0035904, 0.0025216
5: 0.0005493, 0.0057933, 0.0016417, 0.0052667, -0.0045116, 0.0038598
6: -0.0149308, 0.0006856, -0.0142530, -0.0014035, -0.0110572, 0.0132197
7: -0.0060139, 0.0158103, -0.0015961, 0.0161502, -0.0200017, 0.0134779
8: 0.9867551, 1.0007845, 0.9887593, 1.0007461, -0.0122585, 0.0093016
9: -0.0162059, -0.0030552, -0.0164232, -0.0053742, -0.0082886, 0.0118602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094243
time: 1.55 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094243
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0043452, 0.0134271, 0.0055704, 0.0107115, -0.0054280, 0.0076669
1: -0.0002546, 0.0027715, 0.0021193, 0.0028682, -0.0031228, 0.0005339
2: 0.0061901, 0.0119575, 0.0084360, 0.0112801, -0.0050853, 0.0030031
3: -0.0055646, -0.0004148, -0.0059475, -0.0030062, -0.0020857, 0.0053365
4: -0.0017030, 0.0019870, -0.0007749, 0.0024015, -0.0035527, 0.0022488
5: 0.0009071, 0.0055636, 0.0018360, 0.0048459, -0.0037402, 0.0031777
6: -0.0146159, -0.0002255, -0.0150024, -0.0030731, -0.0099134, 0.0125922
7: -0.0042351, 0.0157482, 0.0016195, 0.0178679, -0.0194856, 0.0115057
8: 0.9876292, 1.0006633, 0.9903612, 1.0018021, -0.0120764, 0.0085367
9: -0.0161662, -0.0040179, -0.0175215, -0.0071347, -0.0073537, 0.0117144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085773
time: 1.19 seconds

## Relational analysis of IS_A1_B1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085773
time: 1.43 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0039328, 0.0146708, 0.0046237, 0.0125307, -0.0085979, 0.0099624
1: -0.0011081, 0.0027514, 0.0002395, 0.0027678, -0.0038759, 0.0024721
2: 0.0051871, 0.0121855, 0.0068802, 0.0118035, -0.0066164, 0.0053053
3: -0.0054852, 0.0005212, -0.0055500, -0.0009204, -0.0040277, 0.0060711
4: -0.0020187, 0.0019011, -0.0014987, 0.0019712, -0.0037940, 0.0025840
5: 0.0004353, 0.0058052, 0.0012631, 0.0054005, -0.0047432, 0.0044771
6: -0.0147232, 0.0007330, -0.0142938, -0.0008729, -0.0112362, 0.0142531
7: -0.0062691, 0.0153090, -0.0029923, 0.0156673, -0.0212211, 0.0142621
8: 0.9867097, 1.0004832, 0.9882503, 1.0005230, -0.0130015, 0.0093260
9: -0.0158853, -0.0029568, -0.0161144, -0.0047013, -0.0085249, 0.0125402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0094803
time: 1.59 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0094803
time: 1.59 seconds

## BFS IS instance: IS_A1_B1_B1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0039174, 0.0147702, 0.0047347, 0.0124532, -0.0084758, 0.0100355
1: -0.0011269, 0.0027747, 0.0004233, 0.0028143, -0.0039411, 0.0023183
2: 0.0051411, 0.0121940, 0.0069914, 0.0117422, -0.0066011, 0.0052027
3: -0.0055773, 0.0005507, -0.0057339, -0.0011208, -0.0039407, 0.0062846
4: -0.0020308, 0.0020008, -0.0014181, 0.0021703, -0.0039160, 0.0026558
5: 0.0003692, 0.0058143, 0.0012270, 0.0053355, -0.0048016, 0.0044655
6: -0.0150730, 0.0007689, -0.0149065, -0.0011307, -0.0115377, 0.0146142
7: -0.0063452, 0.0158186, -0.0024986, 0.0166855, -0.0218632, 0.0145378
8: 0.9866752, 1.0008303, 0.9884976, 1.0011979, -0.0134079, 0.0096335
9: -0.0162112, -0.0029159, -0.0167655, -0.0049709, -0.0087533, 0.0129412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B1_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095533
time: 1.55 seconds

## Relational analysis of IS_A1_B1_B1_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095534
time: 1.62 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0043641, 0.0130072, 0.0056015, 0.0107835, -0.0056772, 0.0071240
1: 0.0000160, 0.0027709, 0.0021315, 0.0028802, -0.0028642, 0.0005131
2: 0.0065147, 0.0119470, 0.0083979, 0.0112630, -0.0046807, 0.0031388
3: -0.0055622, -0.0006512, -0.0059949, -0.0030318, -0.0020307, 0.0051663
4: -0.0016706, 0.0019844, -0.0007549, 0.0024529, -0.0036825, 0.0021984
5: 0.0010738, 0.0055525, 0.0017921, 0.0048277, -0.0035241, 0.0033257
6: -0.0144821, -0.0002694, -0.0151899, -0.0031453, -0.0095629, 0.0131955
7: -0.0039282, 0.0157350, 0.0017269, 0.0181306, -0.0199343, 0.0112419
8: 0.9876714, 1.0006155, 0.9904303, 1.0019854, -0.0126592, 0.0083142
9: -0.0161577, -0.0041340, -0.0176895, -0.0072006, -0.0071884, 0.0121225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085452
time: 1.39 seconds

## Relational analysis of IS_A1_B1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085067
time: 1.46 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0039606, 0.0142111, 0.0046070, 0.0125073, -0.0085467, 0.0094106
1: -0.0008298, 0.0027509, 0.0005268, 0.0027864, -0.0036162, 0.0021999
2: 0.0055455, 0.0121701, 0.0069438, 0.0118128, -0.0062673, 0.0052264
3: -0.0054830, 0.0002677, -0.0056238, -0.0012204, -0.0037167, 0.0058916
4: -0.0019813, 0.0018987, -0.0014817, 0.0020512, -0.0038672, 0.0025456
5: 0.0006109, 0.0057889, 0.0012357, 0.0054103, -0.0045261, 0.0045532
6: -0.0145798, 0.0006685, -0.0145918, -0.0008340, -0.0110180, 0.0148102
7: -0.0059420, 0.0152966, -0.0027039, 0.0160763, -0.0213606, 0.0138625
8: 0.9867716, 1.0004340, 0.9882129, 1.0008167, -0.0134388, 0.0092407
9: -0.0158774, -0.0030893, -0.0163759, -0.0047709, -0.0083848, 0.0127615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093919
time: 1.56 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093392
time: 1.45 seconds

## BFS IS instance: IS_A1_B1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0039453, 0.0143173, 0.0047394, 0.0123817, -0.0084364, 0.0094832
1: -0.0008488, 0.0027741, 0.0007179, 0.0028284, -0.0036772, 0.0020372
2: 0.0054946, 0.0121786, 0.0070849, 0.0117396, -0.0062449, 0.0050937
3: -0.0055749, 0.0002972, -0.0057898, -0.0014222, -0.0036213, 0.0060869
4: -0.0019933, 0.0019981, -0.0013883, 0.0022308, -0.0039907, 0.0026062
5: 0.0005426, 0.0057979, 0.0012211, 0.0053327, -0.0045752, 0.0045768
6: -0.0149329, 0.0007039, -0.0151437, -0.0011417, -0.0112819, 0.0151591
7: -0.0060176, 0.0158051, -0.0021642, 0.0169948, -0.0219986, 0.0140908
8: 0.9867375, 1.0007821, 0.9885082, 1.0014232, -0.0138492, 0.0095103
9: -0.0162026, -0.0030487, -0.0169633, -0.0050812, -0.0085782, 0.0131661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094800
time: 1.25 seconds

## Relational analysis of IS_A1_B1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094128
time: 1.52 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0043423, 0.0134435, 0.0055400, 0.0112654, -0.0066514, 0.0077408
1: -0.0002640, 0.0027713, 0.0018589, 0.0028934, -0.0031573, 0.0008588
2: 0.0061776, 0.0119591, 0.0080629, 0.0112969, -0.0051193, 0.0037731
3: -0.0055637, -0.0004140, -0.0060471, -0.0027581, -0.0024084, 0.0056331
4: -0.0017019, 0.0019861, -0.0008166, 0.0025094, -0.0039617, 0.0023105
5: 0.0009005, 0.0055653, 0.0015732, 0.0048637, -0.0037796, 0.0038062
6: -0.0146177, -0.0002188, -0.0155818, -0.0030025, -0.0100457, 0.0144065
7: -0.0042118, 0.0157433, 0.0012612, 0.0184193, -0.0215456, 0.0120037
8: 0.9876228, 1.0006609, 0.9902933, 1.0022480, -0.0136518, 0.0086586
9: -0.0161631, -0.0040234, -0.0178742, -0.0069877, -0.0075691, 0.0130500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085695
time: 1.48 seconds

## Relational analysis of IS_A1_B1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085447
time: 1.56 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0039281, 0.0146881, 0.0045126, 0.0134849, -0.0095568, 0.0100859
1: -0.0011178, 0.0027512, -0.0000195, 0.0028058, -0.0039236, 0.0027637
2: 0.0051734, 0.0121881, 0.0062165, 0.0118650, -0.0066916, 0.0059716
3: -0.0054844, 0.0005232, -0.0057004, -0.0006932, -0.0043459, 0.0062236
4: -0.0020199, 0.0019002, -0.0015811, 0.0021341, -0.0041331, 0.0026675
5: 0.0004289, 0.0058080, 0.0008303, 0.0054656, -0.0048118, 0.0049777
6: -0.0147248, 0.0007441, -0.0151725, -0.0006145, -0.0114804, 0.0159166
7: -0.0062703, 0.0153044, -0.0034893, 0.0165002, -0.0227706, 0.0148363
8: 0.9866990, 1.0004809, 0.9880024, 1.0011986, -0.0143952, 0.0095536
9: -0.0158824, -0.0029529, -0.0166471, -0.0044255, -0.0088084, 0.0136464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0095006
time: 1.45 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0094619
time: 1.95 seconds

## BFS IS instance: IS_A1_B1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0039125, 0.0147872, 0.0046553, 0.0133838, -0.0094713, 0.0101320
1: -0.0011364, 0.0027745, 0.0001456, 0.0028481, -0.0039845, 0.0026289
2: 0.0051276, 0.0121967, 0.0063361, 0.0117861, -0.0066585, 0.0058606
3: -0.0055764, 0.0005522, -0.0058678, -0.0008925, -0.0042841, 0.0064200
4: -0.0020321, 0.0019998, -0.0014831, 0.0023153, -0.0042675, 0.0027350
5: 0.0003627, 0.0058171, 0.0008158, 0.0053820, -0.0048634, 0.0050013
6: -0.0150750, 0.0007802, -0.0157299, -0.0009461, -0.0117557, 0.0163912
7: -0.0063474, 0.0158135, -0.0029297, 0.0174271, -0.0235971, 0.0151167
8: 0.9866644, 1.0008280, 0.9883205, 1.0018170, -0.0148406, 0.0098362
9: -0.0162079, -0.0029117, -0.0172397, -0.0047505, -0.0090246, 0.0140858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095926
time: 1.45 seconds

## Relational analysis of IS_A1_B1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_B1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095386
time: 1.47 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0046134, 0.0135607, 0.0044983, 0.0145240, -0.0099106, 0.0090624
1: -0.0004389, 0.0027452, -0.0007943, 0.0027662, -0.0032050, 0.0035395
2: 0.0060292, 0.0118092, 0.0053377, 0.0118729, -0.0058436, 0.0064716
3: -0.0054605, -0.0003384, -0.0055436, -0.0000507, -0.0052033, 0.0050864
4: -0.0015519, 0.0018744, -0.0016382, 0.0019643, -0.0030880, 0.0028386
5: 0.0009039, 0.0054065, 0.0004810, 0.0054739, -0.0044698, 0.0048639
6: -0.0142146, -0.0008487, -0.0149367, -0.0005814, -0.0114778, 0.0125104
7: -0.0035736, 0.0151722, -0.0041147, 0.0156319, -0.0172575, 0.0161443
8: 0.9882271, 1.0002656, 0.9879706, 1.0007230, -0.0108445, 0.0098575
9: -0.0157979, -0.0045055, -0.0160918, -0.0042154, -0.0093977, 0.0102046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092993, upper bound: 0.0092805
time: 1.39 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092993, upper bound: 0.0093112
time: 1.42 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0045994, 0.0136651, 0.0046327, 0.0144870, -0.0098876, 0.0090324
1: -0.0004587, 0.0027682, -0.0006728, 0.0028115, -0.0032702, 0.0034410
2: 0.0059784, 0.0118170, 0.0054162, 0.0117986, -0.0058202, 0.0064007
3: -0.0055517, -0.0003100, -0.0057229, -0.0001865, -0.0051668, 0.0052025
4: -0.0015632, 0.0019731, -0.0015525, 0.0021584, -0.0031935, 0.0029095
5: 0.0008357, 0.0054147, 0.0004434, 0.0053952, -0.0045201, 0.0048674
6: -0.0145663, -0.0008163, -0.0155493, -0.0008937, -0.0117635, 0.0128390
7: -0.0036470, 0.0156769, -0.0036668, 0.0166247, -0.0177740, 0.0164643
8: 0.9881960, 1.0006119, 0.9882702, 1.0013802, -0.0112276, 0.0101321
9: -0.0161206, -0.0044670, -0.0167266, -0.0044967, -0.0096280, 0.0105477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094492, upper bound: 0.0093717
time: 1.43 seconds

## Relational analysis of IS_A1_B2_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_B2_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094492, upper bound: 0.0094072
time: 1.46 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0041998, 0.0148812, 0.0035829, 0.0174516, -0.0132518, 0.0112983
1: -0.0012759, 0.0027484, -0.0026596, 0.0031806, -0.0044565, 0.0053979
2: 0.0049848, 0.0120379, 0.0030470, 0.0123790, -0.0073941, 0.0089909
3: -0.0054732, 0.0006148, -0.0057159, 0.0020201, -0.0072193, 0.0063307
4: -0.0018715, 0.0018881, -0.0024471, 0.0017528, -0.0033763, 0.0034729
5: 0.0003885, 0.0056488, -0.0005945, 0.0060102, -0.0056217, 0.0062433
6: -0.0146453, 0.0001125, -0.0149427, 0.0015463, -0.0145753, 0.0143149
7: -0.0056601, 0.0152424, -0.0090146, 0.0145506, -0.0192231, 0.0200764
8: 0.9873049, 1.0004220, 0.9858307, 1.0001622, -0.0118953, 0.0131487
9: -0.0158428, -0.0034274, -0.0154004, -0.0013670, -0.0117191, 0.0111844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090251, upper bound: 0.0096919
time: 1.70 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090251, upper bound: 0.0099709
time: 1.53 seconds

## BFS IS instance: IS_A1_B2_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0041843, 0.0149857, 0.0037174, 0.0173008, -0.0131165, 0.0112683
1: -0.0013040, 0.0027715, -0.0025002, 0.0031975, -0.0045015, 0.0052666
2: 0.0049270, 0.0120465, 0.0032004, 0.0123046, -0.0073776, 0.0088461
3: -0.0055648, 0.0006523, -0.0058715, 0.0018590, -0.0071369, 0.0065238
4: -0.0018844, 0.0019873, -0.0023460, 0.0019454, -0.0035747, 0.0035076
5: 0.0003204, 0.0056579, -0.0005733, 0.0059314, -0.0056110, 0.0062312
6: -0.0149963, 0.0001486, -0.0154918, 0.0012338, -0.0146624, 0.0150128
7: -0.0057442, 0.0157496, -0.0084824, 0.0155355, -0.0202180, 0.0202364
8: 0.9872704, 1.0007682, 0.9862292, 1.0008050, -0.0125977, 0.0131532
9: -0.0161671, -0.0033841, -0.0160302, -0.0017134, -0.0118090, 0.0118316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092170, upper bound: 0.0097656
time: 1.27 seconds

## Relational analysis of IS_A1_B2_A1_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0092170, upper bound: 0.0099873
time: 1.43 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0057549, 0.0108976, 0.0040079, 0.0164625, -0.0107076, 0.0068897
1: 0.0021537, 0.0028967, -0.0019502, 0.0030204, -0.0008667, 0.0048468
2: 0.0083349, 0.0111781, 0.0038512, 0.0121440, -0.0038091, 0.0073270
3: -0.0060601, -0.0031195, -0.0057832, 0.0012124, -0.0072725, 0.0023264
4: -0.0006599, 0.0025235, -0.0020740, 0.0020159, -0.0021236, 0.0044629
5: 0.0017253, 0.0047379, -0.0002770, 0.0057612, -0.0040359, 0.0049588
6: -0.0154549, -0.0035019, -0.0154983, 0.0005585, -0.0156240, 0.0103813
7: 0.0022126, 0.0184916, -0.0068660, 0.0158957, -0.0108596, 0.0249513
8: 0.9907725, 1.0022397, 0.9868770, 1.0009913, -0.0083484, 0.0153627
9: -0.0179204, -0.0075111, -0.0162605, -0.0026915, -0.0148365, 0.0069439

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_A2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0084155, upper bound: 0.0091430
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A1_A2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A1_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0085753, upper bound: 0.0093604
time: 1.30 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0049105, 0.0113096, 0.0035535, 0.0181194, -0.0132089, 0.0077561
1: 0.0015144, 0.0028466, -0.0029189, 0.0033916, -0.0018772, 0.0057654
2: 0.0079656, 0.0116450, 0.0025869, 0.0123952, -0.0044296, 0.0090581
3: -0.0058618, -0.0021770, -0.0060879, 0.0022963, -0.0081581, 0.0037939
4: -0.0012265, 0.0023087, -0.0025221, 0.0020300, -0.0024415, 0.0048308
5: 0.0016106, 0.0052325, -0.0009093, 0.0060274, -0.0044168, 0.0061418
6: -0.0149811, -0.0015393, -0.0160008, 0.0016146, -0.0165957, 0.0121410
7: -0.0009704, 0.0173934, -0.0094943, 0.0159683, -0.0127958, 0.0268878
8: 0.9888896, 1.0015714, 0.9854530, 1.0011662, -0.0094910, 0.0161184
9: -0.0172182, -0.0056373, -0.0163069, -0.0010711, -0.0161471, 0.0080064

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=18, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_A2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093631, upper bound: 0.0097749
time: 1.40 seconds

## Relational analysis of IS_A1_B2_A1_A2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094471, upper bound: 0.0098929
time: 1.40 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0046651, 0.0144335, 0.0046150, 0.0146603, -0.0099952, 0.0098185
1: -0.0005549, 0.0028067, -0.0007214, 0.0028295, -0.0033601, 0.0035282
2: 0.0054840, 0.0117807, 0.0053156, 0.0118083, -0.0063243, 0.0064650
3: -0.0057041, -0.0002994, -0.0057942, -0.0001409, -0.0055334, 0.0050342
4: -0.0015227, 0.0021381, -0.0015641, 0.0022356, -0.0028963, 0.0033037
5: 0.0004529, 0.0053763, 0.0003472, 0.0054056, -0.0049527, 0.0049440
6: -0.0154727, -0.0009689, -0.0158598, -0.0008526, -0.0135841, 0.0120467
7: -0.0034584, 0.0165207, -0.0037283, 0.0170194, -0.0163221, 0.0184488
8: 0.9883425, 1.0012932, 0.9882308, 1.0016663, -0.0102046, 0.0116811
9: -0.0166601, -0.0045986, -0.0169790, -0.0044587, -0.0109153, 0.0095804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094337, upper bound: 0.0090380
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095045, upper bound: 0.0092252
time: 2.62 seconds

## BFS IS instance: IS_A1_B2_A1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042336, 0.0158745, 0.0036977, 0.0174841, -0.0132505, 0.0121768
1: -0.0014460, 0.0028101, -0.0025751, 0.0032580, -0.0047041, 0.0053852
2: 0.0043636, 0.0120192, 0.0030738, 0.0123155, -0.0079519, 0.0089454
3: -0.0057174, 0.0007024, -0.0059773, 0.0019340, -0.0075642, 0.0066797
4: -0.0018577, 0.0021525, -0.0023742, 0.0020237, -0.0033319, 0.0039576
5: -0.0001125, 0.0056290, -0.0006758, 0.0059429, -0.0060554, 0.0063049
6: -0.0159702, 0.0000340, -0.0158153, 0.0012794, -0.0167442, 0.0145980
7: -0.0056258, 0.0165944, -0.0086376, 0.0159357, -0.0190472, 0.0224853
8: 0.9873802, 1.0014735, 0.9861128, 1.0010992, -0.0118150, 0.0150226
9: -0.0167073, -0.0034706, -0.0162861, -0.0016092, -0.0132872, 0.0110413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094140, upper bound: 0.0097301
time: 1.31 seconds

## Relational analysis of IS_A1_B2_A1_A2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094140, upper bound: 0.0101791
time: 1.54 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044507, 0.0150601, 0.0044547, 0.0149469, -0.0104961, 0.0106055
1: -0.0012715, 0.0027781, -0.0010521, 0.0027665, -0.0040381, 0.0038301
2: 0.0048959, 0.0118992, 0.0050066, 0.0118970, -0.0070011, 0.0068926
3: -0.0055907, 0.0004840, -0.0055450, 0.0001887, -0.0057631, 0.0060143
4: -0.0017168, 0.0020153, -0.0016836, 0.0019659, -0.0032719, 0.0031920
5: 0.0002962, 0.0055018, 0.0003194, 0.0054995, -0.0052033, 0.0051620
6: -0.0150481, -0.0004708, -0.0150606, -0.0004799, -0.0129990, 0.0130397
7: -0.0048506, 0.0158927, -0.0044696, 0.0156401, -0.0187282, 0.0181379
8: 0.9878645, 1.0008698, 0.9878733, 1.0007656, -0.0112564, 0.0111196
9: -0.0162586, -0.0039344, -0.0160971, -0.0040581, -0.0105685, 0.0108467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094617, upper bound: 0.0092862
time: 1.38 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094617, upper bound: 0.0093173
time: 1.52 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044369, 0.0151553, 0.0045768, 0.0149670, -0.0105302, 0.0105785
1: -0.0012896, 0.0028014, -0.0009618, 0.0028119, -0.0041015, 0.0037632
2: 0.0048504, 0.0119068, 0.0050385, 0.0118295, -0.0069791, 0.0068683
3: -0.0056829, 0.0005110, -0.0057246, 0.0000915, -0.0057622, 0.0061324
4: -0.0017280, 0.0021151, -0.0016077, 0.0021603, -0.0033826, 0.0032622
5: 0.0002307, 0.0055099, 0.0002620, 0.0054280, -0.0051973, 0.0051986
6: -0.0153994, -0.0004385, -0.0156935, -0.0007637, -0.0132844, 0.0134085
7: -0.0049236, 0.0164031, -0.0040832, 0.0166342, -0.0192697, 0.0184732
8: 0.9878336, 1.0012165, 0.9881456, 1.0014279, -0.0116559, 0.0113773
9: -0.0165850, -0.0038964, -0.0167327, -0.0043058, -0.0107968, 0.0112057

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096282, upper bound: 0.0093854
time: 1.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096282, upper bound: 0.0094197
time: 1.28 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043941, 0.0162112, 0.0044476, 0.0149633, -0.0105692, 0.0117636
1: -0.0015485, 0.0028114, -0.0010599, 0.0027664, -0.0043149, 0.0038713
2: 0.0041261, 0.0119305, 0.0049945, 0.0119009, -0.0077748, 0.0069360
3: -0.0057225, 0.0007315, -0.0055444, 0.0001903, -0.0059128, 0.0062759
4: -0.0017683, 0.0021580, -0.0016860, 0.0019652, -0.0033360, 0.0035434
5: -0.0002298, 0.0055350, 0.0003125, 0.0055036, -0.0057334, 0.0052076
6: -0.0159608, -0.0003391, -0.0150652, -0.0004635, -0.0149656, 0.0131973
7: -0.0052368, 0.0166224, -0.0044759, 0.0156367, -0.0192198, 0.0198678
8: 0.9877382, 1.0014747, 0.9878575, 1.0007650, -0.0114024, 0.0125935
9: -0.0167252, -0.0037577, -0.0160949, -0.0040515, -0.0117107, 0.0110677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095318, upper bound: 0.0092862
time: 1.24 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095318, upper bound: 0.0092862
time: 1.23 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043801, 0.0162935, 0.0045818, 0.0149840, -0.0106038, 0.0117118
1: -0.0015659, 0.0028330, -0.0009696, 0.0028117, -0.0043776, 0.0038026
2: 0.0040826, 0.0119382, 0.0050261, 0.0118267, -0.0077442, 0.0069121
3: -0.0058079, 0.0007586, -0.0057239, 0.0000925, -0.0059004, 0.0064381
4: -0.0017795, 0.0022504, -0.0016017, 0.0021595, -0.0034370, 0.0035783
5: -0.0002862, 0.0055432, 0.0002548, 0.0054251, -0.0057112, 0.0052352
6: -0.0162800, -0.0003067, -0.0156988, -0.0007753, -0.0150763, 0.0135309
7: -0.0053100, 0.0170953, -0.0040356, 0.0166300, -0.0197142, 0.0200484
8: 0.9877070, 1.0018063, 0.9881567, 1.0014271, -0.0117677, 0.0127283
9: -0.0170275, -0.0037195, -0.0167300, -0.0043270, -0.0118247, 0.0113959

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096833, upper bound: 0.0093854
time: 1.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096833, upper bound: 0.0093854
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0047681, 0.0117477, 0.0036426, 0.0180047, -0.0132366, 0.0081051
1: 0.0009895, 0.0028323, -0.0028698, 0.0033651, -0.0023755, 0.0057021
2: 0.0075983, 0.0117237, 0.0026720, 0.0123459, -0.0047476, 0.0090517
3: -0.0058052, -0.0016388, -0.0060643, 0.0022204, -0.0080256, 0.0044255
4: -0.0013559, 0.0022475, -0.0024587, 0.0020266, -0.0029837, 0.0044132
5: 0.0014617, 0.0053159, -0.0008746, 0.0059752, -0.0045135, 0.0061905
6: -0.0149365, -0.0012083, -0.0159582, 0.0014075, -0.0161032, 0.0136826
7: -0.0018966, 0.0170803, -0.0091556, 0.0159506, -0.0158929, 0.0250332
8: 0.9885721, 1.0014031, 0.9857036, 1.0011469, -0.0112384, 0.0156994
9: -0.0170179, -0.0051944, -0.0162956, -0.0012866, -0.0148611, 0.0098028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094092, upper bound: 0.0097976
time: 1.36 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094761, upper bound: 0.0099290
time: 1.35 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0046962, 0.0126608, 0.0036368, 0.0180230, -0.0133268, 0.0090240
1: 0.0006489, 0.0028681, -0.0028791, 0.0033683, -0.0027194, 0.0057472
2: 0.0069370, 0.0117635, 0.0026582, 0.0123492, -0.0054122, 0.0091053
3: -0.0059470, -0.0013266, -0.0060660, 0.0022241, -0.0081711, 0.0047395
4: -0.0014234, 0.0024010, -0.0024607, 0.0020255, -0.0030505, 0.0047390
5: 0.0010479, 0.0053580, -0.0008815, 0.0059786, -0.0049308, 0.0062396
6: -0.0157932, -0.0010412, -0.0159597, 0.0014210, -0.0172143, 0.0138244
7: -0.0023981, 0.0178652, -0.0091574, 0.0159449, -0.0164536, 0.0266308
8: 0.9884118, 1.0020430, 0.9856835, 1.0011439, -0.0113700, 0.0163595
9: -0.0175199, -0.0049625, -0.0162919, -0.0012812, -0.0159164, 0.0100360

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094446, upper bound: 0.0097976
time: 1.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_A2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095296, upper bound: 0.0099291
time: 1.38 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0046718, 0.0130786, 0.0034336, 0.0180855, -0.0134137, 0.0096451
1: 0.0002748, 0.0028552, -0.0028952, 0.0033921, -0.0031173, 0.0057504
2: 0.0065996, 0.0117769, 0.0026219, 0.0124615, -0.0058619, 0.0091551
3: -0.0058960, -0.0009883, -0.0060913, 0.0022956, -0.0081916, 0.0051030
4: -0.0014677, 0.0023459, -0.0025855, 0.0020341, -0.0028402, 0.0046395
5: 0.0009095, 0.0053723, -0.0009188, 0.0060977, -0.0051882, 0.0062911
6: -0.0157382, -0.0009845, -0.0160665, 0.0018934, -0.0174080, 0.0134613
7: -0.0028175, 0.0175832, -0.0097795, 0.0159892, -0.0155484, 0.0261467
8: 0.9883574, 1.0018910, 0.9852242, 1.0012020, -0.0106532, 0.0166668
9: -0.0173395, -0.0048034, -0.0163203, -0.0008733, -0.0155966, 0.0093628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095518, upper bound: 0.0097550
time: 1.30 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095546, upper bound: 0.0097550
time: 1.31 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0037733, 0.0156671, 0.0034336, 0.0180855, -0.0143122, 0.0122335
1: -0.0015859, 0.0028417, -0.0028952, 0.0033921, -0.0049780, 0.0057369
2: 0.0044846, 0.0122737, 0.0026219, 0.0124615, -0.0079769, 0.0096518
3: -0.0057463, 0.0010523, -0.0060913, 0.0022956, -0.0080122, 0.0071436
4: -0.0021622, 0.0021576, -0.0025855, 0.0020341, -0.0031662, 0.0041023
5: -0.0000095, 0.0058987, -0.0009188, 0.0060977, -0.0061072, 0.0068175
6: -0.0158209, 0.0011038, -0.0160665, 0.0018934, -0.0166666, 0.0145452
7: -0.0072539, 0.0166206, -0.0097795, 0.0159892, -0.0182113, 0.0233789
8: 0.9863538, 1.0014486, 0.9852242, 1.0012020, -0.0113702, 0.0156118
9: -0.0167240, -0.0024621, -0.0163203, -0.0008733, -0.0138367, 0.0105096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099087, upper bound: 0.0102766
time: 1.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099087, upper bound: 0.0103429
time: 1.39 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0047275, 0.0132293, 0.0054057, 0.0111089, -0.0053209, 0.0075715
1: -0.0000379, 0.0027896, 0.0018011, 0.0028708, -0.0028990, 0.0009455
2: 0.0063605, 0.0117461, 0.0081517, 0.0113712, -0.0049414, 0.0030423
3: -0.0056364, -0.0007240, -0.0059577, -0.0026494, -0.0026675, 0.0048079
4: -0.0014547, 0.0020648, -0.0009029, 0.0024126, -0.0031501, 0.0025705
5: 0.0009620, 0.0053397, 0.0016641, 0.0049424, -0.0037802, 0.0030255
6: -0.0147507, -0.0011141, -0.0152190, -0.0026903, -0.0107968, 0.0113589
7: -0.0028941, 0.0161461, 0.0007988, 0.0179243, -0.0173683, 0.0133410
8: 0.9884816, 1.0008854, 0.9899939, 1.0018992, -0.0107490, 0.0095370
9: -0.0164206, -0.0048354, -0.0175577, -0.0067040, -0.0084197, 0.0103936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0089136, upper bound: 0.0084262
time: 1.45 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0091071, upper bound: 0.0085784
time: 2.27 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0046556, 0.0143930, 0.0054057, 0.0111089, -0.0056438, 0.0089873
1: -0.0004060, 0.0028246, 0.0018011, 0.0028708, -0.0032680, 0.0010012
2: 0.0055480, 0.0117859, 0.0081517, 0.0113712, -0.0058232, 0.0032208
3: -0.0057748, -0.0004257, -0.0059577, -0.0026494, -0.0028877, 0.0051926
4: -0.0015175, 0.0022146, -0.0009029, 0.0024126, -0.0033647, 0.0028089
5: 0.0004456, 0.0053818, 0.0016641, 0.0049424, -0.0044617, 0.0032147
6: -0.0157048, -0.0009469, -0.0152190, -0.0026903, -0.0122007, 0.0121094
7: -0.0033447, 0.0169121, 0.0007988, 0.0179243, -0.0185375, 0.0145600
8: 0.9883213, 1.0015388, 0.9899939, 1.0018992, -0.0114689, 0.0105532
9: -0.0169104, -0.0046207, -0.0175577, -0.0067040, -0.0091992, 0.0111002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0089136, upper bound: 0.0084262
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0091071, upper bound: 0.0085784
time: 1.84 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0043189, 0.0144833, 0.0045760, 0.0130352, -0.0087163, 0.0097871
1: -0.0008540, 0.0027930, 0.0000388, 0.0028102, -0.0036641, 0.0027542
2: 0.0053939, 0.0119721, 0.0065340, 0.0118299, -0.0064360, 0.0054381
3: -0.0056498, 0.0002052, -0.0057177, -0.0007229, -0.0045573, 0.0058554
4: -0.0017722, 0.0020793, -0.0015449, 0.0021528, -0.0035236, 0.0029747
5: 0.0004605, 0.0055791, 0.0010178, 0.0054285, -0.0047414, 0.0044942
6: -0.0151747, -0.0001643, -0.0150433, -0.0007618, -0.0122960, 0.0135148
7: -0.0049426, 0.0162202, -0.0033146, 0.0165960, -0.0197628, 0.0163889
8: 0.9875705, 1.0010430, 0.9881437, 1.0011992, -0.0121553, 0.0106073
9: -0.0164680, -0.0037660, -0.0167083, -0.0045433, -0.0098134, 0.0116497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095429, upper bound: 0.0094861
time: 1.65 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096831, upper bound: 0.0095864
time: 1.37 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0042293, 0.0156927, 0.0045760, 0.0130352, -0.0088059, 0.0111167
1: -0.0012210, 0.0028280, 0.0000388, 0.0028102, -0.0040311, 0.0027891
2: 0.0045579, 0.0120216, 0.0065340, 0.0118299, -0.0072721, 0.0054876
3: -0.0057881, 0.0004967, -0.0057177, -0.0007229, -0.0047970, 0.0062144
4: -0.0018368, 0.0022290, -0.0015449, 0.0021528, -0.0037267, 0.0032342
5: -0.0000787, 0.0056315, 0.0010178, 0.0054285, -0.0054430, 0.0046137
6: -0.0161729, 0.0000439, -0.0150433, -0.0007618, -0.0138657, 0.0142296
7: -0.0053845, 0.0169859, -0.0033146, 0.0165960, -0.0208542, 0.0177157
8: 0.9873707, 1.0017091, 0.9881437, 1.0011992, -0.0128410, 0.0117226
9: -0.0169576, -0.0035489, -0.0167083, -0.0045433, -0.0106618, 0.0123185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095429, upper bound: 0.0094861
time: 1.56 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096831, upper bound: 0.0095864
time: 1.41 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0045691, 0.0144675, 0.0053906, 0.0113135, -0.0057730, 0.0090060
1: -0.0007298, 0.0028269, 0.0016343, 0.0028713, -0.0036011, 0.0011925
2: 0.0054256, 0.0118337, 0.0079997, 0.0113795, -0.0059539, 0.0033628
3: -0.0057838, -0.0000346, -0.0059595, -0.0024997, -0.0031474, 0.0055850
4: -0.0016094, 0.0022244, -0.0009247, 0.0024145, -0.0033133, 0.0029057
5: 0.0004529, 0.0054325, 0.0015771, 0.0049512, -0.0044173, 0.0032357
6: -0.0155906, -0.0007458, -0.0153187, -0.0026553, -0.0123128, 0.0118179
7: -0.0040505, 0.0169620, 0.0006042, 0.0179344, -0.0186500, 0.0151766
8: 0.9881284, 1.0015384, 0.9899603, 1.0019355, -0.0111133, 0.0107757
9: -0.0169423, -0.0043026, -0.0175641, -0.0066267, -0.0095247, 0.0109602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090989, upper bound: 0.0084262
time: 1.33 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085784
time: 1.36 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0045209, 0.0154973, 0.0053906, 0.0113135, -0.0060773, 0.0101067
1: -0.0009990, 0.0028587, 0.0016343, 0.0028713, -0.0038702, 0.0012243
2: 0.0047343, 0.0118604, 0.0079997, 0.0113795, -0.0066453, 0.0035311
3: -0.0059098, 0.0001847, -0.0059595, -0.0024997, -0.0033523, 0.0058801
4: -0.0016485, 0.0023607, -0.0009247, 0.0024145, -0.0035005, 0.0031276
5: -0.0000057, 0.0054607, 0.0015771, 0.0049512, -0.0049569, 0.0034140
6: -0.0164223, -0.0006339, -0.0153187, -0.0026553, -0.0135640, 0.0125253
7: -0.0043352, 0.0176592, 0.0006042, 0.0179344, -0.0196212, 0.0163114
8: 0.9880210, 1.0021166, 0.9899603, 1.0019355, -0.0117919, 0.0117061
9: -0.0173881, -0.0041688, -0.0175641, -0.0066267, -0.0102503, 0.0115722

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090989, upper bound: 0.0084262
time: 2.98 seconds

## Relational analysis of IS_A2_B1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085784
time: 1.44 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0041483, 0.0158387, 0.0045483, 0.0133689, -0.0092206, 0.0112904
1: -0.0015871, 0.0028988, -0.0001764, 0.0028105, -0.0043977, 0.0030753
2: 0.0043704, 0.0120664, 0.0062764, 0.0118452, -0.0074748, 0.0057899
3: -0.0058518, 0.0009405, -0.0057192, -0.0005287, -0.0051796, 0.0066596
4: -0.0019501, 0.0022392, -0.0015760, 0.0021544, -0.0037553, 0.0033319
5: -0.0000861, 0.0056790, 0.0008914, 0.0054447, -0.0054177, 0.0047875
6: -0.0160125, 0.0002321, -0.0151481, -0.0006975, -0.0138353, 0.0142207
7: -0.0062265, 0.0170380, -0.0035739, 0.0166041, -0.0213886, 0.0183583
8: 0.9871901, 1.0016975, 0.9880820, 1.0012356, -0.0128813, 0.0118961
9: -0.0169909, -0.0031402, -0.0167134, -0.0044344, -0.0109921, 0.0124645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0095048
time: 1.32 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0095963
time: 1.51 seconds

## BFS IS instance: IS_A2_B1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0040651, 0.0170834, 0.0045483, 0.0133689, -0.0093038, 0.0125351
1: -0.0019575, 0.0032129, -0.0001764, 0.0028105, -0.0047680, 0.0033893
2: 0.0035224, 0.0121124, 0.0062764, 0.0118452, -0.0083229, 0.0058359
3: -0.0062016, 0.0012591, -0.0057192, -0.0005287, -0.0056729, 0.0069783
4: -0.0020677, 0.0023756, -0.0015760, 0.0021544, -0.0040219, 0.0035629
5: -0.0006422, 0.0057277, 0.0008914, 0.0054447, -0.0060868, 0.0048363
6: -0.0169340, 0.0004255, -0.0151481, -0.0006975, -0.0153190, 0.0153813
7: -0.0068598, 0.0177355, -0.0035739, 0.0166041, -0.0226358, 0.0195391
8: 0.9870046, 1.0022997, 0.9880820, 1.0012356, -0.0142310, 0.0129266
9: -0.0174369, -0.0026841, -0.0167134, -0.0044344, -0.0117472, 0.0134195

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0095048
time: 1.39 seconds

## Relational analysis of IS_A2_B1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0095963
time: 1.65 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0040168, 0.0142339, 0.0043602, 0.0146312, -0.0106144, 0.0098737
1: -0.0008010, 0.0027326, -0.0006496, 0.0028242, -0.0036253, 0.0033822
2: 0.0055348, 0.0121391, 0.0053422, 0.0119492, -0.0064144, 0.0067969
3: -0.0054106, 0.0002204, -0.0057734, -0.0001035, -0.0053071, 0.0057665
4: -0.0019452, 0.0018204, -0.0017121, 0.0022131, -0.0035431, 0.0034194
5: 0.0006222, 0.0057560, 0.0003621, 0.0055549, -0.0048909, 0.0053939
6: -0.0143038, 0.0005378, -0.0157633, -0.0002603, -0.0137335, 0.0145848
7: -0.0057412, 0.0148961, -0.0044252, 0.0169043, -0.0196797, 0.0190780
8: 0.9868968, 1.0001441, 0.9876626, 1.0015761, -0.0125961, 0.0120100
9: -0.0156213, -0.0032079, -0.0169054, -0.0039785, -0.0112970, 0.0117011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090414, upper bound: 0.0082737
time: 1.35 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0090414, upper bound: 0.0092433
time: 1.34 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0041749, 0.0139640, 0.0043417, 0.0147519, -0.0105770, 0.0096222
1: -0.0005485, 0.0027748, -0.0006811, 0.0028471, -0.0033956, 0.0034560
2: 0.0057857, 0.0120517, 0.0052744, 0.0119594, -0.0061737, 0.0067773
3: -0.0055779, -0.0000362, -0.0058638, -0.0000631, -0.0055148, 0.0055637
4: -0.0018364, 0.0020014, -0.0017269, 0.0023109, -0.0035583, 0.0035596
5: 0.0006836, 0.0056634, 0.0002919, 0.0055657, -0.0048380, 0.0053715
6: -0.0148055, 0.0001704, -0.0161218, -0.0002174, -0.0141060, 0.0147327
7: -0.0050868, 0.0158219, -0.0045213, 0.0174045, -0.0196184, 0.0197951
8: 0.9872494, 1.0007393, 0.9876214, 1.0019246, -0.0127261, 0.0124628
9: -0.0162133, -0.0035693, -0.0172253, -0.0039286, -0.0117566, 0.0117399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091760, upper bound: 0.0085694
time: 1.25 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0091760, upper bound: 0.0094345
time: 1.35 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0043134, 0.0144274, 0.0052362, 0.0118934, -0.0075271, 0.0089877
1: -0.0007351, 0.0028022, 0.0011653, 0.0028391, -0.0035742, 0.0016369
2: 0.0054336, 0.0119751, 0.0074941, 0.0114649, -0.0059616, 0.0044810
3: -0.0056862, 0.0000537, -0.0058321, -0.0020574, -0.0035561, 0.0058720
4: -0.0017600, 0.0021187, -0.0010501, 0.0022766, -0.0036661, 0.0028877
5: 0.0004920, 0.0055823, 0.0013870, 0.0050417, -0.0043988, 0.0040627
6: -0.0152758, -0.0001516, -0.0151313, -0.0022964, -0.0121200, 0.0137300
7: -0.0047883, 0.0164214, -0.0002314, 0.0172291, -0.0203517, 0.0153497
8: 0.9875583, 1.0011804, 0.9896160, 1.0015434, -0.0126172, 0.0105796
9: -0.0165966, -0.0038132, -0.0171131, -0.0062019, -0.0094855, 0.0121052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090954, upper bound: 0.0083966
time: 1.41 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0090954, upper bound: 0.0083819
time: 1.29 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0042996, 0.0145536, 0.0053714, 0.0119121, -0.0074331, 0.0091124
1: -0.0007611, 0.0028245, 0.0013188, 0.0028830, -0.0036441, 0.0015057
2: 0.0053654, 0.0119827, 0.0075522, 0.0113902, -0.0060248, 0.0044293
3: -0.0057743, 0.0000869, -0.0060061, -0.0022443, -0.0034983, 0.0060078
4: -0.0017714, 0.0022140, -0.0009577, 0.0024649, -0.0038009, 0.0029559
5: 0.0004157, 0.0055903, 0.0013174, 0.0049625, -0.0044779, 0.0040642
6: -0.0156260, -0.0001195, -0.0157240, -0.0026106, -0.0124530, 0.0141133
7: -0.0048625, 0.0169090, 0.0002964, 0.0181922, -0.0210326, 0.0156332
8: 0.9875275, 1.0015231, 0.9899173, 1.0021839, -0.0130591, 0.0108678
9: -0.0169084, -0.0037749, -0.0177289, -0.0065082, -0.0097037, 0.0125464

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085694
time: 1.38 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085455
time: 1.33 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0038799, 0.0157958, 0.0043373, 0.0144631, -0.0105832, 0.0114432
1: -0.0015823, 0.0028707, -0.0006213, 0.0027886, -0.0043709, 0.0034920
2: 0.0043831, 0.0122148, 0.0054522, 0.0119619, -0.0075788, 0.0067626
3: -0.0057508, 0.0010083, -0.0056324, -0.0001287, -0.0056055, 0.0066406
4: -0.0021054, 0.0021327, -0.0017227, 0.0020604, -0.0039935, 0.0033774
5: -0.0000476, 0.0058362, 0.0004700, 0.0055683, -0.0054784, 0.0053662
6: -0.0157018, 0.0008560, -0.0152154, -0.0002071, -0.0140764, 0.0158172
7: -0.0069676, 0.0164932, -0.0044675, 0.0161235, -0.0224804, 0.0188144
8: 0.9865916, 1.0013412, 0.9876115, 1.0010362, -0.0139925, 0.0120196
9: -0.0166426, -0.0026381, -0.0164061, -0.0039460, -0.0111557, 0.0132318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0095238
time: 2.69 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0094869
time: 1.57 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0038653, 0.0159198, 0.0044760, 0.0143889, -0.0105236, 0.0114437
1: -0.0016130, 0.0029167, -0.0004930, 0.0028309, -0.0044438, 0.0034097
2: 0.0043166, 0.0122229, 0.0055412, 0.0118852, -0.0075686, 0.0066817
3: -0.0058581, 0.0010465, -0.0057996, -0.0002911, -0.0055670, 0.0068461
4: -0.0021214, 0.0022286, -0.0016298, 0.0022414, -0.0042007, 0.0034375
5: -0.0001234, 0.0058448, 0.0004623, 0.0054870, -0.0055249, 0.0053825
6: -0.0160538, 0.0008900, -0.0157884, -0.0005296, -0.0143052, 0.0165415
7: -0.0070611, 0.0169835, -0.0039339, 0.0170492, -0.0235224, 0.0190583
8: 0.9865590, 1.0016854, 0.9879209, 1.0016537, -0.0147496, 0.0122423
9: -0.0169560, -0.0025796, -0.0169980, -0.0042536, -0.0113490, 0.0139156

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0096313
time: 1.47 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0095820
time: 1.64 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0054482, 0.0108171, 0.0042422, 0.0156230, -0.0101749, 0.0059003
1: 0.0020921, 0.0028816, -0.0014638, 0.0028012, -0.0006128, 0.0043454
2: 0.0083755, 0.0113477, 0.0044969, 0.0120145, -0.0032852, 0.0068508
3: -0.0060004, -0.0029268, -0.0056070, 0.0007043, -0.0067047, 0.0022300
4: -0.0008515, 0.0024588, -0.0018552, 0.0020125, -0.0023578, 0.0039211
5: 0.0017762, 0.0049175, 0.0000490, 0.0056240, -0.0034429, 0.0047678
6: -0.0152239, -0.0027890, -0.0152588, 0.0000140, -0.0135714, 0.0109263
7: 0.0012216, 0.0181609, -0.0055975, 0.0158783, -0.0120698, 0.0221077
8: 0.9900885, 1.0020106, 0.9873995, 1.0009147, -0.0091090, 0.0130507
9: -0.0177089, -0.0068837, -0.0162494, -0.0034751, -0.0129814, 0.0077105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0084262, upper bound: 0.0089651
time: 1.54 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0085784, upper bound: 0.0091760
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0046254, 0.0125823, 0.0037956, 0.0171519, -0.0125265, 0.0087867
1: 0.0004069, 0.0028264, -0.0023784, 0.0031915, -0.0027846, 0.0052048
2: 0.0069181, 0.0118026, 0.0033199, 0.0122614, -0.0053433, 0.0084827
3: -0.0057821, -0.0010477, -0.0059269, 0.0017366, -0.0075187, 0.0046883
4: -0.0014895, 0.0022225, -0.0022846, 0.0020269, -0.0027532, 0.0045071
5: 0.0011414, 0.0053995, -0.0005383, 0.0058856, -0.0047441, 0.0059214
6: -0.0151468, -0.0008766, -0.0157366, 0.0010520, -0.0161988, 0.0125372
7: -0.0028762, 0.0169522, -0.0081590, 0.0159521, -0.0150541, 0.0251112
8: 0.9882538, 1.0013938, 0.9864036, 1.0010835, -0.0101752, 0.0149902
9: -0.0169361, -0.0047356, -0.0162965, -0.0019254, -0.0150106, 0.0090739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0094861, upper bound: 0.0096097
time: 1.65 seconds

## Relational analysis of IS_A2_B2_B1_A1_A1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095864, upper bound: 0.0097647
time: 1.39 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0054074, 0.0115030, 0.0042309, 0.0156403, -0.0102329, 0.0071684
1: 0.0017762, 0.0029092, -0.0014729, 0.0028058, -0.0010098, 0.0043821
2: 0.0079085, 0.0113703, 0.0044836, 0.0120207, -0.0041042, 0.0068867
3: -0.0061097, -0.0026279, -0.0056096, 0.0007041, -0.0068138, 0.0026088
4: -0.0009037, 0.0025771, -0.0018575, 0.0020113, -0.0024189, 0.0042846
5: 0.0014556, 0.0049414, 0.0000423, 0.0056306, -0.0040768, 0.0048031
6: -0.0158876, -0.0026943, -0.0152595, 0.0000403, -0.0152780, 0.0110388
7: 0.0007819, 0.0187659, -0.0055996, 0.0158722, -0.0125916, 0.0238971
8: 0.9899976, 1.0025084, 0.9873742, 1.0009112, -0.0092125, 0.0145005
9: -0.0180958, -0.0067003, -0.0162454, -0.0034676, -0.0141656, 0.0079258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=5, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0083966, upper bound: 0.0089651
time: 1.18 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0085695, upper bound: 0.0091760
time: 1.51 seconds

## BFS IS instance: IS_A2_B2_B1_A1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0045471, 0.0135542, 0.0037832, 0.0171706, -0.0126236, 0.0097709
1: 0.0000974, 0.0028606, -0.0023880, 0.0031956, -0.0030982, 0.0052487
2: 0.0062295, 0.0118459, 0.0033056, 0.0122682, -0.0060387, 0.0085403
3: -0.0059175, -0.0007991, -0.0059293, 0.0017397, -0.0076572, 0.0050688
4: -0.0015513, 0.0023691, -0.0022898, 0.0020257, -0.0028312, 0.0046589
5: 0.0007134, 0.0054454, -0.0005453, 0.0058928, -0.0051794, 0.0059800
6: -0.0160377, -0.0006946, -0.0157378, 0.0010807, -0.0171184, 0.0127392
7: -0.0032882, 0.0177019, -0.0081709, 0.0159460, -0.0156437, 0.0258728
8: 0.9880792, 1.0020339, 0.9863762, 1.0010803, -0.0103627, 0.0156578
9: -0.0174154, -0.0045264, -0.0162927, -0.0019103, -0.0155051, 0.0093425

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_A1

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095030, upper bound: 0.0096097
time: 1.41 seconds

## Relational analysis of IS_A2_B2_B1_A1_A2_A2_A2

### Relational analysis result of IS_A2_B2_B1_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096196, upper bound: 0.0097647
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0043146, 0.0162009, 0.0048442, 0.0139926, -0.0096779, 0.0113567
1: -0.0017834, 0.0029619, -0.0003032, 0.0028284, -0.0045979, 0.0032651
2: 0.0040722, 0.0119744, 0.0058351, 0.0116816, -0.0076094, 0.0061393
3: -0.0057830, 0.0009999, -0.0057897, -0.0006115, -0.0051099, 0.0064689
4: -0.0018663, 0.0020783, -0.0013944, 0.0022308, -0.0033126, 0.0029104
5: -0.0001828, 0.0055816, 0.0005957, 0.0052713, -0.0054541, 0.0049837
6: -0.0156081, -0.0001543, -0.0156436, -0.0013852, -0.0125412, 0.0129856
7: -0.0057906, 0.0162151, -0.0026616, 0.0169946, -0.0191406, 0.0162688
8: 0.9875609, 1.0011661, 0.9887419, 1.0015881, -0.0116185, 0.0104516
9: -0.0164647, -0.0033893, -0.0169632, -0.0050276, -0.0096178, 0.0110575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095975, upper bound: 0.0090740
time: 1.45 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097456, upper bound: 0.0091801
time: 1.44 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042477, 0.0172838, 0.0048338, 0.0140107, -0.0097630, 0.0124500
1: -0.0020404, 0.0032115, -0.0003125, 0.0028282, -0.0048686, 0.0035239
2: 0.0033461, 0.0120114, 0.0058217, 0.0116874, -0.0083413, 0.0061897
3: -0.0060919, 0.0012235, -0.0057889, -0.0006129, -0.0054790, 0.0067780
4: -0.0019578, 0.0022288, -0.0013977, 0.0022299, -0.0034333, 0.0032675
5: -0.0006890, 0.0056207, 0.0005883, 0.0052774, -0.0059664, 0.0050324
6: -0.0165982, 0.0000012, -0.0156475, -0.0013612, -0.0144718, 0.0135057
7: -0.0062785, 0.0169846, -0.0026542, 0.0169900, -0.0197844, 0.0180250
8: 0.9874117, 1.0018321, 0.9887187, 1.0015868, -0.0125275, 0.0119338
9: -0.0169568, -0.0030336, -0.0169602, -0.0050183, -0.0107809, 0.0115386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097188, upper bound: 0.0089750
time: 1.33 seconds

## Relational analysis of IS_A2_B2_B1_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098023, upper bound: 0.0091801
time: 1.47 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0044346, 0.0147011, 0.0037284, 0.0166059, -0.0121712, 0.0109727
1: -0.0006561, 0.0028432, -0.0020477, 0.0030618, -0.0037179, 0.0048909
2: 0.0053155, 0.0119081, 0.0037578, 0.0122985, -0.0069830, 0.0081502
3: -0.0058486, -0.0001197, -0.0058247, 0.0014540, -0.0073026, 0.0057049
4: -0.0016723, 0.0022944, -0.0022691, 0.0020276, -0.0030984, 0.0041899
5: 0.0002902, 0.0055112, -0.0003494, 0.0059249, -0.0056347, 0.0058607
6: -0.0161599, -0.0004333, -0.0156481, 0.0012081, -0.0164469, 0.0135341
7: -0.0042496, 0.0173203, -0.0079409, 0.0159559, -0.0173980, 0.0235572
8: 0.9878286, 1.0018890, 0.9862539, 1.0010656, -0.0111438, 0.0150654
9: -0.0171714, -0.0041072, -0.0162990, -0.0020370, -0.0139560, 0.0102452

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096129, upper bound: 0.0096824
time: 1.31 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096261, upper bound: 0.0096824
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0035136, 0.0173036, 0.0037284, 0.0166059, -0.0130923, 0.0135752
1: -0.0024006, 0.0032426, -0.0020477, 0.0030618, -0.0054624, 0.0052903
2: 0.0032555, 0.0124173, 0.0037578, 0.0122985, -0.0090430, 0.0086595
3: -0.0060258, 0.0018815, -0.0058247, 0.0014540, -0.0074799, 0.0076079
4: -0.0024668, 0.0021059, -0.0022691, 0.0020276, -0.0036466, 0.0036430
5: -0.0006450, 0.0060508, -0.0003494, 0.0059249, -0.0065699, 0.0064002
6: -0.0161340, 0.0017075, -0.0156481, 0.0012081, -0.0157380, 0.0158335
7: -0.0090524, 0.0163561, -0.0079409, 0.0159559, -0.0208701, 0.0207199
8: 0.9857748, 1.0013919, 0.9862539, 1.0010656, -0.0137423, 0.0134551
9: -0.0165549, -0.0013341, -0.0162990, -0.0020370, -0.0121626, 0.0122494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 61

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099508, upper bound: 0.0101653
time: 1.69 seconds

## Relational analysis of IS_A2_B2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0099508, upper bound: 0.0102754
time: 1.58 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044582, 0.0133388, 0.0037160, 0.0180969, -0.0136387, 0.0096228
1: -0.0000955, 0.0028284, -0.0028501, 0.0034200, -0.0035156, 0.0056785
2: 0.0063287, 0.0118950, 0.0026308, 0.0123054, -0.0059767, 0.0092642
3: -0.0057899, -0.0005402, -0.0062097, 0.0022211, -0.0080110, 0.0056696
4: -0.0016292, 0.0022309, -0.0024245, 0.0021645, -0.0035993, 0.0042239
5: 0.0008582, 0.0054974, -0.0009472, 0.0059322, -0.0050740, 0.0064446
6: -0.0154157, -0.0004881, -0.0163554, 0.0012370, -0.0161126, 0.0154031
7: -0.0038296, 0.0169956, -0.0090039, 0.0166557, -0.0197134, 0.0240669
8: 0.9878811, 1.0014950, 0.9858283, 1.0016191, -0.0130672, 0.0156667
9: -0.0169638, -0.0042614, -0.0167465, -0.0013903, -0.0142448, 0.0118646

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095048, upper bound: 0.0098318
time: 1.65 seconds

## Relational analysis of IS_A2_B2_B2_A1_A1_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095963, upper bound: 0.0099764
time: 1.33 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0048248, 0.0130414, 0.0046500, 0.0152878, -0.0104630, 0.0083914
1: 0.0004456, 0.0028593, -0.0010167, 0.0028577, -0.0024121, 0.0038760
2: 0.0066193, 0.0116924, 0.0048612, 0.0117890, -0.0051697, 0.0068312
3: -0.0059124, -0.0012341, -0.0059060, 0.0001348, -0.0060471, 0.0046327
4: -0.0013539, 0.0023635, -0.0015686, 0.0023566, -0.0033809, 0.0038819
5: 0.0009037, 0.0052827, 0.0000785, 0.0053851, -0.0044814, 0.0050897
6: -0.0158448, -0.0013402, -0.0163799, -0.0009338, -0.0147185, 0.0138354
7: -0.0021200, 0.0176736, -0.0039413, 0.0176381, -0.0183007, 0.0216149
8: 0.9886985, 1.0019618, 0.9883088, 1.0021149, -0.0121760, 0.0133498
9: -0.0173973, -0.0051841, -0.0173746, -0.0044303, -0.0128297, 0.0111292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_A1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088247, upper bound: 0.0086794
time: 1.42 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B1_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0089815, upper bound: 0.0089565
time: 1.32 seconds

## BFS IS instance: IS_A2_B2_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0043972, 0.0143586, 0.0037077, 0.0181090, -0.0137118, 0.0106509
1: -0.0003976, 0.0028627, -0.0028574, 0.0034226, -0.0038202, 0.0057201
2: 0.0056148, 0.0119288, 0.0026207, 0.0123100, -0.0066952, 0.0093080
3: -0.0059256, -0.0003048, -0.0062109, 0.0022231, -0.0081488, 0.0059061
4: -0.0016781, 0.0023779, -0.0024275, 0.0021632, -0.0036558, 0.0045715
5: 0.0004106, 0.0055332, -0.0009511, 0.0059371, -0.0055265, 0.0064843
6: -0.0163300, -0.0003463, -0.0163500, 0.0012563, -0.0175862, 0.0155143
7: -0.0041619, 0.0177470, -0.0090092, 0.0166493, -0.0201895, 0.0258006
8: 0.9877451, 1.0021480, 0.9858057, 1.0016133, -0.0131734, 0.0163423
9: -0.0174443, -0.0040960, -0.0167424, -0.0013808, -0.0153791, 0.0120636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_A1

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0095238, upper bound: 0.0098318
time: 1.41 seconds

## Relational analysis of IS_A2_B2_B2_A1_A2_B2_A2

### Relational analysis result of IS_A2_B2_B2_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096313, upper bound: 0.0099763
time: 1.45 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0042576, 0.0166716, 0.0046400, 0.0152770, -0.0110193, 0.0120316
1: -0.0020510, 0.0030778, -0.0010107, 0.0028580, -0.0049090, 0.0040885
2: 0.0037056, 0.0120059, 0.0048693, 0.0117945, -0.0080890, 0.0071366
3: -0.0058761, 0.0012576, -0.0059070, 0.0001334, -0.0060096, 0.0070624
4: -0.0019489, 0.0020801, -0.0015787, 0.0023577, -0.0037376, 0.0030916
5: -0.0003669, 0.0056149, 0.0000824, 0.0053910, -0.0057579, 0.0055325
6: -0.0157394, -0.0000220, -0.0163807, -0.0009106, -0.0131123, 0.0147271
7: -0.0063033, 0.0162244, -0.0039861, 0.0176439, -0.0214061, 0.0176563
8: 0.9874339, 1.0012066, 0.9882864, 1.0021185, -0.0134603, 0.0108928
9: -0.0164707, -0.0030785, -0.0173783, -0.0043961, -0.0102433, 0.0125075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 148

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096000, upper bound: 0.0091409
time: 1.31 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0097555, upper bound: 0.0092519
time: 1.84 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0042001, 0.0177406, 0.0046500, 0.0152878, -0.0110877, 0.0130906
1: -0.0022973, 0.0033087, -0.0010167, 0.0028577, -0.0051551, 0.0043253
2: 0.0030010, 0.0120377, 0.0048612, 0.0117890, -0.0087880, 0.0071765
3: -0.0061704, 0.0014739, -0.0059060, 0.0001348, -0.0063052, 0.0073626
4: -0.0020303, 0.0022308, -0.0015686, 0.0023566, -0.0038557, 0.0034449
5: -0.0008699, 0.0056486, 0.0000785, 0.0053851, -0.0062550, 0.0055702
6: -0.0167337, 0.0001118, -0.0163799, -0.0009338, -0.0150452, 0.0152403
7: -0.0067697, 0.0169946, -0.0039413, 0.0176381, -0.0220362, 0.0193892
8: 0.9872014, 1.0018759, 0.9883088, 1.0021149, -0.0143585, 0.0123695
9: -0.0169632, -0.0027482, -0.0173746, -0.0044303, -0.0113937, 0.0129773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=4, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096646, upper bound: 0.0091409
time: 1.45 seconds

## Relational analysis of IS_A2_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0098128, upper bound: 0.0092519
time: 1.27 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0043804, 0.0151003, 0.0035079, 0.0181734, -0.0137930, 0.0115923
1: -0.0008937, 0.0028438, -0.0028728, 0.0034460, -0.0043397, 0.0057166
2: 0.0050108, 0.0119381, 0.0025847, 0.0124204, -0.0074096, 0.0093533
3: -0.0058507, 0.0001174, -0.0062359, 0.0022981, -0.0081488, 0.0063533
4: -0.0017252, 0.0022968, -0.0025519, 0.0021720, -0.0034475, 0.0044646
5: 0.0001428, 0.0055430, -0.0009869, 0.0060541, -0.0059113, 0.0065299
6: -0.0162903, -0.0003072, -0.0164576, 0.0017206, -0.0176051, 0.0149637
7: -0.0046517, 0.0173323, -0.0096254, 0.0166941, -0.0193546, 0.0252621
8: 0.9877076, 1.0019357, 0.9853576, 1.0016712, -0.0123822, 0.0165781
9: -0.0171791, -0.0039238, -0.0167710, -0.0009750, -0.0150279, 0.0113994

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096265, upper bound: 0.0097841
time: 1.26 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A1_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096365, upper bound: 0.0097841
time: 1.67 seconds

## BFS IS instance: IS_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0034470, 0.0177955, 0.0035079, 0.0181734, -0.0147264, 0.0142875
1: -0.0026788, 0.0033527, -0.0028728, 0.0034460, -0.0061248, 0.0062255
2: 0.0028738, 0.0124541, 0.0025847, 0.0124204, -0.0095467, 0.0098694
3: -0.0061147, 0.0021471, -0.0062359, 0.0022981, -0.0084127, 0.0083830
4: -0.0025524, 0.0021080, -0.0025519, 0.0021720, -0.0040379, 0.0039191
5: -0.0008345, 0.0060898, -0.0009869, 0.0060541, -0.0068886, 0.0070767
6: -0.0162767, 0.0018622, -0.0164576, 0.0017206, -0.0169495, 0.0173972
7: -0.0095620, 0.0163671, -0.0096254, 0.0166941, -0.0229886, 0.0224341
8: 0.9854326, 1.0014368, 0.9853576, 1.0016712, -0.0154486, 0.0152182
9: -0.0165619, -0.0010101, -0.0167710, -0.0009750, -0.0132388, 0.0135918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=6, inp2_unstable=6, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 211

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A1

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096265, upper bound: 0.0101050
time: 1.26 seconds

## Relational analysis of IS_A2_B2_B2_A2_B2_A2_A2

### Relational analysis result of IS_A2_B2_B2_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0096365, upper bound: 0.0101050
time: 1.40 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.59 seconds
IS_A1_B1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085450
IS_A1_B1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085450
IS_A1_B1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093570
IS_A1_B1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093570
IS_A1_B1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094243
IS_A1_B1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094243
IS_A1_B1_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085773
IS_A1_B1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085773
IS_A1_B1_B1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0094803
IS_A1_B1_B1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0094803
IS_A1_B1_B1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095533
IS_A1_B1_B1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095534
IS_A1_B1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085452
IS_A1_B1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091977, upper bound: 0.0085067
IS_A1_B1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093919
IS_A1_B1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095827, upper bound: 0.0093392
IS_A1_B1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094800
IS_A1_B1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097172, upper bound: 0.0094128
IS_A1_B1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085695
IS_A1_B1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092082, upper bound: 0.0085447
IS_A1_B1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0095006
IS_A1_B1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096125, upper bound: 0.0094619
IS_A1_B1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095926
IS_A1_B1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097557, upper bound: 0.0095386
IS_A1_B2_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092993, upper bound: 0.0092805
IS_A1_B2_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092993, upper bound: 0.0093112
IS_A1_B2_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094492, upper bound: 0.0093717
IS_A1_B2_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094492, upper bound: 0.0094072
IS_A1_B2_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090251, upper bound: 0.0096919
IS_A1_B2_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090251, upper bound: 0.0099709
IS_A1_B2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092170, upper bound: 0.0097656
IS_A1_B2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0092170, upper bound: 0.0099873
IS_A1_B2_A1_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0084155, upper bound: 0.0091430
IS_A1_B2_A1_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0085753, upper bound: 0.0093604
IS_A1_B2_A1_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0093631, upper bound: 0.0097749
IS_A1_B2_A1_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094471, upper bound: 0.0098929
IS_A1_B2_A1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094337, upper bound: 0.0090380
IS_A1_B2_A1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095045, upper bound: 0.0092252
IS_A1_B2_A1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094140, upper bound: 0.0097301
IS_A1_B2_A1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094140, upper bound: 0.0101791
IS_A1_B2_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094617, upper bound: 0.0092862
IS_A1_B2_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094617, upper bound: 0.0093173
IS_A1_B2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096282, upper bound: 0.0093854
IS_A1_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096282, upper bound: 0.0094197
IS_A1_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095318, upper bound: 0.0092862
IS_A1_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095318, upper bound: 0.0092862
IS_A1_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096833, upper bound: 0.0093854
IS_A1_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096833, upper bound: 0.0093854
IS_A1_B2_A2_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094092, upper bound: 0.0097976
IS_A1_B2_A2_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094761, upper bound: 0.0099290
IS_A1_B2_A2_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094446, upper bound: 0.0097976
IS_A1_B2_A2_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095296, upper bound: 0.0099291
IS_A1_B2_A2_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095518, upper bound: 0.0097550
IS_A1_B2_A2_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095546, upper bound: 0.0097550
IS_A1_B2_A2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099087, upper bound: 0.0102766
IS_A1_B2_A2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099087, upper bound: 0.0103429
IS_A2_B1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0089136, upper bound: 0.0084262
IS_A2_B1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091071, upper bound: 0.0085784
IS_A2_B1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0089136, upper bound: 0.0084262
IS_A2_B1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091071, upper bound: 0.0085784
IS_A2_B1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095429, upper bound: 0.0094861
IS_A2_B1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096831, upper bound: 0.0095864
IS_A2_B1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095429, upper bound: 0.0094861
IS_A2_B1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096831, upper bound: 0.0095864
IS_A2_B1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090989, upper bound: 0.0084262
IS_A2_B1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085784
IS_A2_B1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090989, upper bound: 0.0084262
IS_A2_B1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085784
IS_A2_B1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0095048
IS_A2_B1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0095963
IS_A2_B1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0095048
IS_A2_B1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0095963
IS_A2_B1_B2_A1_A2_A1_B1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090414, upper bound: 0.0082737
IS_A2_B1_B2_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090414, upper bound: 0.0092433
IS_A2_B1_B2_A1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091760, upper bound: 0.0085694
IS_A2_B1_B2_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0091760, upper bound: 0.0094345
IS_A2_B1_B2_A2_B1_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090954, upper bound: 0.0083966
IS_A2_B1_B2_A2_B1_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0090954, upper bound: 0.0083819
IS_A2_B1_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085694
IS_A2_B1_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0093171, upper bound: 0.0085455
IS_A2_B1_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0095238
IS_A2_B1_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097614, upper bound: 0.0094869
IS_A2_B1_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0096313
IS_A2_B1_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099136, upper bound: 0.0095820
IS_A2_B2_B1_A1_A1_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0084262, upper bound: 0.0089651
IS_A2_B2_B1_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0085784, upper bound: 0.0091760
IS_A2_B2_B1_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0094861, upper bound: 0.0096097
IS_A2_B2_B1_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095864, upper bound: 0.0097647
IS_A2_B2_B1_A1_A2_A1_A1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0083966, upper bound: 0.0089651
IS_A2_B2_B1_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0085695, upper bound: 0.0091760
IS_A2_B2_B1_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095030, upper bound: 0.0096097
IS_A2_B2_B1_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096196, upper bound: 0.0097647
IS_A2_B2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095975, upper bound: 0.0090740
IS_A2_B2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097456, upper bound: 0.0091801
IS_A2_B2_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097188, upper bound: 0.0089750
IS_A2_B2_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0098023, upper bound: 0.0091801
IS_A2_B2_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096129, upper bound: 0.0096824
IS_A2_B2_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096261, upper bound: 0.0096824
IS_A2_B2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099508, upper bound: 0.0101653
IS_A2_B2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0099508, upper bound: 0.0102754
IS_A2_B2_B2_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095048, upper bound: 0.0098318
IS_A2_B2_B2_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095963, upper bound: 0.0099764
IS_A2_B2_B2_A1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0088247, upper bound: 0.0086794
IS_A2_B2_B2_A1_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0089815, upper bound: 0.0089565
IS_A2_B2_B2_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0095238, upper bound: 0.0098318
IS_A2_B2_B2_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096313, upper bound: 0.0099763
IS_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096000, upper bound: 0.0091409
IS_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0097555, upper bound: 0.0092519
IS_A2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096646, upper bound: 0.0091409
IS_A2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0098128, upper bound: 0.0092519
IS_A2_B2_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096265, upper bound: 0.0097841
IS_A2_B2_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096365, upper bound: 0.0097841
IS_A2_B2_B2_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096265, upper bound: 0.0101050
IS_A2_B2_B2_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.59
Output dim: 8, lower bound: -0.0096365, upper bound: 0.0101050

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046552, 0.0128636, 0.0056549, 0.0105722, -0.0045400, 0.0069273
1: 0.0000628, 0.0027685, 0.0021393, 0.0028497, -0.0027427, 0.0005021
2: 0.0066110, 0.0117861, 0.0085148, 0.0112334, -0.0045598, 0.0025100
3: -0.0055530, -0.0008008, -0.0058741, -0.0030623, -0.0019872, 0.0045406
4: -0.0014896, 0.0019744, -0.0007218, 0.0023221, -0.0029845, 0.0021512
5: 0.0011419, 0.0053820, 0.0019159, 0.0047964, -0.0034223, 0.0026595
6: -0.0143646, -0.0009460, -0.0146987, -0.0032695, -0.0093068, 0.0105522
7: -0.0030038, 0.0156839, 0.0018961, 0.0174617, -0.0164068, 0.0110007
8: 0.9883204, 1.0005565, 0.9905495, 1.0015142, -0.0101234, 0.0081149
9: -0.0161251, -0.0047264, -0.0172618, -0.0073087, -0.0070342, 0.0098434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 61
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_A1_B1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0089948, upper bound: 0.0083322
time: 1.44 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_A1_B2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0089948, upper bound: 0.0083102
time: 1.21 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0045850, 0.0141528, 0.0056549, 0.0105722, -0.0049292, 0.0084807
1: -0.0003489, 0.0028030, 0.0021393, 0.0028497, -0.0031568, 0.0005597
2: 0.0057080, 0.0118249, 0.0085148, 0.0112334, -0.0055254, 0.0027252
3: -0.0056895, -0.0004582, -0.0058741, -0.0030623, -0.0022152, 0.0049893
4: -0.0015555, 0.0021223, -0.0007218, 0.0023221, -0.0032434, 0.0023981
5: 0.0005734, 0.0054231, 0.0019159, 0.0047964, -0.0041587, 0.0028875
6: -0.0153326, -0.0007829, -0.0146987, -0.0032695, -0.0108088, 0.0114568
7: -0.0035199, 0.0164397, 0.0018961, 0.0174617, -0.0178486, 0.0122630
8: 0.9881639, 1.0012139, 0.9905495, 1.0015142, -0.0109912, 0.0091869
9: -0.0166084, -0.0044990, -0.0172618, -0.0073087, -0.0078413, 0.0106979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 184
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 148
type: B, layer: 1, pos: 182
type: A, layer: 1, pos: 182

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 94

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088812, upper bound: 0.0085450
time: 1.40 seconds

## Relational analysis of IS_A1_B1_B1_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B1_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088812, upper bound: 0.0085450
time: 1.48 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0042585, 0.0140619, 0.0047321, 0.0114691, -0.0069161, 0.0091781
1: -0.0007814, 0.0027486, 0.0008648, 0.0027455, -0.0035269, 0.0018118
2: 0.0056425, 0.0120054, 0.0076660, 0.0117436, -0.0061011, 0.0042533
3: -0.0054739, 0.0001302, -0.0054618, -0.0015188, -0.0033068, 0.0055404
4: -0.0017966, 0.0018889, -0.0013860, 0.0018758, -0.0032703, 0.0024426
5: 0.0006835, 0.0056144, 0.0016952, 0.0053370, -0.0044035, 0.0036756
6: -0.0144478, -0.0000240, -0.0136535, -0.0011247, -0.0106590, 0.0120892
7: -0.0050084, 0.0152463, -0.0020986, 0.0151793, -0.0183845, 0.0131482
8: 0.9874359, 1.0003690, 0.9884919, 1.0000950, -0.0111129, 0.0089335
9: -0.0158452, -0.0036924, -0.0158024, -0.0050924, -0.0080350, 0.0108160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=4, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0086052, upper bound: 0.0087354
time: 1.51 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0086052, upper bound: 0.0092286
time: 1.34 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0041793, 0.0153419, 0.0047321, 0.0114691, -0.0072202, 0.0106091
1: -0.0011352, 0.0027844, 0.0008648, 0.0027455, -0.0038807, 0.0018790
2: 0.0047560, 0.0120492, 0.0076660, 0.0117436, -0.0069876, 0.0043832
3: -0.0056157, 0.0004264, -0.0054618, -0.0015188, -0.0035730, 0.0058882
4: -0.0018649, 0.0020424, -0.0013860, 0.0018758, -0.0034634, 0.0027308
5: 0.0001108, 0.0056608, 0.0016952, 0.0053370, -0.0051101, 0.0038537
6: -0.0154792, 0.0001601, -0.0136535, -0.0011247, -0.0123638, 0.0127959
7: -0.0054807, 0.0160314, -0.0020986, 0.0151793, -0.0194332, 0.0146216
8: 0.9872592, 1.0010616, 0.9884919, 1.0000950, -0.0117909, 0.0102080
9: -0.0163472, -0.0034597, -0.0158024, -0.0050924, -0.0089771, 0.0114499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 122
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0086052, upper bound: 0.0087354
time: 1.37 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0086052, upper bound: 0.0092286
time: 1.31 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0042438, 0.0141657, 0.0048520, 0.0114465, -0.0067830, 0.0092571
1: -0.0008013, 0.0027718, 0.0010427, 0.0027898, -0.0035911, 0.0016581
2: 0.0055925, 0.0120136, 0.0077519, 0.0116773, -0.0060848, 0.0041079
3: -0.0055658, 0.0001592, -0.0056372, -0.0017140, -0.0032209, 0.0056530
4: -0.0018083, 0.0019883, -0.0013014, 0.0020656, -0.0033772, 0.0025086
5: 0.0006156, 0.0056230, 0.0016417, 0.0052667, -0.0044555, 0.0036563
6: -0.0148003, 0.0000102, -0.0142530, -0.0014035, -0.0109395, 0.0124121
7: -0.0050821, 0.0157548, -0.0015961, 0.0161502, -0.0189307, 0.0134114
8: 0.9874031, 1.0007166, 0.9887593, 1.0007461, -0.0114838, 0.0092212
9: -0.0161704, -0.0036529, -0.0164232, -0.0053742, -0.0082461, 0.0111636

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=23, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 184
type: A, layer: 1, pos: 184
type: B, layer: 1, pos: 122
type: A, layer: 1, pos: 122
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 69
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 61
type: A, layer: 1, pos: 61
type: B, layer: 1, pos: 144
type: B, layer: 1, pos: 148
type: A, layer: 1, pos: 144
type: A, layer: 1, pos: 148
type: A, layer: 1, pos: 182
type: B, layer: 1, pos: 182

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1_B1_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0088678, upper bound: 0.0088661
time: 1.30 seconds

## Relational analysis of IS_A1_B1_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_B1_B1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0088678, upper bound: 0.0093014
time: 1.32 seconds

## BFS IS instance: IS_A1_B1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0041643, 0.0154294, 0.0048520, 0.0114465, -0.0071400, 0.0105774
1: -0.0011542, 0.0028063, 0.0010427, 0.0027898, -0.0039440, 0.0017218
2: 0.0047122, 0.0120576, 0.0077519, 0.0116773, -0.0069651, 0.0043053
3: -0.0057024, 0.0004560, -0.0056372, -0.0017140, -0.0034728, 0.0060248
4: -0.0018771, 0.0021362, -0.0013014, 0.0020656, -0.0036086, 0.0027813
5: 0.0000523, 0.0056696, 0.0016417, 0.0052667, -0.0051539, 0.0038654
6: -0.0157946, 0.0001951, -0.0142530, -0.0014035, -0.0125883, 0.0132420
7: -0.0055589, 0.0165112, -0.0015961, 0.0161502, -0.0201717, 0.0148059
8: 0.9872256, 1.0013833, 0.9887593, 1.0007461, -0.0122799, 0.0104188
9: -0.0166541, -0.0034185, -0.0164232, -0.0053742, -0.0091377, 0.0119261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=5, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 4.01 + 596.97 = 600.98 seconds
