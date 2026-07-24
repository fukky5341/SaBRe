## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00206416


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0058981, 0.0086795, 0.0058981, 0.0086795, -0.0023129, 0.0023129)
1: (0.0021744, 0.0025762, 0.0021744, 0.0025762, -0.0003341, 0.0003341)
2: (0.0095612, 0.0110989, 0.0095612, 0.0110989, -0.0012787, 0.0012787)
3: (-0.0047918, -0.0032014, -0.0047918, -0.0032014, -0.0013225, 0.0013225)
4: (-0.0005712, 0.0011505, -0.0005712, 0.0011505, -0.0014317, 0.0014317)
5: (0.0030246, 0.0046539, 0.0030246, 0.0046539, -0.0013549, 0.0013549)
6: (-0.0102995, -0.0038349, -0.0102995, -0.0038349, -0.0053757, 0.0053757)
7: (0.0026661, 0.0114704, 0.0026661, 0.0114704, -0.0073213, 0.0073213)
8: (0.9910919, 0.9972938, 0.9910919, 0.9972938, -0.0051573, 0.0051573)
9: (-0.0134308, -0.0078011, -0.0134308, -0.0078011, -0.0046814, 0.0046814)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 1.62 = 3.28 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0028979, upper bound: 0.0028979

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025557, upper bound: 0.0026458
time: 0.68 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026719, upper bound: 0.0026719
time: 0.74 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.60 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 8, lower bound: -0.0025557, upper bound: 0.0026458
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.60
Output dim: 8, lower bound: -0.0026719, upper bound: 0.0026719

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0062839, 0.0088606, 0.0060258, 0.0086749, -0.0017726, 0.0020855
1: 0.0022301, 0.0026024, 0.0021929, 0.0025756, -0.0002561, 0.0003013
2: 0.0094611, 0.0108857, 0.0095638, 0.0110283, -0.0011530, 0.0009800
3: -0.0048954, -0.0034220, -0.0047892, -0.0032744, -0.0011925, 0.0010136
4: -0.0003325, 0.0012626, -0.0004922, 0.0011476, -0.0010972, 0.0012909
5: 0.0029185, 0.0044280, 0.0030273, 0.0045791, -0.0012217, 0.0010384
6: -0.0107205, -0.0047314, -0.0102888, -0.0041317, -0.0048472, 0.0041199
7: 0.0038871, 0.0120437, 0.0030703, 0.0114557, -0.0056110, 0.0066015
8: 0.9919520, 0.9976977, 0.9913767, 0.9972835, -0.0039525, 0.0046502
9: -0.0137974, -0.0085819, -0.0134215, -0.0080595, -0.0042212, 0.0035878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024747
time: 0.57 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024751
time: 0.58 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0059832, 0.0086759, 0.0058981, 0.0086795, -0.0017580, 0.0023097
1: 0.0021867, 0.0025757, 0.0021744, 0.0025762, -0.0002540, 0.0003337
2: 0.0095632, 0.0110519, 0.0095612, 0.0110989, -0.0012770, 0.0009719
3: -0.0047898, -0.0032501, -0.0047918, -0.0032014, -0.0013207, 0.0010052
4: -0.0005186, 0.0011483, -0.0005712, 0.0011505, -0.0010882, 0.0014298
5: 0.0030267, 0.0046041, 0.0030246, 0.0046539, -0.0013530, 0.0010298
6: -0.0102912, -0.0040327, -0.0102995, -0.0038349, -0.0053685, 0.0040860
7: 0.0029354, 0.0114590, 0.0026661, 0.0114704, -0.0055648, 0.0073114
8: 0.9912817, 0.9972858, 0.9910919, 0.9972938, -0.0039199, 0.0051503
9: -0.0134236, -0.0079733, -0.0134308, -0.0078011, -0.0046751, 0.0035583

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026458, upper bound: 0.0025432
time: 0.75 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0026458, upper bound: 0.0026718
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.17 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024747
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024751
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0026458, upper bound: 0.0025432
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.17
Output dim: 8, lower bound: -0.0026458, upper bound: 0.0026718

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0062889, 0.0087917, 0.0060258, 0.0086749, -0.0017673, 0.0020065
1: 0.0022309, 0.0025924, 0.0021929, 0.0025756, -0.0002553, 0.0002899
2: 0.0094992, 0.0108829, 0.0095638, 0.0110283, -0.0011093, 0.0009771
3: -0.0048560, -0.0034249, -0.0047892, -0.0032744, -0.0011473, 0.0010105
4: -0.0003294, 0.0012199, -0.0004922, 0.0011476, -0.0010940, 0.0012421
5: 0.0029589, 0.0044250, 0.0030273, 0.0045791, -0.0011754, 0.0010353
6: -0.0105604, -0.0047431, -0.0102888, -0.0041317, -0.0046637, 0.0041076
7: 0.0039030, 0.0118256, 0.0030703, 0.0114557, -0.0055942, 0.0063515
8: 0.9919632, 0.9975441, 0.9913767, 0.9972835, -0.0039407, 0.0044741
9: -0.0136579, -0.0085920, -0.0134215, -0.0080595, -0.0040613, 0.0035771

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0023702
time: 0.59 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024747
time: 0.57 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0061468, 0.0087048, 0.0060288, 0.0086208, -0.0019405, 0.0020165
1: 0.0022103, 0.0025799, 0.0021933, 0.0025678, -0.0002803, 0.0002913
2: 0.0095472, 0.0109614, 0.0095937, 0.0110267, -0.0011149, 0.0010728
3: -0.0048063, -0.0033436, -0.0047582, -0.0032761, -0.0011530, 0.0011096
4: -0.0004173, 0.0011661, -0.0004904, 0.0011141, -0.0012012, 0.0012482
5: 0.0030098, 0.0045083, 0.0030590, 0.0045774, -0.0011812, 0.0011367
6: -0.0103583, -0.0044129, -0.0101630, -0.0041386, -0.0046868, 0.0045102
7: 0.0034533, 0.0115504, 0.0030797, 0.0112845, -0.0061425, 0.0063831
8: 0.9916464, 0.9973502, 0.9913833, 0.9971629, -0.0043269, 0.0044964
9: -0.0134820, -0.0083045, -0.0133119, -0.0080656, -0.0040815, 0.0039277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0023649
time: 0.59 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024751
time: 0.59 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0059832, 0.0086759, 0.0062839, 0.0088606, -0.0022280, 0.0017733
1: 0.0021867, 0.0025757, 0.0022301, 0.0026024, -0.0003219, 0.0002562
2: 0.0095632, 0.0110519, 0.0094611, 0.0108857, -0.0009804, 0.0012318
3: -0.0047898, -0.0032501, -0.0048954, -0.0034220, -0.0010140, 0.0012740
4: -0.0005186, 0.0011483, -0.0003325, 0.0012626, -0.0013792, 0.0010977
5: 0.0030267, 0.0046041, 0.0029185, 0.0044280, -0.0010388, 0.0013051
6: -0.0102912, -0.0040327, -0.0107205, -0.0047314, -0.0041217, 0.0051785
7: 0.0029354, 0.0114590, 0.0038871, 0.0120437, -0.0070526, 0.0056134
8: 0.9912817, 0.9972858, 0.9919520, 0.9976977, -0.0049680, 0.0039542
9: -0.0134236, -0.0079733, -0.0137974, -0.0085819, -0.0035893, 0.0045096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024747, upper bound: 0.0023649
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024751, upper bound: 0.0023649
time: 0.76 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0059832, 0.0086759, 0.0059832, 0.0086759, -0.0017548, 0.0017548
1: 0.0021867, 0.0025757, 0.0021867, 0.0025757, -0.0002535, 0.0002535
2: 0.0095632, 0.0110519, 0.0095632, 0.0110519, -0.0009702, 0.0009702
3: -0.0047898, -0.0032501, -0.0047898, -0.0032501, -0.0010034, 0.0010034
4: -0.0005186, 0.0011483, -0.0005186, 0.0011483, -0.0010863, 0.0010863
5: 0.0030267, 0.0046041, 0.0030267, 0.0046041, -0.0010280, 0.0010280
6: -0.0102912, -0.0040327, -0.0102912, -0.0040327, -0.0040787, 0.0040787
7: 0.0029354, 0.0114590, 0.0029354, 0.0114590, -0.0055549, 0.0055549
8: 0.9912817, 0.9972858, 0.9912817, 0.9972858, -0.0039130, 0.0039130
9: -0.0134236, -0.0079733, -0.0134236, -0.0079733, -0.0035520, 0.0035520

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024751, upper bound: 0.0024150
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024751, upper bound: 0.0024129
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.16 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0023702
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024747
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0023649
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0023731, upper bound: 0.0024751
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0024747, upper bound: 0.0023649
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0024751, upper bound: 0.0023649
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0024751, upper bound: 0.0024150
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.16
Output dim: 8, lower bound: -0.0024751, upper bound: 0.0024129

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062889, 0.0087917, 0.0062839, 0.0088606, -0.0017146, 0.0016409
1: 0.0022309, 0.0025924, 0.0022301, 0.0026024, -0.0002477, 0.0002371
2: 0.0094992, 0.0108829, 0.0094611, 0.0108857, -0.0009072, 0.0009480
3: -0.0048560, -0.0034249, -0.0048954, -0.0034220, -0.0009383, 0.0009804
4: -0.0003294, 0.0012199, -0.0003325, 0.0012626, -0.0010614, 0.0010158
5: 0.0029589, 0.0044250, 0.0029185, 0.0044280, -0.0009613, 0.0010044
6: -0.0105604, -0.0047431, -0.0107205, -0.0047314, -0.0038140, 0.0039852
7: 0.0039030, 0.0118256, 0.0038871, 0.0120437, -0.0054275, 0.0051943
8: 0.9919632, 0.9975441, 0.9919520, 0.9976977, -0.0038233, 0.0036590
9: -0.0136579, -0.0085920, -0.0137974, -0.0085819, -0.0033214, 0.0034705

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023682, upper bound: 0.0023702
time: 0.57 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023682, upper bound: 0.0023702
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062889, 0.0087917, 0.0059832, 0.0086759, -0.0017680, 0.0021490
1: 0.0022309, 0.0025924, 0.0021867, 0.0025757, -0.0002554, 0.0003105
2: 0.0094992, 0.0108829, 0.0095632, 0.0110519, -0.0011881, 0.0009775
3: -0.0048560, -0.0034249, -0.0047898, -0.0032501, -0.0012288, 0.0010110
4: -0.0003294, 0.0012199, -0.0005186, 0.0011483, -0.0010944, 0.0013303
5: 0.0029589, 0.0044250, 0.0030267, 0.0046041, -0.0012589, 0.0010357
6: -0.0105604, -0.0047431, -0.0102912, -0.0040327, -0.0049949, 0.0041094
7: 0.0039030, 0.0118256, 0.0029354, 0.0114590, -0.0055966, 0.0068027
8: 0.9919632, 0.9975441, 0.9912817, 0.9972858, -0.0039424, 0.0047919
9: -0.0136579, -0.0085920, -0.0134236, -0.0079733, -0.0043498, 0.0035786

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022215
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0023206
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061468, 0.0087048, 0.0062870, 0.0088087, -0.0018911, 0.0016508
1: 0.0022103, 0.0025799, 0.0022306, 0.0025949, -0.0002732, 0.0002385
2: 0.0095472, 0.0109614, 0.0094898, 0.0108839, -0.0009127, 0.0010455
3: -0.0048063, -0.0033436, -0.0048657, -0.0034238, -0.0009440, 0.0010813
4: -0.0004173, 0.0011661, -0.0003305, 0.0012304, -0.0011706, 0.0010219
5: 0.0030098, 0.0045083, 0.0029489, 0.0044261, -0.0009671, 0.0011078
6: -0.0103583, -0.0044129, -0.0105998, -0.0047388, -0.0038370, 0.0043954
7: 0.0034533, 0.0115504, 0.0038972, 0.0118793, -0.0059862, 0.0052257
8: 0.9916464, 0.9973502, 0.9919591, 0.9975819, -0.0042168, 0.0036811
9: -0.0134820, -0.0083045, -0.0136923, -0.0085883, -0.0033414, 0.0038277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0020895
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021600, upper bound: 0.0021600
time: 0.64 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061468, 0.0087048, 0.0059862, 0.0086218, -0.0019412, 0.0021589
1: 0.0022103, 0.0025799, 0.0021871, 0.0025679, -0.0002805, 0.0003119
2: 0.0095472, 0.0109614, 0.0095931, 0.0110502, -0.0011936, 0.0010733
3: -0.0048063, -0.0033436, -0.0047588, -0.0032518, -0.0012345, 0.0011100
4: -0.0004173, 0.0011661, -0.0005167, 0.0011148, -0.0012017, 0.0013364
5: 0.0030098, 0.0045083, 0.0030584, 0.0046023, -0.0012647, 0.0011372
6: -0.0103583, -0.0044129, -0.0101654, -0.0040396, -0.0050178, 0.0045120
7: 0.0034533, 0.0115504, 0.0029449, 0.0112877, -0.0061449, 0.0068339
8: 0.9916464, 0.9973502, 0.9912884, 0.9971651, -0.0043286, 0.0048139
9: -0.0134820, -0.0083045, -0.0133140, -0.0079794, -0.0043698, 0.0039292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022178
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021600, upper bound: 0.0023076
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0059832, 0.0086759, 0.0062889, 0.0087917, -0.0021490, 0.0017680
1: 0.0021867, 0.0025757, 0.0022309, 0.0025924, -0.0003105, 0.0002554
2: 0.0095632, 0.0110519, 0.0094992, 0.0108829, -0.0009775, 0.0011881
3: -0.0047898, -0.0032501, -0.0048560, -0.0034249, -0.0010110, 0.0012288
4: -0.0005186, 0.0011483, -0.0003294, 0.0012199, -0.0013303, 0.0010944
5: 0.0030267, 0.0046041, 0.0029589, 0.0044250, -0.0010357, 0.0012589
6: -0.0102912, -0.0040327, -0.0105604, -0.0047431, -0.0041094, 0.0049949
7: 0.0029354, 0.0114590, 0.0039030, 0.0118256, -0.0068027, 0.0055966
8: 0.9912817, 0.9972858, 0.9919632, 0.9975441, -0.0047919, 0.0039424
9: -0.0134236, -0.0079733, -0.0136579, -0.0085920, -0.0035786, 0.0043498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023206, upper bound: 0.0021602
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0059862, 0.0086218, 0.0061468, 0.0087048, -0.0021589, 0.0019412
1: 0.0021871, 0.0025679, 0.0022103, 0.0025799, -0.0003119, 0.0002805
2: 0.0095931, 0.0110502, 0.0095472, 0.0109614, -0.0010733, 0.0011936
3: -0.0047588, -0.0032518, -0.0048063, -0.0033436, -0.0011100, 0.0012345
4: -0.0005167, 0.0011148, -0.0004173, 0.0011661, -0.0013364, 0.0012017
5: 0.0030584, 0.0046023, 0.0030098, 0.0045083, -0.0011372, 0.0012647
6: -0.0101654, -0.0040396, -0.0103583, -0.0044129, -0.0045120, 0.0050178
7: 0.0029449, 0.0112877, 0.0034533, 0.0115504, -0.0068339, 0.0061449
8: 0.9912884, 0.9971651, 0.9916464, 0.9973502, -0.0048139, 0.0043286
9: -0.0133140, -0.0079794, -0.0134820, -0.0083045, -0.0039292, 0.0043698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022178, upper bound: 0.0021826
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023076, upper bound: 0.0021600
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0059876, 0.0086095, 0.0059832, 0.0086759, -0.0017501, 0.0016748
1: 0.0021873, 0.0025661, 0.0021867, 0.0025757, -0.0002528, 0.0002420
2: 0.0095999, 0.0110495, 0.0095632, 0.0110519, -0.0009259, 0.0009676
3: -0.0047518, -0.0032526, -0.0047898, -0.0032501, -0.0009576, 0.0010007
4: -0.0005158, 0.0011072, -0.0005186, 0.0011483, -0.0010834, 0.0010367
5: 0.0030656, 0.0046015, 0.0030267, 0.0046041, -0.0009811, 0.0010252
6: -0.0101369, -0.0040429, -0.0102912, -0.0040327, -0.0038926, 0.0040678
7: 0.0029493, 0.0112489, 0.0029354, 0.0114590, -0.0055400, 0.0053014
8: 0.9912914, 0.9971378, 0.9912817, 0.9972858, -0.0039025, 0.0037344
9: -0.0132892, -0.0079822, -0.0134236, -0.0079733, -0.0033899, 0.0035424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0021785
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0023080
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058478, 0.0085182, 0.0059862, 0.0086218, -0.0019234, 0.0016863
1: 0.0021671, 0.0025529, 0.0021871, 0.0025679, -0.0002779, 0.0002436
2: 0.0096504, 0.0111268, 0.0095931, 0.0110502, -0.0009323, 0.0010634
3: -0.0046996, -0.0031726, -0.0047588, -0.0032518, -0.0009643, 0.0010998
4: -0.0006024, 0.0010506, -0.0005167, 0.0011148, -0.0011906, 0.0010439
5: 0.0031191, 0.0046834, 0.0030584, 0.0046023, -0.0009879, 0.0011267
6: -0.0099247, -0.0037178, -0.0101654, -0.0040396, -0.0039195, 0.0044704
7: 0.0025066, 0.0109598, 0.0029449, 0.0112877, -0.0060883, 0.0053381
8: 0.9909796, 0.9969342, 0.9912884, 0.9971651, -0.0042887, 0.0037602
9: -0.0131044, -0.0076991, -0.0133140, -0.0079794, -0.0034133, 0.0038930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0021831
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0023029
time: 0.65 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.94 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023682, upper bound: 0.0023702
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023682, upper bound: 0.0023702
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022215
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0023206
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0020895
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0021600, upper bound: 0.0021600
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022178
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0021600, upper bound: 0.0023076
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023206, upper bound: 0.0021602
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0022178, upper bound: 0.0021826
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023076, upper bound: 0.0021600
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0021785
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0023080
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0021831
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.94
Output dim: 8, lower bound: -0.0023920, upper bound: 0.0023029

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0062889, 0.0087917, 0.0062889, 0.0087917, -0.0016356, 0.0016356
1: 0.0022309, 0.0025924, 0.0022309, 0.0025924, -0.0002363, 0.0002363
2: 0.0094992, 0.0108829, 0.0094992, 0.0108829, -0.0009043, 0.0009043
3: -0.0048560, -0.0034249, -0.0048560, -0.0034249, -0.0009353, 0.0009353
4: -0.0003294, 0.0012199, -0.0003294, 0.0012199, -0.0010125, 0.0010125
5: 0.0029589, 0.0044250, 0.0029589, 0.0044250, -0.0009582, 0.0009582
6: -0.0105604, -0.0047431, -0.0105604, -0.0047431, -0.0038017, 0.0038017
7: 0.0039030, 0.0118256, 0.0039030, 0.0118256, -0.0051776, 0.0051776
8: 0.9919632, 0.9975441, 0.9919632, 0.9975441, -0.0036472, 0.0036472
9: -0.0136579, -0.0085920, -0.0136579, -0.0085920, -0.0033107, 0.0033107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020913, upper bound: 0.0022093
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0021970
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0062889, 0.0087917, 0.0061468, 0.0087048, -0.0016189, 0.0018495
1: 0.0022309, 0.0025924, 0.0022103, 0.0025799, -0.0002339, 0.0002672
2: 0.0094992, 0.0108829, 0.0095472, 0.0109614, -0.0010225, 0.0008951
3: -0.0048560, -0.0034249, -0.0048063, -0.0033436, -0.0010575, 0.0009257
4: -0.0003294, 0.0012199, -0.0004173, 0.0011661, -0.0010021, 0.0011448
5: 0.0029589, 0.0044250, 0.0030098, 0.0045083, -0.0010834, 0.0009484
6: -0.0105604, -0.0047431, -0.0103583, -0.0044129, -0.0042986, 0.0037628
7: 0.0039030, 0.0118256, 0.0034533, 0.0115504, -0.0051246, 0.0058544
8: 0.9919632, 0.9975441, 0.9916464, 0.9973502, -0.0036099, 0.0041239
9: -0.0136579, -0.0085920, -0.0134820, -0.0083045, -0.0037434, 0.0032768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020913, upper bound: 0.0022093
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0021970
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0062891, 0.0087917, 0.0060683, 0.0086723, -0.0017647, 0.0020633
1: 0.0022309, 0.0025924, 0.0021990, 0.0025752, -0.0002549, 0.0002981
2: 0.0094992, 0.0108828, 0.0095652, 0.0110048, -0.0011407, 0.0009757
3: -0.0048560, -0.0034250, -0.0047877, -0.0032987, -0.0011798, 0.0010091
4: -0.0003292, 0.0012199, -0.0004659, 0.0011460, -0.0010924, 0.0012772
5: 0.0029589, 0.0044249, 0.0030288, 0.0045542, -0.0012087, 0.0010338
6: -0.0105603, -0.0047437, -0.0102829, -0.0042305, -0.0047956, 0.0041016
7: 0.0039038, 0.0118256, 0.0032049, 0.0114477, -0.0055861, 0.0065312
8: 0.9919638, 0.9975441, 0.9914715, 0.9972779, -0.0039350, 0.0046007
9: -0.0136579, -0.0085925, -0.0134163, -0.0081456, -0.0041763, 0.0035719

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022215
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022215
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0063276, 0.0087881, 0.0060908, 0.0088358, -0.0019005, 0.0020804
1: 0.0022365, 0.0025919, 0.0022022, 0.0025988, -0.0002746, 0.0003006
2: 0.0095011, 0.0108615, 0.0094748, 0.0109924, -0.0011502, 0.0010507
3: -0.0048539, -0.0034470, -0.0048812, -0.0033116, -0.0011896, 0.0010867
4: -0.0003054, 0.0012177, -0.0004520, 0.0012472, -0.0011764, 0.0012878
5: 0.0029610, 0.0044023, 0.0029331, 0.0045411, -0.0012187, 0.0011133
6: -0.0105520, -0.0048332, -0.0106628, -0.0042827, -0.0048354, 0.0044173
7: 0.0040257, 0.0118142, 0.0032759, 0.0119651, -0.0060159, 0.0065854
8: 0.9920496, 0.9975361, 0.9915215, 0.9976424, -0.0042377, 0.0046389
9: -0.0136507, -0.0086705, -0.0137471, -0.0081910, -0.0042109, 0.0038467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0023206
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0023206
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061471, 0.0087048, 0.0063720, 0.0088051, -0.0018876, 0.0015648
1: 0.0022104, 0.0025799, 0.0022429, 0.0025944, -0.0002727, 0.0002261
2: 0.0095472, 0.0109613, 0.0094918, 0.0108369, -0.0008652, 0.0010436
3: -0.0048063, -0.0033438, -0.0048636, -0.0034724, -0.0008948, 0.0010793
4: -0.0004171, 0.0011661, -0.0002779, 0.0012282, -0.0011684, 0.0009687
5: 0.0030098, 0.0045081, 0.0029511, 0.0043763, -0.0009167, 0.0011057
6: -0.0103583, -0.0044135, -0.0105914, -0.0049363, -0.0036371, 0.0043872
7: 0.0034541, 0.0115504, 0.0041661, 0.0118679, -0.0059750, 0.0049534
8: 0.9916469, 0.9973502, 0.9921486, 0.9975739, -0.0042089, 0.0034893
9: -0.0134820, -0.0083050, -0.0136850, -0.0087603, -0.0031673, 0.0038206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0020902
time: 0.59 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0020902
time: 0.70 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061860, 0.0087013, 0.0063852, 0.0089732, -0.0020608, 0.0015766
1: 0.0022160, 0.0025794, 0.0022448, 0.0026187, -0.0002977, 0.0002278
2: 0.0095491, 0.0109398, 0.0093988, 0.0108296, -0.0008717, 0.0011394
3: -0.0048043, -0.0033660, -0.0049598, -0.0034800, -0.0009015, 0.0011784
4: -0.0003931, 0.0011640, -0.0002697, 0.0013323, -0.0012757, 0.0009760
5: 0.0030118, 0.0044853, 0.0028525, 0.0043686, -0.0009236, 0.0012072
6: -0.0103502, -0.0045039, -0.0109823, -0.0049671, -0.0036645, 0.0047898
7: 0.0035773, 0.0115394, 0.0042080, 0.0124002, -0.0065233, 0.0049907
8: 0.9917338, 0.9973425, 0.9921780, 0.9979488, -0.0045952, 0.0035156
9: -0.0134749, -0.0083838, -0.0140254, -0.0087871, -0.0031912, 0.0041712

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0019548
time: 0.64 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0020773
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0061471, 0.0087048, 0.0060714, 0.0086182, -0.0019379, 0.0020730
1: 0.0022104, 0.0025799, 0.0021994, 0.0025674, -0.0002800, 0.0002995
2: 0.0095472, 0.0109613, 0.0095951, 0.0110031, -0.0011461, 0.0010714
3: -0.0048063, -0.0033438, -0.0047568, -0.0033005, -0.0011854, 0.0011081
4: -0.0004171, 0.0011661, -0.0004640, 0.0011125, -0.0011996, 0.0012833
5: 0.0030098, 0.0045081, 0.0030605, 0.0045524, -0.0012144, 0.0011352
6: -0.0103583, -0.0044135, -0.0101571, -0.0042376, -0.0048183, 0.0045043
7: 0.0034541, 0.0115504, 0.0032146, 0.0112764, -0.0061345, 0.0065622
8: 0.9916469, 0.9973502, 0.9914783, 0.9971572, -0.0043213, 0.0046225
9: -0.0134820, -0.0083050, -0.0133068, -0.0081518, -0.0041960, 0.0039226

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0022178
time: 0.60 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0022178
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0061860, 0.0087013, 0.0060942, 0.0087860, -0.0020728, 0.0020865
1: 0.0022160, 0.0025794, 0.0022027, 0.0025916, -0.0002995, 0.0003014
2: 0.0095491, 0.0109398, 0.0095023, 0.0109905, -0.0011536, 0.0011460
3: -0.0048043, -0.0033660, -0.0048528, -0.0033135, -0.0011931, 0.0011852
4: -0.0003931, 0.0011640, -0.0004499, 0.0012164, -0.0012831, 0.0012916
5: 0.0030118, 0.0044853, 0.0029622, 0.0045391, -0.0012223, 0.0012142
6: -0.0103502, -0.0045039, -0.0105472, -0.0042906, -0.0048495, 0.0048177
7: 0.0035773, 0.0115394, 0.0032867, 0.0118077, -0.0065613, 0.0066046
8: 0.9917338, 0.9973425, 0.9915290, 0.9975314, -0.0046219, 0.0046524
9: -0.0134749, -0.0083838, -0.0136465, -0.0081980, -0.0042232, 0.0041954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0020943
time: 0.72 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0022317
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0060683, 0.0086723, 0.0062891, 0.0087917, -0.0020633, 0.0017647
1: 0.0021990, 0.0025752, 0.0022309, 0.0025924, -0.0002981, 0.0002549
2: 0.0095652, 0.0110048, 0.0094992, 0.0108828, -0.0009757, 0.0011407
3: -0.0047877, -0.0032987, -0.0048560, -0.0034250, -0.0010091, 0.0011798
4: -0.0004659, 0.0011460, -0.0003292, 0.0012199, -0.0012772, 0.0010924
5: 0.0030288, 0.0045542, 0.0029589, 0.0044249, -0.0010338, 0.0012087
6: -0.0102829, -0.0042305, -0.0105603, -0.0047437, -0.0041016, 0.0047956
7: 0.0032049, 0.0114477, 0.0039038, 0.0118256, -0.0065312, 0.0055861
8: 0.9914715, 0.9972779, 0.9919638, 0.9975441, -0.0046007, 0.0039350
9: -0.0134163, -0.0081456, -0.0136579, -0.0085925, -0.0035719, 0.0041763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0060908, 0.0088358, 0.0063276, 0.0087881, -0.0020804, 0.0019005
1: 0.0022022, 0.0025988, 0.0022365, 0.0025919, -0.0003006, 0.0002746
2: 0.0094748, 0.0109924, 0.0095011, 0.0108615, -0.0010507, 0.0011502
3: -0.0048812, -0.0033116, -0.0048539, -0.0034470, -0.0010867, 0.0011896
4: -0.0004520, 0.0012472, -0.0003054, 0.0012177, -0.0012878, 0.0011764
5: 0.0029331, 0.0045411, 0.0029610, 0.0044023, -0.0011133, 0.0012187
6: -0.0106628, -0.0042827, -0.0105520, -0.0048332, -0.0044173, 0.0048354
7: 0.0032759, 0.0119651, 0.0040257, 0.0118142, -0.0065854, 0.0060159
8: 0.9915215, 0.9976424, 0.9920496, 0.9975361, -0.0046389, 0.0042377
9: -0.0137471, -0.0081910, -0.0136507, -0.0086705, -0.0038467, 0.0042109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023206, upper bound: 0.0021602
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023206, upper bound: 0.0021602
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0060714, 0.0086182, 0.0061471, 0.0087048, -0.0020730, 0.0019379
1: 0.0021994, 0.0025674, 0.0022104, 0.0025799, -0.0002995, 0.0002800
2: 0.0095951, 0.0110031, 0.0095472, 0.0109613, -0.0010714, 0.0011461
3: -0.0047568, -0.0033005, -0.0048063, -0.0033438, -0.0011081, 0.0011854
4: -0.0004640, 0.0011125, -0.0004171, 0.0011661, -0.0012833, 0.0011996
5: 0.0030605, 0.0045524, 0.0030098, 0.0045081, -0.0011352, 0.0012144
6: -0.0101571, -0.0042376, -0.0103583, -0.0044135, -0.0045043, 0.0048183
7: 0.0032146, 0.0112764, 0.0034541, 0.0115504, -0.0065622, 0.0061345
8: 0.9914783, 0.9971572, 0.9916469, 0.9973502, -0.0046225, 0.0043213
9: -0.0133068, -0.0081518, -0.0134820, -0.0083050, -0.0039226, 0.0041960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022178, upper bound: 0.0020902
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022178, upper bound: 0.0021600
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0060942, 0.0087860, 0.0061860, 0.0087013, -0.0020865, 0.0020728
1: 0.0022027, 0.0025916, 0.0022160, 0.0025794, -0.0003014, 0.0002995
2: 0.0095023, 0.0109905, 0.0095491, 0.0109398, -0.0011460, 0.0011536
3: -0.0048528, -0.0033135, -0.0048043, -0.0033660, -0.0011852, 0.0011931
4: -0.0004499, 0.0012164, -0.0003931, 0.0011640, -0.0012916, 0.0012831
5: 0.0029622, 0.0045391, 0.0030118, 0.0044853, -0.0012142, 0.0012223
6: -0.0105472, -0.0042906, -0.0103502, -0.0045039, -0.0048177, 0.0048495
7: 0.0032867, 0.0118077, 0.0035773, 0.0115394, -0.0066046, 0.0065613
8: 0.9915290, 0.9975314, 0.9917338, 0.9973425, -0.0046524, 0.0046219
9: -0.0136465, -0.0081980, -0.0134749, -0.0083838, -0.0041954, 0.0042232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020943, upper bound: 0.0020783
time: 0.60 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022317, upper bound: 0.0020783
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0059879, 0.0086095, 0.0060683, 0.0086723, -0.0017465, 0.0015871
1: 0.0021874, 0.0025661, 0.0021990, 0.0025752, -0.0002523, 0.0002293
2: 0.0095999, 0.0110493, 0.0095652, 0.0110048, -0.0008775, 0.0009656
3: -0.0047518, -0.0032527, -0.0047877, -0.0032987, -0.0009075, 0.0009987
4: -0.0005157, 0.0011071, -0.0004659, 0.0011460, -0.0010811, 0.0009824
5: 0.0030656, 0.0046014, 0.0030288, 0.0045542, -0.0009297, 0.0010231
6: -0.0101369, -0.0040435, -0.0102829, -0.0042305, -0.0036889, 0.0040594
7: 0.0029501, 0.0112488, 0.0032049, 0.0114477, -0.0055286, 0.0050239
8: 0.9912920, 0.9971377, 0.9914715, 0.9972779, -0.0038945, 0.0035390
9: -0.0132891, -0.0079827, -0.0134163, -0.0081456, -0.0032124, 0.0035351

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0021785
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0021785
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060293, 0.0086061, 0.0060908, 0.0088358, -0.0019256, 0.0015998
1: 0.0021934, 0.0025656, 0.0022022, 0.0025988, -0.0002782, 0.0002311
2: 0.0096018, 0.0110264, 0.0094748, 0.0109924, -0.0008845, 0.0010646
3: -0.0047499, -0.0032764, -0.0048812, -0.0033116, -0.0009148, 0.0011011
4: -0.0004900, 0.0011051, -0.0004520, 0.0012472, -0.0011920, 0.0009903
5: 0.0030676, 0.0045771, 0.0029331, 0.0045411, -0.0009371, 0.0011280
6: -0.0101290, -0.0041398, -0.0106628, -0.0042827, -0.0037183, 0.0044756
7: 0.0030813, 0.0112381, 0.0032759, 0.0119651, -0.0060953, 0.0050640
8: 0.9913844, 0.9971303, 0.9915215, 0.9976424, -0.0042937, 0.0035672
9: -0.0132823, -0.0080666, -0.0137471, -0.0081910, -0.0032381, 0.0038975

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0023080
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0023080
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058480, 0.0085182, 0.0060714, 0.0086182, -0.0019198, 0.0015986
1: 0.0021672, 0.0025529, 0.0021994, 0.0025674, -0.0002774, 0.0002310
2: 0.0096504, 0.0111267, 0.0095951, 0.0110031, -0.0008838, 0.0010614
3: -0.0046996, -0.0031728, -0.0047568, -0.0033005, -0.0009141, 0.0010978
4: -0.0006023, 0.0010506, -0.0004640, 0.0011125, -0.0011884, 0.0009896
5: 0.0031191, 0.0046833, 0.0030605, 0.0045524, -0.0009365, 0.0011246
6: -0.0099246, -0.0037184, -0.0101571, -0.0042376, -0.0037156, 0.0044621
7: 0.0025074, 0.0109598, 0.0032146, 0.0112764, -0.0060770, 0.0050603
8: 0.9909801, 0.9969342, 0.9914783, 0.9971572, -0.0042808, 0.0035646
9: -0.0131043, -0.0076996, -0.0133068, -0.0081518, -0.0032357, 0.0038858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022688, upper bound: 0.0021831
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022688, upper bound: 0.0021831
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058860, 0.0085149, 0.0060942, 0.0087860, -0.0020878, 0.0016138
1: 0.0021727, 0.0025525, 0.0022027, 0.0025916, -0.0003016, 0.0002331
2: 0.0096522, 0.0111057, 0.0095023, 0.0109905, -0.0008922, 0.0011543
3: -0.0046977, -0.0031945, -0.0048528, -0.0033135, -0.0009228, 0.0011938
4: -0.0005788, 0.0010486, -0.0004499, 0.0012164, -0.0012924, 0.0009990
5: 0.0031210, 0.0046610, 0.0029622, 0.0045391, -0.0009454, 0.0012230
6: -0.0099170, -0.0038067, -0.0105472, -0.0042906, -0.0037509, 0.0048527
7: 0.0026276, 0.0109494, 0.0032867, 0.0118077, -0.0066089, 0.0051085
8: 0.9910649, 0.9969268, 0.9915290, 0.9975314, -0.0046555, 0.0035985
9: -0.0130977, -0.0077765, -0.0136465, -0.0081980, -0.0032665, 0.0042259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023486, upper bound: 0.0021269
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023486, upper bound: 0.0022554
time: 0.76 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.27 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020913, upper bound: 0.0022093
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0021970
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020913, upper bound: 0.0022093
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0021970
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022215
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0021826, upper bound: 0.0022215
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0023206
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0021602, upper bound: 0.0023206
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0020902
IS_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0020902
IS_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0019548
IS_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0020773
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0022178
IS_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0022178
IS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0020943
IS_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0022317
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023206, upper bound: 0.0021602
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023206, upper bound: 0.0021602
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0022178, upper bound: 0.0020902
IS_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0022178, upper bound: 0.0021600
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0020943, upper bound: 0.0020783
IS_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0022317, upper bound: 0.0020783
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0021785
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0021785
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0023080
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023815, upper bound: 0.0023080
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0022688, upper bound: 0.0021831
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0022688, upper bound: 0.0021831
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023486, upper bound: 0.0021269
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.27
Output dim: 8, lower bound: -0.0023486, upper bound: 0.0022554

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0062891, 0.0087917, -0.0015497, 0.0016322
1: 0.0022431, 0.0025919, 0.0022309, 0.0025924, -0.0002239, 0.0002358
2: 0.0095011, 0.0108359, 0.0094992, 0.0108828, -0.0009024, 0.0008568
3: -0.0048540, -0.0034734, -0.0048560, -0.0034250, -0.0009333, 0.0008861
4: -0.0002768, 0.0012178, -0.0003292, 0.0012199, -0.0009593, 0.0010104
5: 0.0029609, 0.0043753, 0.0029589, 0.0044249, -0.0009562, 0.0009078
6: -0.0105522, -0.0049405, -0.0105603, -0.0047437, -0.0037938, 0.0036018
7: 0.0041718, 0.0118144, 0.0039038, 0.0118256, -0.0049054, 0.0051668
8: 0.9921526, 0.9975362, 0.9919638, 0.9975441, -0.0034554, 0.0036396
9: -0.0136508, -0.0087639, -0.0136579, -0.0085925, -0.0033038, 0.0031366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021622, upper bound: 0.0021623
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021622, upper bound: 0.0022692
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063872, 0.0089668, 0.0063276, 0.0087881, -0.0015648, 0.0018146
1: 0.0022451, 0.0026177, 0.0022365, 0.0025919, -0.0002261, 0.0002622
2: 0.0094024, 0.0108286, 0.0095011, 0.0108615, -0.0010032, 0.0008651
3: -0.0049561, -0.0034811, -0.0048539, -0.0034470, -0.0010376, 0.0008948
4: -0.0002685, 0.0013283, -0.0003054, 0.0012177, -0.0009686, 0.0011232
5: 0.0028563, 0.0043675, 0.0029610, 0.0044023, -0.0010630, 0.0009166
6: -0.0109672, -0.0049716, -0.0105520, -0.0048332, -0.0042176, 0.0036370
7: 0.0042141, 0.0123797, 0.0040257, 0.0118142, -0.0049533, 0.0057439
8: 0.9921824, 0.9979343, 0.9920496, 0.9975361, -0.0034892, 0.0040461
9: -0.0140123, -0.0087910, -0.0136507, -0.0086705, -0.0036728, 0.0031672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022108, upper bound: 0.0020293
time: 0.58 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022110, upper bound: 0.0022110
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0061471, 0.0087048, -0.0015329, 0.0018461
1: 0.0022431, 0.0025919, 0.0022104, 0.0025799, -0.0002215, 0.0002667
2: 0.0095011, 0.0108359, 0.0095472, 0.0109613, -0.0010206, 0.0008475
3: -0.0048540, -0.0034734, -0.0048063, -0.0033438, -0.0010556, 0.0008765
4: -0.0002768, 0.0012178, -0.0004171, 0.0011661, -0.0009489, 0.0011427
5: 0.0029609, 0.0043753, 0.0030098, 0.0045081, -0.0010814, 0.0008980
6: -0.0105522, -0.0049405, -0.0103583, -0.0044135, -0.0042908, 0.0035629
7: 0.0041718, 0.0118144, 0.0034541, 0.0115504, -0.0048524, 0.0058437
8: 0.9921526, 0.9975362, 0.9916469, 0.9973502, -0.0034181, 0.0041164
9: -0.0136508, -0.0087639, -0.0134820, -0.0083050, -0.0037366, 0.0031028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020915, upper bound: 0.0021122
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020915, upper bound: 0.0021970
time: 0.58 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063872, 0.0089668, 0.0061860, 0.0087013, -0.0015481, 0.0020262
1: 0.0022451, 0.0026177, 0.0022160, 0.0025794, -0.0002237, 0.0002927
2: 0.0094024, 0.0108286, 0.0095491, 0.0109398, -0.0011202, 0.0008559
3: -0.0049561, -0.0034811, -0.0048043, -0.0033660, -0.0011586, 0.0008852
4: -0.0002685, 0.0013283, -0.0003931, 0.0011640, -0.0009583, 0.0012542
5: 0.0028563, 0.0043675, 0.0030118, 0.0044853, -0.0011869, 0.0009069
6: -0.0109672, -0.0049716, -0.0103502, -0.0045039, -0.0047094, 0.0035981
7: 0.0042141, 0.0123797, 0.0035773, 0.0115394, -0.0049003, 0.0064138
8: 0.9921824, 0.9979343, 0.9917338, 0.9973425, -0.0034519, 0.0045180
9: -0.0140123, -0.0087910, -0.0134749, -0.0083838, -0.0041011, 0.0031334

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0021212
time: 0.75 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020780, upper bound: 0.0021212
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0062891, 0.0087917, 0.0060728, 0.0086059, -0.0016853, 0.0020583
1: 0.0022309, 0.0025924, 0.0021996, 0.0025656, -0.0002435, 0.0002974
2: 0.0094992, 0.0108828, 0.0096019, 0.0110024, -0.0011380, 0.0009318
3: -0.0048560, -0.0034250, -0.0047497, -0.0033013, -0.0011770, 0.0009637
4: -0.0003292, 0.0012199, -0.0004631, 0.0011049, -0.0010433, 0.0012741
5: 0.0029589, 0.0044249, 0.0030677, 0.0045516, -0.0012058, 0.0009873
6: -0.0105603, -0.0047437, -0.0101285, -0.0042408, -0.0047841, 0.0039172
7: 0.0039038, 0.0118256, 0.0032189, 0.0112374, -0.0053349, 0.0065155
8: 0.9919638, 0.9975441, 0.9914814, 0.9971297, -0.0037580, 0.0045897
9: -0.0136579, -0.0085925, -0.0132819, -0.0081546, -0.0041662, 0.0034113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0062891, 0.0087917, 0.0059356, 0.0085146, -0.0016519, 0.0022633
1: 0.0022309, 0.0025924, 0.0021798, 0.0025524, -0.0002387, 0.0003270
2: 0.0094992, 0.0108828, 0.0096524, 0.0110782, -0.0012513, 0.0009133
3: -0.0048560, -0.0034250, -0.0046975, -0.0032228, -0.0012942, 0.0009446
4: -0.0003292, 0.0012199, -0.0005481, 0.0010484, -0.0010226, 0.0014010
5: 0.0029589, 0.0044249, 0.0031212, 0.0046320, -0.0013258, 0.0009677
6: -0.0105603, -0.0047437, -0.0099162, -0.0039219, -0.0052605, 0.0038396
7: 0.0039038, 0.0118256, 0.0027846, 0.0109483, -0.0052292, 0.0071644
8: 0.9919638, 0.9975441, 0.9911755, 0.9969261, -0.0036835, 0.0050467
9: -0.0136579, -0.0085925, -0.0130970, -0.0078769, -0.0045811, 0.0033437

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
time: 0.70 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0063276, 0.0087881, 0.0060957, 0.0087656, -0.0018222, 0.0020751
1: 0.0022365, 0.0025919, 0.0022029, 0.0025887, -0.0002633, 0.0002998
2: 0.0095011, 0.0108615, 0.0095136, 0.0109897, -0.0011473, 0.0010074
3: -0.0048539, -0.0034470, -0.0048411, -0.0033144, -0.0011866, 0.0010419
4: -0.0003054, 0.0012177, -0.0004490, 0.0012038, -0.0011280, 0.0012845
5: 0.0029610, 0.0044023, 0.0029742, 0.0045382, -0.0012156, 0.0010674
6: -0.0105520, -0.0048332, -0.0104996, -0.0042940, -0.0048231, 0.0042353
7: 0.0040257, 0.0118142, 0.0032913, 0.0117429, -0.0057681, 0.0065687
8: 0.9920496, 0.9975361, 0.9915324, 0.9974858, -0.0040632, 0.0046271
9: -0.0136507, -0.0086705, -0.0136051, -0.0082009, -0.0042002, 0.0036883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0020835
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0022504
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0063276, 0.0087881, 0.0059479, 0.0086848, -0.0017820, 0.0022786
1: 0.0022365, 0.0025919, 0.0021816, 0.0025770, -0.0002574, 0.0003292
2: 0.0095011, 0.0108615, 0.0095583, 0.0110714, -0.0012598, 0.0009852
3: -0.0048539, -0.0034470, -0.0047949, -0.0032299, -0.0013029, 0.0010190
4: -0.0003054, 0.0012177, -0.0005404, 0.0011538, -0.0011031, 0.0014105
5: 0.0029610, 0.0044023, 0.0030215, 0.0046247, -0.0013348, 0.0010439
6: -0.0105520, -0.0048332, -0.0103119, -0.0039507, -0.0052962, 0.0041419
7: 0.0040257, 0.0118142, 0.0028238, 0.0114872, -0.0056409, 0.0072129
8: 0.9920496, 0.9975361, 0.9912030, 0.9973058, -0.0039736, 0.0050809
9: -0.0136507, -0.0086705, -0.0134416, -0.0079019, -0.0046121, 0.0036069

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0020835
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0022504
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062289, 0.0087011, 0.0063720, 0.0088051, -0.0018049, 0.0015615
1: 0.0022222, 0.0025794, 0.0022429, 0.0025944, -0.0002608, 0.0002256
2: 0.0095492, 0.0109161, 0.0094918, 0.0108369, -0.0008633, 0.0009979
3: -0.0048042, -0.0033905, -0.0048636, -0.0034724, -0.0008929, 0.0010321
4: -0.0003665, 0.0011639, -0.0002779, 0.0012282, -0.0011173, 0.0009666
5: 0.0030120, 0.0044602, 0.0029511, 0.0043763, -0.0009147, 0.0010573
6: -0.0103498, -0.0046036, -0.0105914, -0.0049363, -0.0036293, 0.0041951
7: 0.0037130, 0.0115388, 0.0041661, 0.0118679, -0.0057134, 0.0049428
8: 0.9918294, 0.9973421, 0.9921486, 0.9975739, -0.0040246, 0.0034818
9: -0.0134746, -0.0084705, -0.0136850, -0.0087603, -0.0031606, 0.0036533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0019070
time: 0.77 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020111
time: 0.69 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062414, 0.0088688, 0.0063720, 0.0088051, -0.0018041, 0.0017785
1: 0.0022240, 0.0026036, 0.0022429, 0.0025944, -0.0002606, 0.0002569
2: 0.0094566, 0.0109091, 0.0094918, 0.0108369, -0.0009833, 0.0009974
3: -0.0049001, -0.0033977, -0.0048636, -0.0034724, -0.0010170, 0.0010316
4: -0.0003587, 0.0012676, -0.0002779, 0.0012282, -0.0011167, 0.0011009
5: 0.0029137, 0.0044528, 0.0029511, 0.0043763, -0.0010419, 0.0010568
6: -0.0107395, -0.0046328, -0.0105914, -0.0049363, -0.0041338, 0.0041931
7: 0.0037528, 0.0120695, 0.0041661, 0.0118679, -0.0057107, 0.0056299
8: 0.9918574, 0.9977159, 0.9921486, 0.9975739, -0.0040227, 0.0039658
9: -0.0138139, -0.0084960, -0.0136850, -0.0087603, -0.0035999, 0.0036516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0019070
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020111
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0086170, 0.0063905, 0.0089509, -0.0019995, 0.0014931
1: 0.0022255, 0.0025672, 0.0022455, 0.0026154, -0.0002889, 0.0002157
2: 0.0095958, 0.0109035, 0.0094111, 0.0108267, -0.0008255, 0.0011055
3: -0.0047561, -0.0034035, -0.0049470, -0.0034829, -0.0008537, 0.0011433
4: -0.0003524, 0.0011118, -0.0002665, 0.0013185, -0.0012377, 0.0009242
5: 0.0030613, 0.0044469, 0.0028656, 0.0043655, -0.0008746, 0.0011713
6: -0.0101542, -0.0046565, -0.0109304, -0.0049792, -0.0034703, 0.0046474
7: 0.0037850, 0.0112724, 0.0042246, 0.0123295, -0.0063293, 0.0047262
8: 0.9918801, 0.9971544, 0.9921898, 0.9978991, -0.0044585, 0.0033293
9: -0.0133042, -0.0085166, -0.0139802, -0.0087977, -0.0030221, 0.0040472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0019548
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0019548
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061923, 0.0086705, 0.0063866, 0.0089674, -0.0020491, 0.0014960
1: 0.0022169, 0.0025749, 0.0022450, 0.0026178, -0.0002960, 0.0002161
2: 0.0095661, 0.0109363, 0.0094020, 0.0108289, -0.0008271, 0.0011329
3: -0.0047867, -0.0033696, -0.0049565, -0.0034807, -0.0008554, 0.0011717
4: -0.0003892, 0.0011449, -0.0002689, 0.0013287, -0.0012685, 0.0009260
5: 0.0030299, 0.0044816, 0.0028559, 0.0043678, -0.0008763, 0.0012004
6: -0.0102787, -0.0045185, -0.0109688, -0.0049702, -0.0034770, 0.0047628
7: 0.0035971, 0.0114420, 0.0042123, 0.0123818, -0.0064865, 0.0047354
8: 0.9917478, 0.9972739, 0.9921811, 0.9979358, -0.0045692, 0.0033357
9: -0.0134127, -0.0083964, -0.0140136, -0.0087898, -0.0030279, 0.0041476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0020773
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0020773
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062289, 0.0087011, 0.0060714, 0.0086182, -0.0018553, 0.0020697
1: 0.0022222, 0.0025794, 0.0021994, 0.0025674, -0.0002680, 0.0002990
2: 0.0095492, 0.0109161, 0.0095951, 0.0110031, -0.0011443, 0.0010257
3: -0.0048042, -0.0033905, -0.0047568, -0.0033005, -0.0011835, 0.0010609
4: -0.0003665, 0.0011639, -0.0004640, 0.0011125, -0.0011484, 0.0012812
5: 0.0030120, 0.0044602, 0.0030605, 0.0045524, -0.0012124, 0.0010868
6: -0.0103498, -0.0046036, -0.0101571, -0.0042376, -0.0048106, 0.0043122
7: 0.0037130, 0.0115388, 0.0032146, 0.0112764, -0.0058728, 0.0065516
8: 0.9918294, 0.9973421, 0.9914783, 0.9971572, -0.0041369, 0.0046151
9: -0.0134746, -0.0084705, -0.0133068, -0.0081518, -0.0041893, 0.0037552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020435
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021456
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062414, 0.0088688, 0.0060714, 0.0086182, -0.0018544, 0.0022868
1: 0.0022240, 0.0026036, 0.0021994, 0.0025674, -0.0002679, 0.0003304
2: 0.0094566, 0.0109091, 0.0095951, 0.0110031, -0.0012643, 0.0010253
3: -0.0049001, -0.0033977, -0.0047568, -0.0033005, -0.0013076, 0.0010604
4: -0.0003587, 0.0012676, -0.0004640, 0.0011125, -0.0011479, 0.0014155
5: 0.0029137, 0.0044528, 0.0030605, 0.0045524, -0.0013396, 0.0010863
6: -0.0107395, -0.0046328, -0.0101571, -0.0042376, -0.0053151, 0.0043102
7: 0.0037528, 0.0120695, 0.0032146, 0.0112764, -0.0058702, 0.0072387
8: 0.9918574, 0.9977159, 0.9914783, 0.9971572, -0.0041351, 0.0050991
9: -0.0138139, -0.0084960, -0.0133068, -0.0081518, -0.0046286, 0.0037535

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020435
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021456
time: 0.75 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0086170, 0.0060986, 0.0087634, -0.0020137, 0.0020018
1: 0.0022255, 0.0025672, 0.0022034, 0.0025884, -0.0002909, 0.0002892
2: 0.0095958, 0.0109035, 0.0095148, 0.0109881, -0.0011067, 0.0011133
3: -0.0047561, -0.0034035, -0.0048398, -0.0033161, -0.0011446, 0.0011514
4: -0.0003524, 0.0011118, -0.0004471, 0.0012024, -0.0012465, 0.0012391
5: 0.0030613, 0.0044469, 0.0029755, 0.0045365, -0.0011726, 0.0011796
6: -0.0101542, -0.0046565, -0.0104944, -0.0043009, -0.0046526, 0.0046804
7: 0.0037850, 0.0112724, 0.0033008, 0.0117358, -0.0063742, 0.0063365
8: 0.9918801, 0.9971544, 0.9915390, 0.9974808, -0.0044902, 0.0044636
9: -0.0133042, -0.0085166, -0.0136005, -0.0082070, -0.0040517, 0.0040759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0020943
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0020943
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061923, 0.0086705, 0.0060955, 0.0087805, -0.0020600, 0.0020293
1: 0.0022169, 0.0025749, 0.0022029, 0.0025908, -0.0002976, 0.0002932
2: 0.0095661, 0.0109363, 0.0095054, 0.0109898, -0.0011220, 0.0011389
3: -0.0047867, -0.0033696, -0.0048496, -0.0033143, -0.0011604, 0.0011779
4: -0.0003892, 0.0011449, -0.0004491, 0.0012130, -0.0012752, 0.0012562
5: 0.0030299, 0.0044816, 0.0029654, 0.0045383, -0.0011888, 0.0012068
6: -0.0102787, -0.0045185, -0.0105343, -0.0042935, -0.0047167, 0.0047881
7: 0.0035971, 0.0114420, 0.0032907, 0.0117901, -0.0065209, 0.0064238
8: 0.9917478, 0.9972739, 0.9915319, 0.9975191, -0.0045935, 0.0045250
9: -0.0134127, -0.0083964, -0.0136353, -0.0082005, -0.0041075, 0.0041697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0022317
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0022317
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0060728, 0.0086059, 0.0062891, 0.0087917, -0.0020583, 0.0016853
1: 0.0021996, 0.0025656, 0.0022309, 0.0025924, -0.0002974, 0.0002435
2: 0.0096019, 0.0110024, 0.0094992, 0.0108828, -0.0009318, 0.0011380
3: -0.0047497, -0.0033013, -0.0048560, -0.0034250, -0.0009637, 0.0011770
4: -0.0004631, 0.0011049, -0.0003292, 0.0012199, -0.0012741, 0.0010433
5: 0.0030677, 0.0045516, 0.0029589, 0.0044249, -0.0009873, 0.0012058
6: -0.0101285, -0.0042408, -0.0105603, -0.0047437, -0.0039172, 0.0047841
7: 0.0032189, 0.0112374, 0.0039038, 0.0118256, -0.0065155, 0.0053349
8: 0.9914814, 0.9971297, 0.9919638, 0.9975441, -0.0045897, 0.0037580
9: -0.0132819, -0.0081546, -0.0136579, -0.0085925, -0.0034113, 0.0041662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021690
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0059356, 0.0085146, 0.0062891, 0.0087917, -0.0022633, 0.0016519
1: 0.0021798, 0.0025524, 0.0022309, 0.0025924, -0.0003270, 0.0002387
2: 0.0096524, 0.0110782, 0.0094992, 0.0108828, -0.0009133, 0.0012513
3: -0.0046975, -0.0032228, -0.0048560, -0.0034250, -0.0009446, 0.0012942
4: -0.0005481, 0.0010484, -0.0003292, 0.0012199, -0.0014010, 0.0010226
5: 0.0031212, 0.0046320, 0.0029589, 0.0044249, -0.0009677, 0.0013258
6: -0.0099162, -0.0039219, -0.0105603, -0.0047437, -0.0038396, 0.0052605
7: 0.0027846, 0.0109483, 0.0039038, 0.0118256, -0.0071644, 0.0052292
8: 0.9911755, 0.9969261, 0.9919638, 0.9975441, -0.0050467, 0.0036835
9: -0.0130970, -0.0078769, -0.0136579, -0.0085925, -0.0033437, 0.0045811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021733
time: 0.65 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0060957, 0.0087656, 0.0063276, 0.0087881, -0.0020751, 0.0018222
1: 0.0022029, 0.0025887, 0.0022365, 0.0025919, -0.0002998, 0.0002633
2: 0.0095136, 0.0109897, 0.0095011, 0.0108615, -0.0010074, 0.0011473
3: -0.0048411, -0.0033144, -0.0048539, -0.0034470, -0.0010419, 0.0011866
4: -0.0004490, 0.0012038, -0.0003054, 0.0012177, -0.0012845, 0.0011280
5: 0.0029742, 0.0045382, 0.0029610, 0.0044023, -0.0010674, 0.0012156
6: -0.0104996, -0.0042940, -0.0105520, -0.0048332, -0.0042353, 0.0048231
7: 0.0032913, 0.0117429, 0.0040257, 0.0118142, -0.0065687, 0.0057681
8: 0.9915324, 0.9974858, 0.9920496, 0.9975361, -0.0046271, 0.0040632
9: -0.0136051, -0.0082009, -0.0136507, -0.0086705, -0.0036883, 0.0042002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020790
time: 0.71 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020790
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0059479, 0.0086848, 0.0063276, 0.0087881, -0.0022786, 0.0017820
1: 0.0021816, 0.0025770, 0.0022365, 0.0025919, -0.0003292, 0.0002574
2: 0.0095583, 0.0110714, 0.0095011, 0.0108615, -0.0009852, 0.0012598
3: -0.0047949, -0.0032299, -0.0048539, -0.0034470, -0.0010190, 0.0013029
4: -0.0005404, 0.0011538, -0.0003054, 0.0012177, -0.0014105, 0.0011031
5: 0.0030215, 0.0046247, 0.0029610, 0.0044023, -0.0010439, 0.0013348
6: -0.0103119, -0.0039507, -0.0105520, -0.0048332, -0.0041419, 0.0052962
7: 0.0028238, 0.0114872, 0.0040257, 0.0118142, -0.0072129, 0.0056409
8: 0.9912030, 0.9973058, 0.9920496, 0.9975361, -0.0050809, 0.0039736
9: -0.0134416, -0.0079019, -0.0136507, -0.0086705, -0.0036069, 0.0046121

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020790
time: 0.66 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020790
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060714, 0.0086182, 0.0062289, 0.0087011, -0.0020697, 0.0018553
1: 0.0021994, 0.0025674, 0.0022222, 0.0025794, -0.0002990, 0.0002680
2: 0.0095951, 0.0110031, 0.0095492, 0.0109161, -0.0010257, 0.0011443
3: -0.0047568, -0.0033005, -0.0048042, -0.0033905, -0.0010609, 0.0011835
4: -0.0004640, 0.0011125, -0.0003665, 0.0011639, -0.0012812, 0.0011484
5: 0.0030605, 0.0045524, 0.0030120, 0.0044602, -0.0010868, 0.0012124
6: -0.0101571, -0.0042376, -0.0103498, -0.0046036, -0.0043122, 0.0048106
7: 0.0032146, 0.0112764, 0.0037130, 0.0115388, -0.0065516, 0.0058728
8: 0.9914783, 0.9971572, 0.9918294, 0.9973421, -0.0046151, 0.0041369
9: -0.0133068, -0.0081518, -0.0134746, -0.0084705, -0.0037552, 0.0041893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020435, upper bound: 0.0021097
time: 0.72 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021456, upper bound: 0.0021097
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060714, 0.0086182, 0.0062414, 0.0088688, -0.0022868, 0.0018544
1: 0.0021994, 0.0025674, 0.0022240, 0.0026036, -0.0003304, 0.0002679
2: 0.0095951, 0.0110031, 0.0094566, 0.0109091, -0.0010253, 0.0012643
3: -0.0047568, -0.0033005, -0.0049001, -0.0033977, -0.0010604, 0.0013076
4: -0.0004640, 0.0011125, -0.0003587, 0.0012676, -0.0014155, 0.0011479
5: 0.0030605, 0.0045524, 0.0029137, 0.0044528, -0.0010863, 0.0013396
6: -0.0101571, -0.0042376, -0.0107395, -0.0046328, -0.0043102, 0.0053151
7: 0.0032146, 0.0112764, 0.0037528, 0.0120695, -0.0072387, 0.0058702
8: 0.9914783, 0.9971572, 0.9918574, 0.9977159, -0.0050991, 0.0041351
9: -0.0133068, -0.0081518, -0.0138139, -0.0084960, -0.0037535, 0.0046286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020435, upper bound: 0.0021143
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021456, upper bound: 0.0021143
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0060986, 0.0087634, 0.0062516, 0.0086170, -0.0020018, 0.0020137
1: 0.0022034, 0.0025884, 0.0022255, 0.0025672, -0.0002892, 0.0002909
2: 0.0095148, 0.0109881, 0.0095958, 0.0109035, -0.0011133, 0.0011067
3: -0.0048398, -0.0033161, -0.0047561, -0.0034035, -0.0011514, 0.0011446
4: -0.0004471, 0.0012024, -0.0003524, 0.0011118, -0.0012391, 0.0012465
5: 0.0029755, 0.0045365, 0.0030613, 0.0044469, -0.0011796, 0.0011726
6: -0.0104944, -0.0043009, -0.0101542, -0.0046565, -0.0046804, 0.0046526
7: 0.0033008, 0.0117358, 0.0037850, 0.0112724, -0.0063365, 0.0063742
8: 0.9915390, 0.9974808, 0.9918801, 0.9971544, -0.0044636, 0.0044902
9: -0.0136005, -0.0082070, -0.0133042, -0.0085166, -0.0040759, 0.0040517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020783
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020783
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0060955, 0.0087805, 0.0061923, 0.0086705, -0.0020293, 0.0020600
1: 0.0022029, 0.0025908, 0.0022169, 0.0025749, -0.0002932, 0.0002976
2: 0.0095054, 0.0109898, 0.0095661, 0.0109363, -0.0011389, 0.0011220
3: -0.0048496, -0.0033143, -0.0047867, -0.0033696, -0.0011779, 0.0011604
4: -0.0004491, 0.0012130, -0.0003892, 0.0011449, -0.0012562, 0.0012752
5: 0.0029654, 0.0045383, 0.0030299, 0.0044816, -0.0012068, 0.0011888
6: -0.0105343, -0.0042935, -0.0102787, -0.0045185, -0.0047881, 0.0047167
7: 0.0032907, 0.0117901, 0.0035971, 0.0114420, -0.0064238, 0.0065209
8: 0.9915319, 0.9975191, 0.9917478, 0.9972739, -0.0045250, 0.0045935
9: -0.0136353, -0.0082005, -0.0134127, -0.0083964, -0.0041697, 0.0041075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020783
time: 0.79 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020783
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0059879, 0.0086095, 0.0060728, 0.0086059, -0.0016665, 0.0015823
1: 0.0021874, 0.0025661, 0.0021996, 0.0025656, -0.0002408, 0.0002286
2: 0.0095999, 0.0110493, 0.0096019, 0.0110024, -0.0008748, 0.0009213
3: -0.0047518, -0.0032527, -0.0047497, -0.0033013, -0.0009048, 0.0009529
4: -0.0005157, 0.0011071, -0.0004631, 0.0011049, -0.0010316, 0.0009795
5: 0.0030656, 0.0046014, 0.0030677, 0.0045516, -0.0009269, 0.0009762
6: -0.0101369, -0.0040435, -0.0101285, -0.0042408, -0.0036777, 0.0038733
7: 0.0029501, 0.0112488, 0.0032189, 0.0112374, -0.0052751, 0.0050087
8: 0.9912920, 0.9971377, 0.9914814, 0.9971297, -0.0037159, 0.0035283
9: -0.0132891, -0.0079827, -0.0132819, -0.0081546, -0.0032027, 0.0033731

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0059879, 0.0086095, 0.0059356, 0.0085146, -0.0016507, 0.0017968
1: 0.0021874, 0.0025661, 0.0021798, 0.0025524, -0.0002385, 0.0002596
2: 0.0095999, 0.0110493, 0.0096524, 0.0110782, -0.0009934, 0.0009126
3: -0.0047518, -0.0032527, -0.0046975, -0.0032228, -0.0010274, 0.0009439
4: -0.0005157, 0.0011071, -0.0005481, 0.0010484, -0.0010218, 0.0011123
5: 0.0030656, 0.0046014, 0.0031212, 0.0046320, -0.0010526, 0.0009670
6: -0.0101369, -0.0040435, -0.0099162, -0.0039219, -0.0041763, 0.0038367
7: 0.0029501, 0.0112488, 0.0027846, 0.0109483, -0.0052252, 0.0056878
8: 0.9912920, 0.9971377, 0.9911755, 0.9969261, -0.0036807, 0.0040066
9: -0.0132891, -0.0079827, -0.0130970, -0.0078769, -0.0036369, 0.0033411

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060293, 0.0086061, 0.0060957, 0.0087656, -0.0018513, 0.0015947
1: 0.0021934, 0.0025656, 0.0022029, 0.0025887, -0.0002675, 0.0002304
2: 0.0096018, 0.0110264, 0.0095136, 0.0109897, -0.0008817, 0.0010235
3: -0.0047499, -0.0032764, -0.0048411, -0.0033144, -0.0009119, 0.0010586
4: -0.0004900, 0.0011051, -0.0004490, 0.0012038, -0.0011460, 0.0009871
5: 0.0030676, 0.0045771, 0.0029742, 0.0045382, -0.0009342, 0.0010845
6: -0.0101290, -0.0041398, -0.0104996, -0.0042940, -0.0037065, 0.0043029
7: 0.0030813, 0.0112381, 0.0032913, 0.0117429, -0.0058602, 0.0050480
8: 0.9913844, 0.9971303, 0.9915324, 0.9974858, -0.0041281, 0.0035559
9: -0.0132823, -0.0080666, -0.0136051, -0.0082009, -0.0032278, 0.0037472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021820, upper bound: 0.0022639
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022639
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060293, 0.0086061, 0.0059479, 0.0086848, -0.0018088, 0.0018096
1: 0.0021934, 0.0025656, 0.0021816, 0.0025770, -0.0002613, 0.0002614
2: 0.0096018, 0.0110264, 0.0095583, 0.0110714, -0.0010005, 0.0010000
3: -0.0047499, -0.0032764, -0.0047949, -0.0032299, -0.0010347, 0.0010343
4: -0.0004900, 0.0011051, -0.0005404, 0.0011538, -0.0011197, 0.0011201
5: 0.0030676, 0.0045771, 0.0030215, 0.0046247, -0.0010600, 0.0010596
6: -0.0101290, -0.0041398, -0.0103119, -0.0039507, -0.0042059, 0.0042041
7: 0.0030813, 0.0112381, 0.0028238, 0.0114872, -0.0057257, 0.0057281
8: 0.9913844, 0.9971303, 0.9912030, 0.9973058, -0.0040333, 0.0040350
9: -0.0132823, -0.0080666, -0.0134416, -0.0079019, -0.0036627, 0.0036611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021820, upper bound: 0.0022639
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022639
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0059356, 0.0085146, 0.0060714, 0.0086182, -0.0018366, 0.0015952
1: 0.0021798, 0.0025524, 0.0021994, 0.0025674, -0.0002653, 0.0002305
2: 0.0096524, 0.0110782, 0.0095951, 0.0110031, -0.0008819, 0.0010154
3: -0.0046975, -0.0032228, -0.0047568, -0.0033005, -0.0009122, 0.0010502
4: -0.0005481, 0.0010484, -0.0004640, 0.0011125, -0.0011369, 0.0009875
5: 0.0031212, 0.0046320, 0.0030605, 0.0045524, -0.0009345, 0.0010759
6: -0.0099162, -0.0039219, -0.0101571, -0.0042376, -0.0037077, 0.0042688
7: 0.0027846, 0.0109483, 0.0032146, 0.0112764, -0.0058137, 0.0050496
8: 0.9911755, 0.9969261, 0.9914783, 0.9971572, -0.0040953, 0.0035570
9: -0.0130970, -0.0078769, -0.0133068, -0.0081518, -0.0032288, 0.0037174

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0020511
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021362
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0059479, 0.0086848, 0.0060714, 0.0086182, -0.0018334, 0.0018110
1: 0.0021816, 0.0025770, 0.0021994, 0.0025674, -0.0002649, 0.0002616
2: 0.0095583, 0.0110714, 0.0095951, 0.0110031, -0.0010012, 0.0010136
3: -0.0047949, -0.0032299, -0.0047568, -0.0033005, -0.0010355, 0.0010484
4: -0.0005404, 0.0011538, -0.0004640, 0.0011125, -0.0011349, 0.0011210
5: 0.0030215, 0.0046247, 0.0030605, 0.0045524, -0.0010609, 0.0010740
6: -0.0103119, -0.0039507, -0.0101571, -0.0042376, -0.0042092, 0.0042614
7: 0.0028238, 0.0114872, 0.0032146, 0.0112764, -0.0058036, 0.0057326
8: 0.9912030, 0.9973058, 0.9914783, 0.9971572, -0.0040882, 0.0040382
9: -0.0134416, -0.0079019, -0.0133068, -0.0081518, -0.0036656, 0.0037110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0020511
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021362
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0059487, 0.0084320, 0.0060986, 0.0087634, -0.0020323, 0.0015349
1: 0.0021817, 0.0025405, 0.0022034, 0.0025884, -0.0002936, 0.0002217
2: 0.0096981, 0.0110710, 0.0095148, 0.0109881, -0.0008486, 0.0011236
3: -0.0046503, -0.0032304, -0.0048398, -0.0033161, -0.0008777, 0.0011621
4: -0.0005399, 0.0009972, -0.0004471, 0.0012024, -0.0012580, 0.0009501
5: 0.0031696, 0.0046243, 0.0029755, 0.0045365, -0.0008991, 0.0011905
6: -0.0097242, -0.0039525, -0.0104944, -0.0043009, -0.0035675, 0.0047237
7: 0.0028263, 0.0106868, 0.0033008, 0.0117358, -0.0064333, 0.0048587
8: 0.9912048, 0.9967418, 0.9915390, 0.9974808, -0.0045317, 0.0034225
9: -0.0129298, -0.0079036, -0.0136005, -0.0082070, -0.0031068, 0.0041136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021269
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021269
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0058921, 0.0084844, 0.0060955, 0.0087805, -0.0020761, 0.0015369
1: 0.0021735, 0.0025480, 0.0022029, 0.0025908, -0.0002999, 0.0002220
2: 0.0096691, 0.0111023, 0.0095054, 0.0109898, -0.0008497, 0.0011478
3: -0.0046802, -0.0031980, -0.0048496, -0.0033143, -0.0008788, 0.0011871
4: -0.0005750, 0.0010297, -0.0004491, 0.0012130, -0.0012851, 0.0009514
5: 0.0031389, 0.0046575, 0.0029654, 0.0045383, -0.0009003, 0.0012162
6: -0.0098460, -0.0038209, -0.0105343, -0.0042935, -0.0035722, 0.0048255
7: 0.0026471, 0.0108527, 0.0032907, 0.0117901, -0.0065718, 0.0048650
8: 0.9910785, 0.9968587, 0.9915319, 0.9975191, -0.0046293, 0.0034270
9: -0.0130358, -0.0077889, -0.0136353, -0.0082005, -0.0031108, 0.0042022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022554
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022554
time: 0.70 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.22 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021622, upper bound: 0.0021623
IS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021622, upper bound: 0.0022692
IS_A1_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022108, upper bound: 0.0020293
IS_A1_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022110, upper bound: 0.0022110
IS_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020915, upper bound: 0.0021122
IS_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020915, upper bound: 0.0021970
IS_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0021212
IS_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020780, upper bound: 0.0021212
IS_A1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
IS_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
IS_A1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
IS_A1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021691, upper bound: 0.0022215
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0020835
IS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0022504
IS_A1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0020835
IS_A1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020790, upper bound: 0.0022504
IS_A1_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0019070
IS_A1_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020111
IS_A1_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0019070
IS_A1_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020111
IS_A1_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0019548
IS_A1_A2_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0019548
IS_A1_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0020773
IS_A1_A2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020773, upper bound: 0.0020773
IS_A1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020435
IS_A1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021456
IS_A1_A2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020435
IS_A1_A2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021456
IS_A1_A2_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0020943
IS_A1_A2_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0020943
IS_A1_A2_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0022317
IS_A1_A2_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020783, upper bound: 0.0022317
IS_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021690
IS_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021733
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022215, upper bound: 0.0021826
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020790
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020790
IS_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020790
IS_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020790
IS_A2_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020435, upper bound: 0.0021097
IS_A2_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021456, upper bound: 0.0021097
IS_A2_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020435, upper bound: 0.0021143
IS_A2_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021456, upper bound: 0.0021143
IS_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020783
IS_A2_B1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020783
IS_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020783
IS_A2_B1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020783
IS_A2_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
IS_A2_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
IS_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
IS_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023135, upper bound: 0.0021785
IS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021820, upper bound: 0.0022639
IS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022639
IS_A2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0021820, upper bound: 0.0022639
IS_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022639
IS_A2_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0020511
IS_A2_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021362
IS_A2_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0020511
IS_A2_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021362
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021269
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021269
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022554
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.22
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0022554

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0063738, 0.0087882, -0.0015465, 0.0015465
1: 0.0022431, 0.0025919, 0.0022431, 0.0025919, -0.0002234, 0.0002234
2: 0.0095011, 0.0108359, 0.0095011, 0.0108359, -0.0008550, 0.0008550
3: -0.0048540, -0.0034734, -0.0048540, -0.0034734, -0.0008843, 0.0008843
4: -0.0002768, 0.0012178, -0.0002768, 0.0012178, -0.0009573, 0.0009573
5: 0.0029609, 0.0043753, 0.0029609, 0.0043753, -0.0009060, 0.0009060
6: -0.0105522, -0.0049405, -0.0105522, -0.0049405, -0.0035946, 0.0035946
7: 0.0041718, 0.0118144, 0.0041718, 0.0118144, -0.0048955, 0.0048955
8: 0.9921526, 0.9975362, 0.9921526, 0.9975362, -0.0034485, 0.0034485
9: -0.0136508, -0.0087639, -0.0136508, -0.0087639, -0.0031303, 0.0031303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0019991
time: 0.59 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0021703
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0063872, 0.0089668, -0.0017623, 0.0015582
1: 0.0022431, 0.0025919, 0.0022451, 0.0026177, -0.0002546, 0.0002251
2: 0.0095011, 0.0108359, 0.0094024, 0.0108286, -0.0008615, 0.0009743
3: -0.0048540, -0.0034734, -0.0049561, -0.0034811, -0.0008910, 0.0010077
4: -0.0002768, 0.0012178, -0.0002685, 0.0013283, -0.0010909, 0.0009646
5: 0.0029609, 0.0043753, 0.0028563, 0.0043675, -0.0009128, 0.0010324
6: -0.0105522, -0.0049405, -0.0109672, -0.0049716, -0.0036218, 0.0040961
7: 0.0041718, 0.0118144, 0.0042141, 0.0123797, -0.0055786, 0.0049325
8: 0.9921526, 0.9975362, 0.9921824, 0.9979343, -0.0039297, 0.0034746
9: -0.0136508, -0.0087639, -0.0140123, -0.0087910, -0.0031540, 0.0035671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0022108
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0022110
time: 0.59 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0064493, 0.0088852, 0.0063323, 0.0087654, -0.0014792, 0.0017350
1: 0.0022540, 0.0026060, 0.0022371, 0.0025886, -0.0002137, 0.0002507
2: 0.0094475, 0.0107942, 0.0095137, 0.0108589, -0.0009592, 0.0008178
3: -0.0049094, -0.0035166, -0.0048409, -0.0034497, -0.0009921, 0.0008458
4: -0.0002301, 0.0012778, -0.0003025, 0.0012036, -0.0009157, 0.0010740
5: 0.0029041, 0.0043311, 0.0029743, 0.0043996, -0.0010164, 0.0008665
6: -0.0107776, -0.0051160, -0.0104991, -0.0048441, -0.0040326, 0.0034382
7: 0.0044108, 0.0121215, 0.0040405, 0.0117422, -0.0046825, 0.0054920
8: 0.9923210, 0.9977525, 0.9920601, 0.9974853, -0.0032984, 0.0038687
9: -0.0138472, -0.0089167, -0.0136046, -0.0086799, -0.0035118, 0.0029941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022108, upper bound: 0.0019637
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022108, upper bound: 0.0019637
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0063940, 0.0089355, 0.0063288, 0.0087824, -0.0015543, 0.0017374
1: 0.0022460, 0.0026132, 0.0022366, 0.0025911, -0.0002245, 0.0002510
2: 0.0094197, 0.0108248, 0.0095043, 0.0108608, -0.0009606, 0.0008593
3: -0.0049382, -0.0034849, -0.0048507, -0.0034477, -0.0009935, 0.0008887
4: -0.0002643, 0.0013089, -0.0003046, 0.0012142, -0.0009621, 0.0010755
5: 0.0028746, 0.0043635, 0.0029643, 0.0044016, -0.0010178, 0.0009105
6: -0.0108946, -0.0049873, -0.0105388, -0.0048360, -0.0040383, 0.0036125
7: 0.0042356, 0.0122808, 0.0040295, 0.0117962, -0.0049199, 0.0054998
8: 0.9921975, 0.9978647, 0.9920523, 0.9975234, -0.0034657, 0.0038742
9: -0.0139490, -0.0088047, -0.0136391, -0.0086729, -0.0035167, 0.0031459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022110, upper bound: 0.0021100
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022110, upper bound: 0.0021100
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0062289, 0.0087011, -0.0015295, 0.0017634
1: 0.0022431, 0.0025919, 0.0022222, 0.0025794, -0.0002210, 0.0002548
2: 0.0095011, 0.0108359, 0.0095492, 0.0109161, -0.0009749, 0.0008456
3: -0.0048540, -0.0034734, -0.0048042, -0.0033905, -0.0010083, 0.0008746
4: -0.0002768, 0.0012178, -0.0003665, 0.0011639, -0.0009468, 0.0010916
5: 0.0029609, 0.0043753, 0.0030120, 0.0044602, -0.0010330, 0.0008959
6: -0.0105522, -0.0049405, -0.0103498, -0.0046036, -0.0040986, 0.0035549
7: 0.0041718, 0.0118144, 0.0037130, 0.0115388, -0.0048414, 0.0055820
8: 0.9921526, 0.9975362, 0.9918294, 0.9973421, -0.0034104, 0.0039321
9: -0.0136508, -0.0087639, -0.0134746, -0.0084705, -0.0035693, 0.0030957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021310
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021310
time: 0.65 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0062414, 0.0088688, -0.0017248, 0.0017626
1: 0.0022431, 0.0025919, 0.0022240, 0.0026036, -0.0002492, 0.0002546
2: 0.0095011, 0.0108359, 0.0094566, 0.0109091, -0.0009745, 0.0009536
3: -0.0048540, -0.0034734, -0.0049001, -0.0033977, -0.0010078, 0.0009863
4: -0.0002768, 0.0012178, -0.0003587, 0.0012676, -0.0010677, 0.0010911
5: 0.0029609, 0.0043753, 0.0029137, 0.0044528, -0.0010325, 0.0010104
6: -0.0105522, -0.0049405, -0.0107395, -0.0046328, -0.0040967, 0.0040090
7: 0.0041718, 0.0118144, 0.0037528, 0.0120695, -0.0054599, 0.0055793
8: 0.9921526, 0.9975362, 0.9918574, 0.9977159, -0.0038461, 0.0039302
9: -0.0136508, -0.0087639, -0.0138139, -0.0084960, -0.0035676, 0.0034912

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021434
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021434
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063924, 0.0089446, 0.0062516, 0.0086170, -0.0014698, 0.0019645
1: 0.0022458, 0.0026145, 0.0022255, 0.0025672, -0.0002123, 0.0002838
2: 0.0094146, 0.0108257, 0.0095958, 0.0109035, -0.0010861, 0.0008126
3: -0.0049434, -0.0034840, -0.0047561, -0.0034035, -0.0011233, 0.0008404
4: -0.0002653, 0.0013146, -0.0003524, 0.0011118, -0.0009098, 0.0012161
5: 0.0028693, 0.0043644, 0.0030613, 0.0044469, -0.0011508, 0.0008610
6: -0.0109156, -0.0049837, -0.0101542, -0.0046565, -0.0045660, 0.0034161
7: 0.0042307, 0.0123094, 0.0037850, 0.0112724, -0.0046525, 0.0062185
8: 0.9921940, 0.9978850, 0.9918801, 0.9971544, -0.0032773, 0.0043805
9: -0.0139673, -0.0088016, -0.0133042, -0.0085166, -0.0039763, 0.0029749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063885, 0.0089608, 0.0061923, 0.0086705, -0.0014704, 0.0020149
1: 0.0022453, 0.0026169, 0.0022169, 0.0025749, -0.0002124, 0.0002911
2: 0.0094057, 0.0108278, 0.0095661, 0.0109363, -0.0011140, 0.0008129
3: -0.0049527, -0.0034818, -0.0047867, -0.0033696, -0.0011521, 0.0008408
4: -0.0002677, 0.0013246, -0.0003892, 0.0011449, -0.0009102, 0.0012472
5: 0.0028598, 0.0043667, 0.0030299, 0.0044816, -0.0011803, 0.0008613
6: -0.0109535, -0.0049747, -0.0102787, -0.0045185, -0.0046831, 0.0034175
7: 0.0042184, 0.0123609, 0.0035971, 0.0114420, -0.0046544, 0.0063780
8: 0.9921855, 0.9979211, 0.9917478, 0.9972739, -0.0032787, 0.0044928
9: -0.0140003, -0.0087937, -0.0134127, -0.0083964, -0.0040782, 0.0029761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020780, upper bound: 0.0020392
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020780, upper bound: 0.0020392
time: 0.59 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0060728, 0.0086059, -0.0015996, 0.0020552
1: 0.0022431, 0.0025919, 0.0021996, 0.0025656, -0.0002311, 0.0002969
2: 0.0095011, 0.0108359, 0.0096019, 0.0110024, -0.0011363, 0.0008844
3: -0.0048540, -0.0034734, -0.0047497, -0.0033013, -0.0011752, 0.0009147
4: -0.0002768, 0.0012178, -0.0004631, 0.0011049, -0.0009902, 0.0012722
5: 0.0029609, 0.0043753, 0.0030677, 0.0045516, -0.0012039, 0.0009371
6: -0.0105522, -0.0049405, -0.0101285, -0.0042408, -0.0047768, 0.0037180
7: 0.0041718, 0.0118144, 0.0032189, 0.0112374, -0.0050635, 0.0065056
8: 0.9921526, 0.9975362, 0.9914814, 0.9971297, -0.0035669, 0.0045827
9: -0.0136508, -0.0087639, -0.0132819, -0.0081546, -0.0041599, 0.0032378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0020703
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0021982
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063872, 0.0089668, 0.0060728, 0.0086059, -0.0016113, 0.0022710
1: 0.0022451, 0.0026177, 0.0021996, 0.0025656, -0.0002328, 0.0003281
2: 0.0094024, 0.0108286, 0.0096019, 0.0110024, -0.0012556, 0.0008909
3: -0.0049561, -0.0034811, -0.0047497, -0.0033013, -0.0012986, 0.0009214
4: -0.0002685, 0.0013283, -0.0004631, 0.0011049, -0.0009974, 0.0014058
5: 0.0028563, 0.0043675, 0.0030677, 0.0045516, -0.0013303, 0.0009439
6: -0.0109672, -0.0049716, -0.0101285, -0.0042408, -0.0052784, 0.0037452
7: 0.0042141, 0.0123797, 0.0032189, 0.0112374, -0.0051006, 0.0071887
8: 0.9921824, 0.9979343, 0.9914814, 0.9971297, -0.0035930, 0.0050639
9: -0.0140123, -0.0087910, -0.0132819, -0.0081546, -0.0045967, 0.0032615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0020703
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0021982
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063738, 0.0087882, 0.0059356, 0.0085146, -0.0015662, 0.0022602
1: 0.0022431, 0.0025919, 0.0021798, 0.0025524, -0.0002263, 0.0003265
2: 0.0095011, 0.0108359, 0.0096524, 0.0110782, -0.0012496, 0.0008659
3: -0.0048540, -0.0034734, -0.0046975, -0.0032228, -0.0012924, 0.0008956
4: -0.0002768, 0.0012178, -0.0005481, 0.0010484, -0.0009695, 0.0013991
5: 0.0029609, 0.0043753, 0.0031212, 0.0046320, -0.0013240, 0.0009175
6: -0.0105522, -0.0049405, -0.0099162, -0.0039219, -0.0052532, 0.0036403
7: 0.0041718, 0.0118144, 0.0027846, 0.0109483, -0.0049578, 0.0071545
8: 0.9921526, 0.9975362, 0.9911755, 0.9969261, -0.0034924, 0.0050398
9: -0.0136508, -0.0087639, -0.0130970, -0.0078769, -0.0045748, 0.0031702

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020229
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021559
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063872, 0.0089668, 0.0059356, 0.0085146, -0.0015779, 0.0024760
1: 0.0022451, 0.0026177, 0.0021798, 0.0025524, -0.0002280, 0.0003577
2: 0.0094024, 0.0108286, 0.0096524, 0.0110782, -0.0013689, 0.0008724
3: -0.0049561, -0.0034811, -0.0046975, -0.0032228, -0.0014158, 0.0009023
4: -0.0002685, 0.0013283, -0.0005481, 0.0010484, -0.0009768, 0.0015327
5: 0.0028563, 0.0043675, 0.0031212, 0.0046320, -0.0014504, 0.0009244
6: -0.0109672, -0.0049716, -0.0099162, -0.0039219, -0.0057548, 0.0036676
7: 0.0042141, 0.0123797, 0.0027846, 0.0109483, -0.0049949, 0.0078376
8: 0.9921824, 0.9979343, 0.9911755, 0.9969261, -0.0035185, 0.0055209
9: -0.0140123, -0.0087910, -0.0130970, -0.0078769, -0.0050116, 0.0031939

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020229
time: 0.77 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021558
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063909, 0.0087029, 0.0061001, 0.0087421, -0.0017412, 0.0019896
1: 0.0022456, 0.0025796, 0.0022036, 0.0025853, -0.0002516, 0.0002874
2: 0.0095482, 0.0108265, 0.0095266, 0.0109873, -0.0011000, 0.0009627
3: -0.0048052, -0.0034832, -0.0048277, -0.0033169, -0.0011377, 0.0009956
4: -0.0002662, 0.0011650, -0.0004462, 0.0011893, -0.0010778, 0.0012316
5: 0.0030109, 0.0043652, 0.0029879, 0.0045356, -0.0011655, 0.0010200
6: -0.0103540, -0.0049803, -0.0104452, -0.0043043, -0.0046244, 0.0040470
7: 0.0042261, 0.0115445, 0.0033054, 0.0116687, -0.0055117, 0.0062981
8: 0.9921908, 0.9973460, 0.9915423, 0.9974335, -0.0038826, 0.0044365
9: -0.0134782, -0.0087986, -0.0135576, -0.0082099, -0.0040272, 0.0035243

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0021347
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0020703
time: 0.68 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063336, 0.0087596, 0.0060969, 0.0087605, -0.0018107, 0.0020182
1: 0.0022373, 0.0025878, 0.0022031, 0.0025879, -0.0002616, 0.0002916
2: 0.0095169, 0.0108582, 0.0095164, 0.0109890, -0.0011158, 0.0010011
3: -0.0048376, -0.0034504, -0.0048382, -0.0033151, -0.0011540, 0.0010354
4: -0.0003017, 0.0012001, -0.0004482, 0.0012006, -0.0011209, 0.0012493
5: 0.0029777, 0.0043989, 0.0029771, 0.0045375, -0.0011822, 0.0010607
6: -0.0104858, -0.0048470, -0.0104879, -0.0042970, -0.0046908, 0.0042087
7: 0.0040444, 0.0117240, 0.0032954, 0.0117269, -0.0057319, 0.0063884
8: 0.9920629, 0.9974726, 0.9915352, 0.9974745, -0.0040376, 0.0045002
9: -0.0135930, -0.0086825, -0.0135948, -0.0082035, -0.0040849, 0.0036651

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0023245
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0021982
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063909, 0.0087029, 0.0059528, 0.0086626, -0.0017016, 0.0021926
1: 0.0022456, 0.0025796, 0.0021823, 0.0025738, -0.0002458, 0.0003168
2: 0.0095482, 0.0108265, 0.0095705, 0.0110687, -0.0012122, 0.0009408
3: -0.0048052, -0.0034832, -0.0047822, -0.0032327, -0.0012537, 0.0009730
4: -0.0002662, 0.0011650, -0.0005374, 0.0011400, -0.0010533, 0.0013572
5: 0.0030109, 0.0043652, 0.0030345, 0.0046219, -0.0012844, 0.0009968
6: -0.0103540, -0.0049803, -0.0102602, -0.0039619, -0.0050962, 0.0039550
7: 0.0042261, 0.0115445, 0.0028391, 0.0114168, -0.0053864, 0.0069405
8: 0.9921908, 0.9973460, 0.9912138, 0.9972562, -0.0037943, 0.0048891
9: -0.0134782, -0.0087986, -0.0133966, -0.0079117, -0.0044380, 0.0034442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0020835
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0020229
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063336, 0.0087596, 0.0059493, 0.0086791, -0.0017702, 0.0022270
1: 0.0022373, 0.0025878, 0.0021818, 0.0025762, -0.0002557, 0.0003217
2: 0.0095169, 0.0108582, 0.0095614, 0.0110706, -0.0012313, 0.0009787
3: -0.0048376, -0.0034504, -0.0047916, -0.0032307, -0.0012734, 0.0010122
4: -0.0003017, 0.0012001, -0.0005396, 0.0011502, -0.0010958, 0.0013786
5: 0.0029777, 0.0043989, 0.0030249, 0.0046240, -0.0013046, 0.0010370
6: -0.0104858, -0.0048470, -0.0102985, -0.0039538, -0.0051762, 0.0041143
7: 0.0040444, 0.0117240, 0.0028280, 0.0114690, -0.0056034, 0.0070496
8: 0.9920629, 0.9974726, 0.9912060, 0.9972929, -0.0039471, 0.0049659
9: -0.0135930, -0.0086825, -0.0134299, -0.0079047, -0.0045077, 0.0035829

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0022504
time: 0.65 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021559
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0063767, 0.0087814, -0.0017427, 0.0014780
1: 0.0022315, 0.0025672, 0.0022435, 0.0025910, -0.0002518, 0.0002135
2: 0.0095960, 0.0108805, 0.0095048, 0.0108344, -0.0008171, 0.0009635
3: -0.0047559, -0.0034273, -0.0048501, -0.0034750, -0.0008451, 0.0009965
4: -0.0003267, 0.0011115, -0.0002750, 0.0012136, -0.0010788, 0.0009149
5: 0.0030615, 0.0044225, 0.0029649, 0.0043736, -0.0008658, 0.0010209
6: -0.0101534, -0.0047530, -0.0105364, -0.0049471, -0.0034352, 0.0040506
7: 0.0039165, 0.0112713, 0.0041809, 0.0117930, -0.0055166, 0.0046784
8: 0.9919728, 0.9971536, 0.9921589, 0.9975211, -0.0038860, 0.0032956
9: -0.0133035, -0.0086007, -0.0136371, -0.0087697, -0.0029915, 0.0035274

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0020339
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0020339
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0063732, 0.0087996, -0.0017936, 0.0014834
1: 0.0022231, 0.0025749, 0.0022430, 0.0025936, -0.0002591, 0.0002143
2: 0.0095662, 0.0109127, 0.0094948, 0.0108363, -0.0008202, 0.0009916
3: -0.0047866, -0.0033940, -0.0048605, -0.0034730, -0.0008482, 0.0010256
4: -0.0003628, 0.0011448, -0.0002772, 0.0012248, -0.0011103, 0.0009183
5: 0.0030299, 0.0044566, 0.0029542, 0.0043757, -0.0008690, 0.0010507
6: -0.0102784, -0.0046177, -0.0105788, -0.0049390, -0.0034479, 0.0041688
7: 0.0037322, 0.0114416, 0.0041697, 0.0118507, -0.0056775, 0.0046958
8: 0.9918429, 0.9972736, 0.9921511, 0.9975618, -0.0039993, 0.0033078
9: -0.0134124, -0.0084828, -0.0136740, -0.0087626, -0.0030026, 0.0036303

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021577
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021577
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0063083, 0.0087857, 0.0063767, 0.0087814, -0.0017356, 0.0016997
1: 0.0022337, 0.0025916, 0.0022435, 0.0025910, -0.0002507, 0.0002456
2: 0.0095025, 0.0108722, 0.0095048, 0.0108344, -0.0009397, 0.0009596
3: -0.0048525, -0.0034360, -0.0048501, -0.0034750, -0.0009719, 0.0009925
4: -0.0003173, 0.0012162, -0.0002750, 0.0012136, -0.0010744, 0.0010522
5: 0.0029624, 0.0044136, 0.0029649, 0.0043736, -0.0009957, 0.0010167
6: -0.0105463, -0.0047883, -0.0105364, -0.0049471, -0.0039506, 0.0040341
7: 0.0039645, 0.0118064, 0.0041809, 0.0117930, -0.0054941, 0.0053804
8: 0.9920066, 0.9975305, 0.9921589, 0.9975211, -0.0038702, 0.0037901
9: -0.0136457, -0.0086314, -0.0136371, -0.0087697, -0.0034404, 0.0035131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0019070
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0019070
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0063732, 0.0087996, -0.0017922, 0.0017023
1: 0.0022250, 0.0025993, 0.0022430, 0.0025936, -0.0002589, 0.0002459
2: 0.0094731, 0.0109054, 0.0094948, 0.0108363, -0.0009412, 0.0009908
3: -0.0048830, -0.0034016, -0.0048605, -0.0034730, -0.0009734, 0.0010248
4: -0.0003545, 0.0012491, -0.0002772, 0.0012248, -0.0011094, 0.0010537
5: 0.0029312, 0.0044488, 0.0029542, 0.0043757, -0.0009972, 0.0010499
6: -0.0106700, -0.0046486, -0.0105788, -0.0049390, -0.0039566, 0.0041655
7: 0.0037743, 0.0119750, 0.0041697, 0.0118507, -0.0056731, 0.0053885
8: 0.9918726, 0.9976493, 0.9921511, 0.9975618, -0.0039962, 0.0037958
9: -0.0137535, -0.0085097, -0.0136740, -0.0087626, -0.0034456, 0.0036275

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020111
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020111
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0086170, 0.0063924, 0.0089446, -0.0019645, 0.0014698
1: 0.0022255, 0.0025672, 0.0022458, 0.0026145, -0.0002838, 0.0002123
2: 0.0095958, 0.0109035, 0.0094146, 0.0108257, -0.0008126, 0.0010861
3: -0.0047561, -0.0034035, -0.0049434, -0.0034840, -0.0008404, 0.0011233
4: -0.0003524, 0.0011118, -0.0002653, 0.0013146, -0.0012161, 0.0009098
5: 0.0030613, 0.0044469, 0.0028693, 0.0043644, -0.0008610, 0.0011508
6: -0.0101542, -0.0046565, -0.0109156, -0.0049837, -0.0034161, 0.0045660
7: 0.0037850, 0.0112724, 0.0042307, 0.0123094, -0.0062185, 0.0046525
8: 0.9918801, 0.9971544, 0.9921940, 0.9978850, -0.0043805, 0.0032773
9: -0.0133042, -0.0085166, -0.0139673, -0.0088016, -0.0029749, 0.0039763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019548
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019070
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0086170, 0.0062467, 0.0088458, -0.0017758, 0.0015169
1: 0.0022255, 0.0025672, 0.0022248, 0.0026003, -0.0002565, 0.0002192
2: 0.0095958, 0.0109035, 0.0094692, 0.0109062, -0.0008387, 0.0009818
3: -0.0047561, -0.0034035, -0.0048869, -0.0034007, -0.0008674, 0.0010154
4: -0.0003524, 0.0011118, -0.0003555, 0.0012534, -0.0010992, 0.0009390
5: 0.0030613, 0.0044469, 0.0029272, 0.0044497, -0.0008886, 0.0010402
6: -0.0101542, -0.0046565, -0.0106861, -0.0046451, -0.0035258, 0.0041274
7: 0.0037850, 0.0112724, 0.0037695, 0.0119969, -0.0056212, 0.0048018
8: 0.9918801, 0.9971544, 0.9918692, 0.9976647, -0.0039597, 0.0033825
9: -0.0133042, -0.0085166, -0.0137675, -0.0085066, -0.0030704, 0.0035943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019548
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019070
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061923, 0.0086705, 0.0063885, 0.0089608, -0.0020149, 0.0014704
1: 0.0022169, 0.0025749, 0.0022453, 0.0026169, -0.0002911, 0.0002124
2: 0.0095661, 0.0109363, 0.0094057, 0.0108278, -0.0008129, 0.0011140
3: -0.0047867, -0.0033696, -0.0049527, -0.0034818, -0.0008408, 0.0011521
4: -0.0003892, 0.0011449, -0.0002677, 0.0013246, -0.0012472, 0.0009102
5: 0.0030299, 0.0044816, 0.0028598, 0.0043667, -0.0008613, 0.0011803
6: -0.0102787, -0.0045185, -0.0109535, -0.0049747, -0.0034175, 0.0046831
7: 0.0035971, 0.0114420, 0.0042184, 0.0123609, -0.0063780, 0.0046544
8: 0.9917478, 0.9972739, 0.9921855, 0.9979211, -0.0044928, 0.0032787
9: -0.0134127, -0.0083964, -0.0140003, -0.0087937, -0.0029761, 0.0040782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020773
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020111
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061923, 0.0086705, 0.0062428, 0.0088631, -0.0018460, 0.0015217
1: 0.0022169, 0.0025749, 0.0022242, 0.0026028, -0.0002667, 0.0002198
2: 0.0095661, 0.0109363, 0.0094597, 0.0109084, -0.0008413, 0.0010206
3: -0.0047867, -0.0033696, -0.0048968, -0.0033985, -0.0008701, 0.0010555
4: -0.0003892, 0.0011449, -0.0003579, 0.0012641, -0.0011427, 0.0009419
5: 0.0030299, 0.0044816, 0.0029170, 0.0044520, -0.0008914, 0.0010814
6: -0.0102787, -0.0045185, -0.0107264, -0.0046359, -0.0035368, 0.0042905
7: 0.0035971, 0.0114420, 0.0037570, 0.0120517, -0.0058433, 0.0048168
8: 0.9917478, 0.9972739, 0.9918604, 0.9977034, -0.0041161, 0.0033931
9: -0.0134127, -0.0083964, -0.0138025, -0.0084987, -0.0030800, 0.0037364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020773
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020111
time: 0.67 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0060757, 0.0085953, -0.0017964, 0.0019849
1: 0.0022315, 0.0025672, 0.0022001, 0.0025641, -0.0002595, 0.0002868
2: 0.0095960, 0.0108805, 0.0096077, 0.0110007, -0.0010974, 0.0009932
3: -0.0047559, -0.0034273, -0.0047437, -0.0033030, -0.0011350, 0.0010272
4: -0.0003267, 0.0011115, -0.0004613, 0.0010984, -0.0011120, 0.0012287
5: 0.0030615, 0.0044225, 0.0030739, 0.0045499, -0.0011628, 0.0010523
6: -0.0101534, -0.0047530, -0.0101038, -0.0042477, -0.0046136, 0.0041753
7: 0.0039165, 0.0112713, 0.0032282, 0.0112038, -0.0056864, 0.0062833
8: 0.9919728, 0.9971536, 0.9914879, 0.9971061, -0.0040056, 0.0044261
9: -0.0133035, -0.0086007, -0.0132604, -0.0081606, -0.0040177, 0.0036361

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021375
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021375
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0060725, 0.0086125, -0.0018421, 0.0020135
1: 0.0022231, 0.0025749, 0.0021996, 0.0025666, -0.0002661, 0.0002909
2: 0.0095662, 0.0109127, 0.0095982, 0.0110025, -0.0011132, 0.0010185
3: -0.0047866, -0.0033940, -0.0047535, -0.0033011, -0.0011513, 0.0010534
4: -0.0003628, 0.0011448, -0.0004633, 0.0011090, -0.0011403, 0.0012464
5: 0.0030299, 0.0044566, 0.0030639, 0.0045518, -0.0011795, 0.0010791
6: -0.0102784, -0.0046177, -0.0101438, -0.0042402, -0.0046799, 0.0042816
7: 0.0037322, 0.0114416, 0.0032181, 0.0112583, -0.0058312, 0.0063736
8: 0.9918429, 0.9972736, 0.9914808, 0.9971444, -0.0041076, 0.0044897
9: -0.0134124, -0.0084828, -0.0132952, -0.0081541, -0.0040754, 0.0037286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0022570
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0022570
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0063083, 0.0087857, 0.0060757, 0.0085953, -0.0017893, 0.0022067
1: 0.0022337, 0.0025916, 0.0022001, 0.0025641, -0.0002585, 0.0003188
2: 0.0095025, 0.0108722, 0.0096077, 0.0110007, -0.0012200, 0.0009893
3: -0.0048525, -0.0034360, -0.0047437, -0.0033030, -0.0012618, 0.0010231
4: -0.0003173, 0.0012162, -0.0004613, 0.0010984, -0.0011076, 0.0013660
5: 0.0029624, 0.0044136, 0.0030739, 0.0045499, -0.0012927, 0.0010482
6: -0.0105463, -0.0047883, -0.0101038, -0.0042477, -0.0051290, 0.0041588
7: 0.0039645, 0.0118064, 0.0032282, 0.0112038, -0.0056640, 0.0069852
8: 0.9920066, 0.9975305, 0.9914879, 0.9971061, -0.0039898, 0.0049205
9: -0.0136457, -0.0086314, -0.0132604, -0.0081606, -0.0044666, 0.0036217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020435
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020435
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0060725, 0.0086125, -0.0018407, 0.0022323
1: 0.0022250, 0.0025993, 0.0021996, 0.0025666, -0.0002659, 0.0003225
2: 0.0094731, 0.0109054, 0.0095982, 0.0110025, -0.0012342, 0.0010177
3: -0.0048830, -0.0034016, -0.0047535, -0.0033011, -0.0012765, 0.0010526
4: -0.0003545, 0.0012491, -0.0004633, 0.0011090, -0.0011394, 0.0013818
5: 0.0029312, 0.0044488, 0.0030639, 0.0045518, -0.0013077, 0.0010783
6: -0.0106700, -0.0046486, -0.0101438, -0.0042402, -0.0051885, 0.0042784
7: 0.0037743, 0.0119750, 0.0032181, 0.0112583, -0.0058268, 0.0070663
8: 0.9918726, 0.9976493, 0.9914808, 0.9971444, -0.0041045, 0.0049777
9: -0.0137535, -0.0085097, -0.0132952, -0.0081541, -0.0045184, 0.0037258

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0021456
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0021456
time: 0.63 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0086170, 0.0061001, 0.0087421, -0.0019741, 0.0019789
1: 0.0022255, 0.0025672, 0.0022036, 0.0025853, -0.0002852, 0.0002859
2: 0.0095958, 0.0109035, 0.0095266, 0.0109873, -0.0010941, 0.0010914
3: -0.0047561, -0.0034035, -0.0048277, -0.0033169, -0.0011316, 0.0011288
4: -0.0003524, 0.0011118, -0.0004462, 0.0011893, -0.0012220, 0.0012250
5: 0.0030613, 0.0044469, 0.0029879, 0.0045356, -0.0011592, 0.0011564
6: -0.0101542, -0.0046565, -0.0104452, -0.0043043, -0.0045996, 0.0045883
7: 0.0037850, 0.0112724, 0.0033054, 0.0116687, -0.0062489, 0.0062642
8: 0.9918801, 0.9971544, 0.9915423, 0.9974335, -0.0044019, 0.0044126
9: -0.0133042, -0.0085166, -0.0135576, -0.0082099, -0.0040055, 0.0039957

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020943
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020435
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062516, 0.0086170, 0.0059528, 0.0086626, -0.0017853, 0.0020263
1: 0.0022255, 0.0025672, 0.0021823, 0.0025738, -0.0002579, 0.0002927
2: 0.0095958, 0.0109035, 0.0095705, 0.0110687, -0.0011203, 0.0009871
3: -0.0047561, -0.0034035, -0.0047822, -0.0032327, -0.0011586, 0.0010209
4: -0.0003524, 0.0011118, -0.0005374, 0.0011400, -0.0011051, 0.0012543
5: 0.0030613, 0.0044469, 0.0030345, 0.0046219, -0.0011870, 0.0010458
6: -0.0101542, -0.0046565, -0.0102602, -0.0039619, -0.0047096, 0.0041496
7: 0.0037850, 0.0112724, 0.0028391, 0.0114168, -0.0056513, 0.0064141
8: 0.9918801, 0.9971544, 0.9912138, 0.9972562, -0.0039809, 0.0045182
9: -0.0133042, -0.0085166, -0.0133966, -0.0079117, -0.0041013, 0.0036136

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020943
time: 0.80 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020435
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061923, 0.0086705, 0.0060969, 0.0087605, -0.0020216, 0.0020042
1: 0.0022169, 0.0025749, 0.0022031, 0.0025879, -0.0002921, 0.0002896
2: 0.0095661, 0.0109363, 0.0095164, 0.0109890, -0.0011081, 0.0011177
3: -0.0047867, -0.0033696, -0.0048382, -0.0033151, -0.0011460, 0.0011559
4: -0.0003892, 0.0011449, -0.0004482, 0.0012006, -0.0012514, 0.0012407
5: 0.0030299, 0.0044816, 0.0029771, 0.0045375, -0.0011741, 0.0011842
6: -0.0102787, -0.0045185, -0.0104879, -0.0042970, -0.0046584, 0.0046987
7: 0.0035971, 0.0114420, 0.0032954, 0.0117269, -0.0063992, 0.0063443
8: 0.9917478, 0.9972739, 0.9915352, 0.9974745, -0.0045077, 0.0044691
9: -0.0134127, -0.0083964, -0.0135948, -0.0082035, -0.0040567, 0.0040918

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0022317
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0021456
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061923, 0.0086705, 0.0059493, 0.0086791, -0.0018530, 0.0020539
1: 0.0022169, 0.0025749, 0.0021818, 0.0025762, -0.0002677, 0.0002967
2: 0.0095661, 0.0109363, 0.0095614, 0.0110706, -0.0011356, 0.0010245
3: -0.0047867, -0.0033696, -0.0047916, -0.0032307, -0.0011745, 0.0010596
4: -0.0003892, 0.0011449, -0.0005396, 0.0011502, -0.0011470, 0.0012714
5: 0.0030299, 0.0044816, 0.0030249, 0.0046240, -0.0012032, 0.0010855
6: -0.0102787, -0.0045185, -0.0102985, -0.0039538, -0.0047739, 0.0043069
7: 0.0035971, 0.0114420, 0.0028280, 0.0114690, -0.0058657, 0.0065016
8: 0.9917478, 0.9972739, 0.9912060, 0.9972929, -0.0041319, 0.0045799
9: -0.0134127, -0.0083964, -0.0134299, -0.0079047, -0.0041573, 0.0037507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0022317
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0021456
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0060728, 0.0086059, 0.0063738, 0.0087882, -0.0020552, 0.0015996
1: 0.0021996, 0.0025656, 0.0022431, 0.0025919, -0.0002969, 0.0002311
2: 0.0096019, 0.0110024, 0.0095011, 0.0108359, -0.0008844, 0.0011363
3: -0.0047497, -0.0033013, -0.0048540, -0.0034734, -0.0009147, 0.0011752
4: -0.0004631, 0.0011049, -0.0002768, 0.0012178, -0.0012722, 0.0009902
5: 0.0030677, 0.0045516, 0.0029609, 0.0043753, -0.0009371, 0.0012039
6: -0.0101285, -0.0042408, -0.0105522, -0.0049405, -0.0037180, 0.0047768
7: 0.0032189, 0.0112374, 0.0041718, 0.0118144, -0.0065056, 0.0050635
8: 0.9914814, 0.9971297, 0.9921526, 0.9975362, -0.0045827, 0.0035669
9: -0.0132819, -0.0081546, -0.0136508, -0.0087639, -0.0032378, 0.0041599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0021703
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021982, upper bound: 0.0021703
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060728, 0.0086059, 0.0063872, 0.0089668, -0.0022710, 0.0016113
1: 0.0021996, 0.0025656, 0.0022451, 0.0026177, -0.0003281, 0.0002328
2: 0.0096019, 0.0110024, 0.0094024, 0.0108286, -0.0008909, 0.0012556
3: -0.0047497, -0.0033013, -0.0049561, -0.0034811, -0.0009214, 0.0012986
4: -0.0004631, 0.0011049, -0.0002685, 0.0013283, -0.0014058, 0.0009974
5: 0.0030677, 0.0045516, 0.0028563, 0.0043675, -0.0009439, 0.0013303
6: -0.0101285, -0.0042408, -0.0109672, -0.0049716, -0.0037452, 0.0052784
7: 0.0032189, 0.0112374, 0.0042141, 0.0123797, -0.0071887, 0.0051006
8: 0.9914814, 0.9971297, 0.9921824, 0.9979343, -0.0050639, 0.0035930
9: -0.0132819, -0.0081546, -0.0140123, -0.0087910, -0.0032615, 0.0045967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0022108
time: 0.61 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021982, upper bound: 0.0022110
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0059356, 0.0085146, 0.0063738, 0.0087882, -0.0022602, 0.0015662
1: 0.0021798, 0.0025524, 0.0022431, 0.0025919, -0.0003265, 0.0002263
2: 0.0096524, 0.0110782, 0.0095011, 0.0108359, -0.0008659, 0.0012496
3: -0.0046975, -0.0032228, -0.0048540, -0.0034734, -0.0008956, 0.0012924
4: -0.0005481, 0.0010484, -0.0002768, 0.0012178, -0.0013991, 0.0009695
5: 0.0031212, 0.0046320, 0.0029609, 0.0043753, -0.0009175, 0.0013240
6: -0.0099162, -0.0039219, -0.0105522, -0.0049405, -0.0036403, 0.0052532
7: 0.0027846, 0.0109483, 0.0041718, 0.0118144, -0.0071545, 0.0049578
8: 0.9911755, 0.9969261, 0.9921526, 0.9975362, -0.0050398, 0.0034924
9: -0.0130970, -0.0078769, -0.0136508, -0.0087639, -0.0031702, 0.0045748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0021097
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0021097
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0059356, 0.0085146, 0.0063872, 0.0089668, -0.0024760, 0.0015779
1: 0.0021798, 0.0025524, 0.0022451, 0.0026177, -0.0003577, 0.0002280
2: 0.0096524, 0.0110782, 0.0094024, 0.0108286, -0.0008724, 0.0013689
3: -0.0046975, -0.0032228, -0.0049561, -0.0034811, -0.0009023, 0.0014158
4: -0.0005481, 0.0010484, -0.0002685, 0.0013283, -0.0015327, 0.0009768
5: 0.0031212, 0.0046320, 0.0028563, 0.0043675, -0.0009244, 0.0014504
6: -0.0099162, -0.0039219, -0.0109672, -0.0049716, -0.0036676, 0.0057548
7: 0.0027846, 0.0109483, 0.0042141, 0.0123797, -0.0078376, 0.0049949
8: 0.9911755, 0.9969261, 0.9921824, 0.9979343, -0.0055209, 0.0035185
9: -0.0130970, -0.0078769, -0.0140123, -0.0087910, -0.0031939, 0.0050116

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0021143
time: 0.64 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0021143
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061001, 0.0087421, 0.0063909, 0.0087029, -0.0019896, 0.0017412
1: 0.0022036, 0.0025853, 0.0022456, 0.0025796, -0.0002874, 0.0002516
2: 0.0095266, 0.0109873, 0.0095482, 0.0108265, -0.0009627, 0.0011000
3: -0.0048277, -0.0033169, -0.0048052, -0.0034832, -0.0009956, 0.0011377
4: -0.0004462, 0.0011893, -0.0002662, 0.0011650, -0.0012316, 0.0010778
5: 0.0029879, 0.0045356, 0.0030109, 0.0043652, -0.0010200, 0.0011655
6: -0.0104452, -0.0043043, -0.0103540, -0.0049803, -0.0040470, 0.0046244
7: 0.0033054, 0.0116687, 0.0042261, 0.0115445, -0.0062981, 0.0055117
8: 0.9915423, 0.9974335, 0.9921908, 0.9973460, -0.0044365, 0.0038826
9: -0.0135576, -0.0082099, -0.0134782, -0.0087986, -0.0035243, 0.0040272

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021346, upper bound: 0.0021100
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021347, upper bound: 0.0021100
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0060969, 0.0087605, 0.0063336, 0.0087596, -0.0020182, 0.0018107
1: 0.0022031, 0.0025879, 0.0022373, 0.0025878, -0.0002916, 0.0002616
2: 0.0095164, 0.0109890, 0.0095169, 0.0108582, -0.0010011, 0.0011158
3: -0.0048382, -0.0033151, -0.0048376, -0.0034504, -0.0010354, 0.0011540
4: -0.0004482, 0.0012006, -0.0003017, 0.0012001, -0.0012493, 0.0011209
5: 0.0029771, 0.0045375, 0.0029777, 0.0043989, -0.0010607, 0.0011822
6: -0.0104879, -0.0042970, -0.0104858, -0.0048470, -0.0042087, 0.0046908
7: 0.0032954, 0.0117269, 0.0040444, 0.0117240, -0.0063884, 0.0057319
8: 0.9915352, 0.9974745, 0.9920629, 0.9974726, -0.0045001, 0.0040376
9: -0.0135948, -0.0082035, -0.0135930, -0.0086825, -0.0036651, 0.0040849

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023245, upper bound: 0.0021100
time: 0.62 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023245, upper bound: 0.0021100
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0059528, 0.0086626, 0.0063909, 0.0087029, -0.0021926, 0.0017016
1: 0.0021823, 0.0025738, 0.0022456, 0.0025796, -0.0003168, 0.0002458
2: 0.0095705, 0.0110687, 0.0095482, 0.0108265, -0.0009408, 0.0012122
3: -0.0047822, -0.0032327, -0.0048052, -0.0034832, -0.0009730, 0.0012537
4: -0.0005374, 0.0011400, -0.0002662, 0.0011650, -0.0013572, 0.0010533
5: 0.0030345, 0.0046219, 0.0030109, 0.0043652, -0.0009968, 0.0012844
6: -0.0102602, -0.0039619, -0.0103540, -0.0049803, -0.0039550, 0.0050962
7: 0.0028391, 0.0114168, 0.0042261, 0.0115445, -0.0069405, 0.0053864
8: 0.9912138, 0.9972562, 0.9921908, 0.9973460, -0.0048891, 0.0037943
9: -0.0133966, -0.0079117, -0.0134782, -0.0087986, -0.0034442, 0.0044380

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020130
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020130
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0059493, 0.0086791, 0.0063336, 0.0087596, -0.0022270, 0.0017702
1: 0.0021818, 0.0025762, 0.0022373, 0.0025878, -0.0003217, 0.0002557
2: 0.0095614, 0.0110706, 0.0095169, 0.0108582, -0.0009787, 0.0012313
3: -0.0047916, -0.0032307, -0.0048376, -0.0034504, -0.0010122, 0.0012734
4: -0.0005396, 0.0011502, -0.0003017, 0.0012001, -0.0013786, 0.0010958
5: 0.0030249, 0.0046240, 0.0029777, 0.0043989, -0.0010370, 0.0013046
6: -0.0102985, -0.0039538, -0.0104858, -0.0048470, -0.0041143, 0.0051762
7: 0.0028280, 0.0114690, 0.0040444, 0.0117240, -0.0070496, 0.0056034
8: 0.9912060, 0.9972929, 0.9920629, 0.9974726, -0.0049659, 0.0039471
9: -0.0134299, -0.0079047, -0.0135930, -0.0086825, -0.0035829, 0.0045077

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020130
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020130
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0060757, 0.0085953, 0.0062932, 0.0086166, -0.0019849, 0.0017964
1: 0.0022001, 0.0025641, 0.0022315, 0.0025672, -0.0002868, 0.0002595
2: 0.0096077, 0.0110007, 0.0095960, 0.0108805, -0.0009932, 0.0010974
3: -0.0047437, -0.0033030, -0.0047559, -0.0034273, -0.0010272, 0.0011350
4: -0.0004613, 0.0010984, -0.0003267, 0.0011115, -0.0012287, 0.0011120
5: 0.0030739, 0.0045499, 0.0030615, 0.0044225, -0.0010523, 0.0011628
6: -0.0101038, -0.0042477, -0.0101534, -0.0047530, -0.0041753, 0.0046136
7: 0.0032282, 0.0112038, 0.0039165, 0.0112713, -0.0062833, 0.0056864
8: 0.9914879, 0.9971061, 0.9919728, 0.9971536, -0.0044261, 0.0040056
9: -0.0132604, -0.0081606, -0.0133035, -0.0086007, -0.0036361, 0.0040177

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020989, upper bound: 0.0021497
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020989, upper bound: 0.0021577
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060725, 0.0086125, 0.0062349, 0.0086704, -0.0020135, 0.0018421
1: 0.0021996, 0.0025666, 0.0022231, 0.0025749, -0.0002909, 0.0002661
2: 0.0095982, 0.0110025, 0.0095662, 0.0109127, -0.0010185, 0.0011132
3: -0.0047535, -0.0033011, -0.0047866, -0.0033940, -0.0010534, 0.0011513
4: -0.0004633, 0.0011090, -0.0003628, 0.0011448, -0.0012464, 0.0011403
5: 0.0030639, 0.0045518, 0.0030299, 0.0044566, -0.0010791, 0.0011795
6: -0.0101438, -0.0042402, -0.0102784, -0.0046177, -0.0042816, 0.0046799
7: 0.0032181, 0.0112583, 0.0037322, 0.0114416, -0.0063736, 0.0058312
8: 0.9914808, 0.9971444, 0.9918429, 0.9972736, -0.0044897, 0.0041076
9: -0.0132952, -0.0081541, -0.0134124, -0.0084828, -0.0037286, 0.0040754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022493, upper bound: 0.0021497
time: 0.74 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022493, upper bound: 0.0021577
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060757, 0.0085953, 0.0063083, 0.0087857, -0.0022067, 0.0017893
1: 0.0022001, 0.0025641, 0.0022337, 0.0025916, -0.0003188, 0.0002585
2: 0.0096077, 0.0110007, 0.0095025, 0.0108722, -0.0009893, 0.0012200
3: -0.0047437, -0.0033030, -0.0048525, -0.0034360, -0.0010231, 0.0012618
4: -0.0004613, 0.0010984, -0.0003173, 0.0012162, -0.0013660, 0.0011076
5: 0.0030739, 0.0045499, 0.0029624, 0.0044136, -0.0010482, 0.0012927
6: -0.0101038, -0.0042477, -0.0105463, -0.0047883, -0.0041588, 0.0051290
7: 0.0032282, 0.0112038, 0.0039645, 0.0118064, -0.0069852, 0.0056640
8: 0.9914879, 0.9971061, 0.9920066, 0.9975305, -0.0049206, 0.0039898
9: -0.0132604, -0.0081606, -0.0136457, -0.0086314, -0.0036217, 0.0044666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020220, upper bound: 0.0021143
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020220, upper bound: 0.0021143
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060725, 0.0086125, 0.0062482, 0.0088389, -0.0022323, 0.0018407
1: 0.0021996, 0.0025666, 0.0022250, 0.0025993, -0.0003225, 0.0002659
2: 0.0095982, 0.0110025, 0.0094731, 0.0109054, -0.0010177, 0.0012342
3: -0.0047535, -0.0033011, -0.0048830, -0.0034016, -0.0010526, 0.0012765
4: -0.0004633, 0.0011090, -0.0003545, 0.0012491, -0.0013818, 0.0011394
5: 0.0030639, 0.0045518, 0.0029312, 0.0044488, -0.0010783, 0.0013077
6: -0.0101438, -0.0042402, -0.0106700, -0.0046486, -0.0042784, 0.0051885
7: 0.0032181, 0.0112583, 0.0037743, 0.0119750, -0.0070663, 0.0058268
8: 0.9914808, 0.9971444, 0.9918726, 0.9976493, -0.0049777, 0.0041045
9: -0.0132952, -0.0081541, -0.0137535, -0.0085097, -0.0037258, 0.0045184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021426, upper bound: 0.0021143
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021426, upper bound: 0.0021143
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061001, 0.0087421, 0.0062516, 0.0086170, -0.0019789, 0.0019741
1: 0.0022036, 0.0025853, 0.0022255, 0.0025672, -0.0002859, 0.0002852
2: 0.0095266, 0.0109873, 0.0095958, 0.0109035, -0.0010914, 0.0010941
3: -0.0048277, -0.0033169, -0.0047561, -0.0034035, -0.0011288, 0.0011316
4: -0.0004462, 0.0011893, -0.0003524, 0.0011118, -0.0012250, 0.0012220
5: 0.0029879, 0.0045356, 0.0030613, 0.0044469, -0.0011564, 0.0011592
6: -0.0104452, -0.0043043, -0.0101542, -0.0046565, -0.0045883, 0.0045996
7: 0.0033054, 0.0116687, 0.0037850, 0.0112724, -0.0062642, 0.0062489
8: 0.9915423, 0.9974335, 0.9918801, 0.9971544, -0.0044126, 0.0044019
9: -0.0135576, -0.0082099, -0.0133042, -0.0085166, -0.0039957, 0.0040055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020107
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020107
time: 0.74 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0059528, 0.0086626, 0.0062516, 0.0086170, -0.0020263, 0.0017853
1: 0.0021823, 0.0025738, 0.0022255, 0.0025672, -0.0002927, 0.0002579
2: 0.0095705, 0.0110687, 0.0095958, 0.0109035, -0.0009871, 0.0011203
3: -0.0047822, -0.0032327, -0.0047561, -0.0034035, -0.0010209, 0.0011586
4: -0.0005374, 0.0011400, -0.0003524, 0.0011118, -0.0012543, 0.0011051
5: 0.0030345, 0.0046219, 0.0030613, 0.0044469, -0.0010458, 0.0011870
6: -0.0102602, -0.0039619, -0.0101542, -0.0046565, -0.0041496, 0.0047096
7: 0.0028391, 0.0114168, 0.0037850, 0.0112724, -0.0064141, 0.0056513
8: 0.9912138, 0.9972562, 0.9918801, 0.9971544, -0.0045182, 0.0039809
9: -0.0133966, -0.0079117, -0.0133042, -0.0085166, -0.0036136, 0.0041013

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020111
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020111
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0060969, 0.0087605, 0.0061923, 0.0086705, -0.0020042, 0.0020216
1: 0.0022031, 0.0025879, 0.0022169, 0.0025749, -0.0002896, 0.0002921
2: 0.0095164, 0.0109890, 0.0095661, 0.0109363, -0.0011177, 0.0011081
3: -0.0048382, -0.0033151, -0.0047867, -0.0033696, -0.0011559, 0.0011460
4: -0.0004482, 0.0012006, -0.0003892, 0.0011449, -0.0012407, 0.0012514
5: 0.0029771, 0.0045375, 0.0030299, 0.0044816, -0.0011842, 0.0011741
6: -0.0104879, -0.0042970, -0.0102787, -0.0045185, -0.0046987, 0.0046584
7: 0.0032954, 0.0117269, 0.0035971, 0.0114420, -0.0063443, 0.0063992
8: 0.9915352, 0.9974745, 0.9917478, 0.9972739, -0.0044691, 0.0045077
9: -0.0135948, -0.0082035, -0.0134127, -0.0083964, -0.0040918, 0.0040567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020107
time: 0.78 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020107
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0059493, 0.0086791, 0.0061923, 0.0086705, -0.0020539, 0.0018530
1: 0.0021818, 0.0025762, 0.0022169, 0.0025749, -0.0002967, 0.0002677
2: 0.0095614, 0.0110706, 0.0095661, 0.0109363, -0.0010245, 0.0011356
3: -0.0047916, -0.0032307, -0.0047867, -0.0033696, -0.0010596, 0.0011745
4: -0.0005396, 0.0011502, -0.0003892, 0.0011449, -0.0012714, 0.0011470
5: 0.0030249, 0.0046240, 0.0030299, 0.0044816, -0.0010855, 0.0012032
6: -0.0102985, -0.0039538, -0.0102787, -0.0045185, -0.0043069, 0.0047739
7: 0.0028280, 0.0114690, 0.0035971, 0.0114420, -0.0065016, 0.0058657
8: 0.9912060, 0.9972929, 0.9917478, 0.9972739, -0.0045799, 0.0041319
9: -0.0134299, -0.0079047, -0.0134127, -0.0083964, -0.0037507, 0.0041573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020111
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020111
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0060728, 0.0086059, 0.0060728, 0.0086059, -0.0015790, 0.0015790
1: 0.0021996, 0.0025656, 0.0021996, 0.0025656, -0.0002281, 0.0002281
2: 0.0096019, 0.0110024, 0.0096019, 0.0110024, -0.0008730, 0.0008730
3: -0.0047497, -0.0033013, -0.0047497, -0.0033013, -0.0009029, 0.0009029
4: -0.0004631, 0.0011049, -0.0004631, 0.0011049, -0.0009774, 0.0009774
5: 0.0030677, 0.0045516, 0.0030677, 0.0045516, -0.0009250, 0.0009250
6: -0.0101285, -0.0042408, -0.0101285, -0.0042408, -0.0036700, 0.0036700
7: 0.0032189, 0.0112374, 0.0032189, 0.0112374, -0.0049983, 0.0049983
8: 0.9914814, 0.9971297, 0.9914814, 0.9971297, -0.0035209, 0.0035209
9: -0.0132819, -0.0081546, -0.0132819, -0.0081546, -0.0031960, 0.0031960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0020507
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0021607
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0060957, 0.0087656, 0.0060728, 0.0086059, -0.0015916, 0.0017953
1: 0.0022029, 0.0025887, 0.0021996, 0.0025656, -0.0002299, 0.0002594
2: 0.0095136, 0.0109897, 0.0096019, 0.0110024, -0.0009926, 0.0008799
3: -0.0048411, -0.0033144, -0.0047497, -0.0033013, -0.0010266, 0.0009101
4: -0.0004490, 0.0012038, -0.0004631, 0.0011049, -0.0009852, 0.0011113
5: 0.0029742, 0.0045382, 0.0030677, 0.0045516, -0.0010517, 0.0009323
6: -0.0104996, -0.0042940, -0.0101285, -0.0042408, -0.0041727, 0.0036993
7: 0.0032913, 0.0117429, 0.0032189, 0.0112374, -0.0050381, 0.0056829
8: 0.9915324, 0.9974858, 0.9914814, 0.9971297, -0.0035489, 0.0040031
9: -0.0136051, -0.0082009, -0.0132819, -0.0081546, -0.0036338, 0.0032215

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_B1_A2_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0020507
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_A2_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0021607
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0060728, 0.0086059, 0.0059356, 0.0085146, -0.0015632, 0.0017935
1: 0.0021996, 0.0025656, 0.0021798, 0.0025524, -0.0002258, 0.0002591
2: 0.0096019, 0.0110024, 0.0096524, 0.0110782, -0.0009916, 0.0008643
3: -0.0047497, -0.0033013, -0.0046975, -0.0032228, -0.0010255, 0.0008939
4: -0.0004631, 0.0011049, -0.0005481, 0.0010484, -0.0009677, 0.0011102
5: 0.0030677, 0.0045516, 0.0031212, 0.0046320, -0.0010506, 0.0009157
6: -0.0101285, -0.0042408, -0.0099162, -0.0039219, -0.0041686, 0.0036334
7: 0.0032189, 0.0112374, 0.0027846, 0.0109483, -0.0049483, 0.0056773
8: 0.9914814, 0.9971297, 0.9911755, 0.9969261, -0.0034857, 0.0039992
9: -0.0132819, -0.0081546, -0.0130970, -0.0078769, -0.0036302, 0.0031641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021642, upper bound: 0.0021351
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021351
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0060957, 0.0087656, 0.0059356, 0.0085146, -0.0015758, 0.0020098
1: 0.0022029, 0.0025887, 0.0021798, 0.0025524, -0.0002277, 0.0002904
2: 0.0095136, 0.0109897, 0.0096524, 0.0110782, -0.0011112, 0.0008712
3: -0.0048411, -0.0033144, -0.0046975, -0.0032228, -0.0011492, 0.0009011
4: -0.0004490, 0.0012038, -0.0005481, 0.0010484, -0.0009754, 0.0012441
5: 0.0029742, 0.0045382, 0.0031212, 0.0046320, -0.0011773, 0.0009231
6: -0.0104996, -0.0042940, -0.0099162, -0.0039219, -0.0046713, 0.0036626
7: 0.0032913, 0.0117429, 0.0027846, 0.0109483, -0.0049881, 0.0063619
8: 0.9915324, 0.9974858, 0.9911755, 0.9969261, -0.0035137, 0.0044815
9: -0.0136051, -0.0082009, -0.0130970, -0.0078769, -0.0040680, 0.0031896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021642, upper bound: 0.0021351
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021351
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0060335, 0.0085833, 0.0061661, 0.0086790, -0.0017760, 0.0015204
1: 0.0021940, 0.0025623, 0.0022131, 0.0025762, -0.0002566, 0.0002197
2: 0.0096144, 0.0110241, 0.0095615, 0.0109508, -0.0008406, 0.0009819
3: -0.0047368, -0.0032788, -0.0047916, -0.0033546, -0.0008694, 0.0010155
4: -0.0004874, 0.0010909, -0.0004054, 0.0011502, -0.0010994, 0.0009412
5: 0.0030810, 0.0045746, 0.0030249, 0.0044970, -0.0008907, 0.0010404
6: -0.0100759, -0.0041495, -0.0102984, -0.0044577, -0.0035339, 0.0041278
7: 0.0030945, 0.0111658, 0.0035142, 0.0114689, -0.0056217, 0.0048128
8: 0.9913937, 0.9970793, 0.9916894, 0.9972928, -0.0039601, 0.0033902
9: -0.0132361, -0.0080751, -0.0134299, -0.0083434, -0.0030774, 0.0035947

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021308, upper bound: 0.0022910
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021308, upper bound: 0.0021607
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060305, 0.0086005, 0.0061020, 0.0087390, -0.0017769, 0.0015838
1: 0.0021935, 0.0025648, 0.0022039, 0.0025848, -0.0002567, 0.0002288
2: 0.0096049, 0.0110258, 0.0095283, 0.0109862, -0.0008756, 0.0009824
3: -0.0047467, -0.0032771, -0.0048259, -0.0033180, -0.0009056, 0.0010161
4: -0.0004893, 0.0011016, -0.0004451, 0.0011873, -0.0011000, 0.0009804
5: 0.0030709, 0.0045764, 0.0029898, 0.0045345, -0.0009278, 0.0010409
6: -0.0101160, -0.0041425, -0.0104379, -0.0043087, -0.0036812, 0.0041301
7: 0.0030850, 0.0112204, 0.0033113, 0.0116588, -0.0056249, 0.0050134
8: 0.9913870, 0.9971177, 0.9915464, 0.9974266, -0.0039623, 0.0035316
9: -0.0132710, -0.0080690, -0.0135513, -0.0082137, -0.0032057, 0.0035967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022394, upper bound: 0.0022910
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022394, upper bound: 0.0021607
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060335, 0.0085833, 0.0060113, 0.0086027, -0.0017327, 0.0017536
1: 0.0021940, 0.0025623, 0.0021908, 0.0025651, -0.0002503, 0.0002534
2: 0.0096144, 0.0110241, 0.0096036, 0.0110364, -0.0009695, 0.0009579
3: -0.0047368, -0.0032788, -0.0047479, -0.0032661, -0.0010027, 0.0009908
4: -0.0004874, 0.0010909, -0.0005012, 0.0011030, -0.0010725, 0.0010855
5: 0.0030810, 0.0045746, 0.0030696, 0.0045876, -0.0010273, 0.0010150
6: -0.0100759, -0.0041495, -0.0101211, -0.0040980, -0.0040759, 0.0040272
7: 0.0030945, 0.0111658, 0.0030244, 0.0112274, -0.0054847, 0.0055511
8: 0.9913937, 0.9970793, 0.9913443, 0.9971227, -0.0038635, 0.0039103
9: -0.0132361, -0.0080751, -0.0132755, -0.0080302, -0.0035495, 0.0035071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0022639
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0021351
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060305, 0.0086005, 0.0059547, 0.0086548, -0.0017337, 0.0017977
1: 0.0021935, 0.0025648, 0.0021826, 0.0025727, -0.0002505, 0.0002597
2: 0.0096049, 0.0110258, 0.0095748, 0.0110677, -0.0009939, 0.0009585
3: -0.0047467, -0.0032771, -0.0047777, -0.0032338, -0.0010280, 0.0009914
4: -0.0004893, 0.0011016, -0.0005362, 0.0011352, -0.0010732, 0.0011128
5: 0.0030709, 0.0045764, 0.0030391, 0.0046208, -0.0010531, 0.0010156
6: -0.0101160, -0.0041425, -0.0102422, -0.0039664, -0.0041785, 0.0040297
7: 0.0030850, 0.0112204, 0.0028452, 0.0113922, -0.0054880, 0.0056907
8: 0.9913870, 0.9971177, 0.9912180, 0.9972388, -0.0038659, 0.0040087
9: -0.0132710, -0.0080690, -0.0133808, -0.0079156, -0.0036388, 0.0035092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022153, upper bound: 0.0022639
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022153, upper bound: 0.0021351
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0059965, 0.0084314, 0.0060757, 0.0085953, -0.0017773, 0.0015161
1: 0.0021886, 0.0025404, 0.0022001, 0.0025641, -0.0002568, 0.0002190
2: 0.0096983, 0.0110446, 0.0096077, 0.0110007, -0.0008382, 0.0009826
3: -0.0046500, -0.0032577, -0.0047437, -0.0033030, -0.0008669, 0.0010163
4: -0.0005104, 0.0009969, -0.0004613, 0.0010984, -0.0011002, 0.0009385
5: 0.0031699, 0.0045963, 0.0030739, 0.0045499, -0.0008881, 0.0010411
6: -0.0097229, -0.0040635, -0.0101038, -0.0042477, -0.0035239, 0.0041310
7: 0.0029774, 0.0106851, 0.0032282, 0.0112038, -0.0056260, 0.0047992
8: 0.9913112, 0.9967407, 0.9914879, 0.9971061, -0.0039631, 0.0033807
9: -0.0129287, -0.0080002, -0.0132604, -0.0081606, -0.0030688, 0.0035974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0020894
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0020894
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0059414, 0.0084840, 0.0060725, 0.0086125, -0.0018253, 0.0015186
1: 0.0021807, 0.0025480, 0.0021996, 0.0025666, -0.0002637, 0.0002194
2: 0.0096693, 0.0110750, 0.0095982, 0.0110025, -0.0008396, 0.0010092
3: -0.0046801, -0.0032262, -0.0047535, -0.0033011, -0.0008684, 0.0010437
4: -0.0005445, 0.0010295, -0.0004633, 0.0011090, -0.0011299, 0.0009400
5: 0.0031391, 0.0046286, 0.0030639, 0.0045518, -0.0008896, 0.0010693
6: -0.0098452, -0.0039354, -0.0101438, -0.0042402, -0.0035297, 0.0042426
7: 0.0028030, 0.0108516, 0.0032181, 0.0112583, -0.0057781, 0.0048071
8: 0.9911883, 0.9968579, 0.9914808, 0.9971444, -0.0040702, 0.0033862
9: -0.0130352, -0.0078887, -0.0132952, -0.0081541, -0.0030738, 0.0036946

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0021953
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0021953
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0060113, 0.0086027, 0.0060757, 0.0085953, -0.0017715, 0.0017359
1: 0.0021908, 0.0025651, 0.0022001, 0.0025641, -0.0002559, 0.0002508
2: 0.0096036, 0.0110364, 0.0096077, 0.0110007, -0.0009598, 0.0009794
3: -0.0047479, -0.0032661, -0.0047437, -0.0033030, -0.0009926, 0.0010130
4: -0.0005012, 0.0011030, -0.0004613, 0.0010984, -0.0010966, 0.0010746
5: 0.0030696, 0.0045876, 0.0030739, 0.0045499, -0.0010169, 0.0010378
6: -0.0101211, -0.0040980, -0.0101038, -0.0042477, -0.0040348, 0.0041176
7: 0.0030244, 0.0112274, 0.0032282, 0.0112038, -0.0056078, 0.0054950
8: 0.9913443, 0.9971227, 0.9914879, 0.9971061, -0.0039502, 0.0038708
9: -0.0132755, -0.0080302, -0.0132604, -0.0081606, -0.0035137, 0.0035858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0020511
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0020511
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0059547, 0.0086548, 0.0060725, 0.0086125, -0.0018219, 0.0017373
1: 0.0021826, 0.0025727, 0.0021996, 0.0025666, -0.0002632, 0.0002510
2: 0.0095748, 0.0110677, 0.0095982, 0.0110025, -0.0009605, 0.0010073
3: -0.0047777, -0.0032338, -0.0047535, -0.0033011, -0.0009934, 0.0010418
4: -0.0005362, 0.0011352, -0.0004633, 0.0011090, -0.0011278, 0.0010754
5: 0.0030391, 0.0046208, 0.0030639, 0.0045518, -0.0010177, 0.0010672
6: -0.0102422, -0.0039664, -0.0101438, -0.0042402, -0.0040379, 0.0042345
7: 0.0028452, 0.0113922, 0.0032181, 0.0112583, -0.0057670, 0.0054993
8: 0.9912180, 0.9972388, 0.9914808, 0.9971444, -0.0040624, 0.0038738
9: -0.0133808, -0.0079156, -0.0132952, -0.0081541, -0.0035164, 0.0036876

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021362
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021362
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0059487, 0.0084320, 0.0061001, 0.0087421, -0.0020005, 0.0015022
1: 0.0021817, 0.0025405, 0.0022036, 0.0025853, -0.0002890, 0.0002170
2: 0.0096981, 0.0110710, 0.0095266, 0.0109873, -0.0008305, 0.0011060
3: -0.0046503, -0.0032304, -0.0048277, -0.0033169, -0.0008590, 0.0011439
4: -0.0005399, 0.0009972, -0.0004462, 0.0011893, -0.0012383, 0.0009299
5: 0.0031696, 0.0046243, 0.0029879, 0.0045356, -0.0008800, 0.0011719
6: -0.0097242, -0.0039525, -0.0104452, -0.0043043, -0.0034914, 0.0046496
7: 0.0028263, 0.0106868, 0.0033054, 0.0116687, -0.0063324, 0.0047550
8: 0.9912048, 0.9967418, 0.9915423, 0.9974335, -0.0044607, 0.0033495
9: -0.0129298, -0.0079036, -0.0135576, -0.0082099, -0.0030405, 0.0040491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021269
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0020511
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0059487, 0.0084320, 0.0059528, 0.0086626, -0.0018190, 0.0015596
1: 0.0021817, 0.0025405, 0.0021823, 0.0025738, -0.0002628, 0.0002253
2: 0.0096981, 0.0110710, 0.0095705, 0.0110687, -0.0008623, 0.0010057
3: -0.0046503, -0.0032304, -0.0047822, -0.0032327, -0.0008918, 0.0010401
4: -0.0005399, 0.0009972, -0.0005374, 0.0011400, -0.0011260, 0.0009654
5: 0.0031696, 0.0046243, 0.0030345, 0.0046219, -0.0009136, 0.0010656
6: -0.0097242, -0.0039525, -0.0102602, -0.0039619, -0.0036249, 0.0042278
7: 0.0028263, 0.0106868, 0.0028391, 0.0114168, -0.0057580, 0.0049368
8: 0.9912048, 0.9967418, 0.9912138, 0.9972562, -0.0040560, 0.0034776
9: -0.0129298, -0.0079036, -0.0133966, -0.0079117, -0.0031567, 0.0036818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021269
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0020511
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0058921, 0.0084844, 0.0060969, 0.0087605, -0.0020437, 0.0015053
1: 0.0021735, 0.0025480, 0.0022031, 0.0025879, -0.0002953, 0.0002175
2: 0.0096691, 0.0111023, 0.0095164, 0.0109890, -0.0008322, 0.0011299
3: -0.0046802, -0.0031980, -0.0048382, -0.0033151, -0.0008607, 0.0011686
4: -0.0005750, 0.0010297, -0.0004482, 0.0012006, -0.0012651, 0.0009318
5: 0.0031389, 0.0046575, 0.0029771, 0.0045375, -0.0008818, 0.0011972
6: -0.0098460, -0.0038209, -0.0104879, -0.0042970, -0.0034986, 0.0047501
7: 0.0026471, 0.0108527, 0.0032954, 0.0117269, -0.0064692, 0.0047648
8: 0.9910785, 0.9968587, 0.9915352, 0.9974745, -0.0045570, 0.0033564
9: -0.0130358, -0.0077889, -0.0135948, -0.0082035, -0.0030467, 0.0041366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0022554
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021362
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0058921, 0.0084844, 0.0059493, 0.0086791, -0.0018795, 0.0015617
1: 0.0021735, 0.0025480, 0.0021818, 0.0025762, -0.0002715, 0.0002256
2: 0.0096691, 0.0111023, 0.0095614, 0.0110706, -0.0008634, 0.0010391
3: -0.0046802, -0.0031980, -0.0047916, -0.0032307, -0.0008930, 0.0010747
4: -0.0005750, 0.0010297, -0.0005396, 0.0011502, -0.0011634, 0.0009667
5: 0.0031389, 0.0046575, 0.0030249, 0.0046240, -0.0009148, 0.0011010
6: -0.0098460, -0.0038209, -0.0102985, -0.0039538, -0.0036298, 0.0043685
7: 0.0026471, 0.0108527, 0.0028280, 0.0114690, -0.0059495, 0.0049434
8: 0.9910785, 0.9968587, 0.9912060, 0.9972929, -0.0041909, 0.0034823
9: -0.0130358, -0.0077889, -0.0134299, -0.0079047, -0.0031610, 0.0038042

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0022554
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021362
time: 0.74 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.21 seconds
IS_A1_A1_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0019991
IS_A1_A1_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0021703
IS_A1_A1_B1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0022108
IS_A1_A1_B1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0022110
IS_A1_A1_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022108, upper bound: 0.0019637
IS_A1_A1_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022108, upper bound: 0.0019637
IS_A1_A1_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022110, upper bound: 0.0021100
IS_A1_A1_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022110, upper bound: 0.0021100
IS_A1_A1_B1_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021310
IS_A1_A1_B1_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021310
IS_A1_A1_B1_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021434
IS_A1_A1_B1_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021434
IS_A1_A1_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
IS_A1_A1_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
IS_A1_A1_B1_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020780, upper bound: 0.0020392
IS_A1_A1_B1_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020780, upper bound: 0.0020392
IS_A1_A1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0020703
IS_A1_A1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0021982
IS_A1_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0020703
IS_A1_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021702, upper bound: 0.0021982
IS_A1_A1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020229
IS_A1_A1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021559
IS_A1_A1_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0020229
IS_A1_A1_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021097, upper bound: 0.0021558
IS_A1_A1_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0021347
IS_A1_A1_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0020703
IS_A1_A1_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0023245
IS_A1_A1_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0021982
IS_A1_A1_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0020835
IS_A1_A1_B2_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0020229
IS_A1_A1_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0022504
IS_A1_A1_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021559
IS_A1_A2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0020339
IS_A1_A2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0020339
IS_A1_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021577
IS_A1_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021577
IS_A1_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0019070
IS_A1_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0019070
IS_A1_A2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020111
IS_A1_A2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020111
IS_A1_A2_B1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019548
IS_A1_A2_B1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019070
IS_A1_A2_B1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019548
IS_A1_A2_B1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0019070
IS_A1_A2_B1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020773
IS_A1_A2_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020111
IS_A1_A2_B1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020773
IS_A1_A2_B1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020111
IS_A1_A2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021375
IS_A1_A2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0021375
IS_A1_A2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0022570
IS_A1_A2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021497, upper bound: 0.0022570
IS_A1_A2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020435
IS_A1_A2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0020435
IS_A1_A2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0021456
IS_A1_A2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021143, upper bound: 0.0021456
IS_A1_A2_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020943
IS_A1_A2_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020435
IS_A1_A2_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020943
IS_A1_A2_B2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0020435
IS_A1_A2_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0022317
IS_A1_A2_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0021456
IS_A1_A2_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0022317
IS_A1_A2_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020107, upper bound: 0.0021456
IS_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0021703
IS_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021982, upper bound: 0.0021703
IS_A2_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0022108
IS_A2_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021982, upper bound: 0.0022110
IS_A2_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0021097
IS_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0021097
IS_A2_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0021143
IS_A2_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0021143
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021346, upper bound: 0.0021100
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021347, upper bound: 0.0021100
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023245, upper bound: 0.0021100
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023245, upper bound: 0.0021100
IS_A2_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020130
IS_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020130
IS_A2_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020130
IS_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020130
IS_A2_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020989, upper bound: 0.0021497
IS_A2_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020989, upper bound: 0.0021577
IS_A2_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022493, upper bound: 0.0021497
IS_A2_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022493, upper bound: 0.0021577
IS_A2_B1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020220, upper bound: 0.0021143
IS_A2_B1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020220, upper bound: 0.0021143
IS_A2_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021426, upper bound: 0.0021143
IS_A2_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021426, upper bound: 0.0021143
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020107
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020107
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020111
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020111
IS_A2_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020107
IS_A2_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020107
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020111
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020111
IS_A2_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0020507
IS_A2_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0021607
IS_A2_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0020507
IS_A2_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0021607
IS_A2_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021642, upper bound: 0.0021351
IS_A2_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021351
IS_A2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021642, upper bound: 0.0021351
IS_A2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021351
IS_A2_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021308, upper bound: 0.0022910
IS_A2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021308, upper bound: 0.0021607
IS_A2_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022394, upper bound: 0.0022910
IS_A2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022394, upper bound: 0.0021607
IS_A2_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0022639
IS_A2_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0021351
IS_A2_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022153, upper bound: 0.0022639
IS_A2_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022153, upper bound: 0.0021351
IS_A2_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0020894
IS_A2_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0020894
IS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0021953
IS_A2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0021953
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0020511
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0020511
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021362
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021362
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021269
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0020511
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021269
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0020511
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0022554
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021362
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0022554
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.21
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021362

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0064408, 0.0087028, 0.0063785, 0.0087654, -0.0014658, 0.0014622
1: 0.0022528, 0.0025796, 0.0022438, 0.0025887, -0.0002118, 0.0002112
2: 0.0095483, 0.0107989, 0.0095137, 0.0108334, -0.0008084, 0.0008104
3: -0.0048052, -0.0035117, -0.0048410, -0.0034761, -0.0008361, 0.0008381
4: -0.0002353, 0.0011649, -0.0002739, 0.0012037, -0.0009073, 0.0009051
5: 0.0030109, 0.0043360, 0.0029743, 0.0043725, -0.0008566, 0.0008586
6: -0.0103538, -0.0050962, -0.0104993, -0.0049514, -0.0033986, 0.0034068
7: 0.0043839, 0.0115443, 0.0041866, 0.0117424, -0.0046398, 0.0046286
8: 0.9923019, 0.9973459, 0.9921631, 0.9974855, -0.0032684, 0.0032605
9: -0.0134781, -0.0088995, -0.0136047, -0.0087734, -0.0029597, 0.0029668

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0020470
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0020470
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063796, 0.0087597, 0.0063750, 0.0087825, -0.0015365, 0.0014691
1: 0.0022440, 0.0025878, 0.0022433, 0.0025911, -0.0002220, 0.0002122
2: 0.0095168, 0.0108327, 0.0095043, 0.0108353, -0.0008122, 0.0008495
3: -0.0048377, -0.0034767, -0.0048507, -0.0034741, -0.0008401, 0.0008786
4: -0.0002732, 0.0012001, -0.0002761, 0.0012142, -0.0009511, 0.0009094
5: 0.0029776, 0.0043719, 0.0029643, 0.0043746, -0.0008606, 0.0009001
6: -0.0104860, -0.0049540, -0.0105389, -0.0049432, -0.0034147, 0.0035713
7: 0.0041902, 0.0117243, 0.0041755, 0.0117963, -0.0048638, 0.0046505
8: 0.9921655, 0.9974728, 0.9921552, 0.9975234, -0.0034262, 0.0032759
9: -0.0135932, -0.0087757, -0.0136392, -0.0087663, -0.0029736, 0.0031101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021954
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021954
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0063785, 0.0087654, 0.0064493, 0.0088852, -0.0016828, 0.0014752
1: 0.0022438, 0.0025887, 0.0022540, 0.0026060, -0.0002431, 0.0002131
2: 0.0095137, 0.0108334, 0.0094475, 0.0107942, -0.0008156, 0.0009304
3: -0.0048410, -0.0034761, -0.0049094, -0.0035166, -0.0008436, 0.0009622
4: -0.0002739, 0.0012037, -0.0002301, 0.0012778, -0.0010417, 0.0009132
5: 0.0029743, 0.0043725, 0.0029041, 0.0043311, -0.0008642, 0.0009858
6: -0.0104993, -0.0049514, -0.0107776, -0.0051160, -0.0034289, 0.0039113
7: 0.0041866, 0.0117424, 0.0044108, 0.0121215, -0.0053269, 0.0046698
8: 0.9921631, 0.9974855, 0.9923210, 0.9977525, -0.0037524, 0.0032895
9: -0.0136047, -0.0087734, -0.0138472, -0.0089167, -0.0029860, 0.0034061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0020293
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0022108
time: 0.62 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0063750, 0.0087825, 0.0063940, 0.0089355, -0.0016863, 0.0015475
1: 0.0022433, 0.0025911, 0.0022460, 0.0026132, -0.0002436, 0.0002236
2: 0.0095043, 0.0108353, 0.0094197, 0.0108248, -0.0008556, 0.0009323
3: -0.0048507, -0.0034741, -0.0049382, -0.0034849, -0.0008849, 0.0009643
4: -0.0002761, 0.0012142, -0.0002643, 0.0013089, -0.0010439, 0.0009580
5: 0.0029643, 0.0043746, 0.0028746, 0.0043635, -0.0009066, 0.0009879
6: -0.0105389, -0.0049432, -0.0108946, -0.0049873, -0.0035969, 0.0039195
7: 0.0041755, 0.0117963, 0.0042356, 0.0122808, -0.0053380, 0.0048987
8: 0.9921552, 0.9975234, 0.9921975, 0.9978647, -0.0037602, 0.0034508
9: -0.0136392, -0.0087663, -0.0139490, -0.0088047, -0.0031324, 0.0034133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0020293
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0022110
time: 0.61 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0064493, 0.0088852, 0.0063785, 0.0087654, -0.0014752, 0.0016828
1: 0.0022540, 0.0026060, 0.0022438, 0.0025887, -0.0002131, 0.0002431
2: 0.0094475, 0.0107942, 0.0095137, 0.0108334, -0.0009304, 0.0008156
3: -0.0049094, -0.0035166, -0.0048410, -0.0034761, -0.0009622, 0.0008436
4: -0.0002301, 0.0012778, -0.0002739, 0.0012037, -0.0009132, 0.0010417
5: 0.0029041, 0.0043311, 0.0029743, 0.0043725, -0.0009858, 0.0008642
6: -0.0107776, -0.0051160, -0.0104993, -0.0049514, -0.0039113, 0.0034289
7: 0.0044108, 0.0121215, 0.0041866, 0.0117424, -0.0046698, 0.0053269
8: 0.9923210, 0.9977525, 0.9921631, 0.9974855, -0.0032895, 0.0037524
9: -0.0138472, -0.0089167, -0.0136047, -0.0087734, -0.0034061, 0.0029860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0064493, 0.0088852, 0.0063924, 0.0089446, -0.0015300, 0.0015318
1: 0.0022540, 0.0026060, 0.0022458, 0.0026145, -0.0002210, 0.0002213
2: 0.0094475, 0.0107942, 0.0094146, 0.0108257, -0.0008469, 0.0008459
3: -0.0049094, -0.0035166, -0.0049434, -0.0034840, -0.0008759, 0.0008749
4: -0.0002301, 0.0012778, -0.0002653, 0.0013146, -0.0009471, 0.0009482
5: 0.0029041, 0.0043311, 0.0028693, 0.0043644, -0.0008973, 0.0008963
6: -0.0107776, -0.0051160, -0.0109156, -0.0049837, -0.0035603, 0.0035561
7: 0.0044108, 0.0121215, 0.0042307, 0.0123094, -0.0048431, 0.0048489
8: 0.9923210, 0.9977525, 0.9921940, 0.9978850, -0.0034116, 0.0034157
9: -0.0138472, -0.0089167, -0.0139673, -0.0088016, -0.0031005, 0.0030968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063940, 0.0089355, 0.0063750, 0.0087825, -0.0015475, 0.0016863
1: 0.0022460, 0.0026132, 0.0022433, 0.0025911, -0.0002236, 0.0002436
2: 0.0094197, 0.0108248, 0.0095043, 0.0108353, -0.0009323, 0.0008556
3: -0.0049382, -0.0034849, -0.0048507, -0.0034741, -0.0009643, 0.0008849
4: -0.0002643, 0.0013089, -0.0002761, 0.0012142, -0.0009580, 0.0010439
5: 0.0028746, 0.0043635, 0.0029643, 0.0043746, -0.0009879, 0.0009066
6: -0.0108946, -0.0049873, -0.0105389, -0.0049432, -0.0039195, 0.0035969
7: 0.0042356, 0.0122808, 0.0041755, 0.0117963, -0.0048987, 0.0053380
8: 0.9921975, 0.9978647, 0.9921552, 0.9975234, -0.0034508, 0.0037602
9: -0.0139490, -0.0088047, -0.0136392, -0.0087663, -0.0034133, 0.0031324

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
time: 0.80 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063940, 0.0089355, 0.0063885, 0.0089608, -0.0016055, 0.0015345
1: 0.0022460, 0.0026132, 0.0022453, 0.0026169, -0.0002320, 0.0002217
2: 0.0094197, 0.0108248, 0.0094057, 0.0108278, -0.0008484, 0.0008877
3: -0.0049382, -0.0034849, -0.0049527, -0.0034818, -0.0008775, 0.0009181
4: -0.0002643, 0.0013089, -0.0002677, 0.0013246, -0.0009938, 0.0009499
5: 0.0028746, 0.0043635, 0.0028598, 0.0043667, -0.0008989, 0.0009405
6: -0.0108946, -0.0049873, -0.0109535, -0.0049747, -0.0035666, 0.0037317
7: 0.0042356, 0.0122808, 0.0042184, 0.0123609, -0.0050822, 0.0048575
8: 0.9921975, 0.9978647, 0.9921855, 0.9979211, -0.0035800, 0.0034217
9: -0.0139490, -0.0088047, -0.0140003, -0.0087937, -0.0031060, 0.0032497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0063785, 0.0087654, 0.0062932, 0.0086166, -0.0014511, 0.0017009
1: 0.0022438, 0.0025887, 0.0022315, 0.0025672, -0.0002096, 0.0002457
2: 0.0095137, 0.0108334, 0.0095960, 0.0108805, -0.0009404, 0.0008023
3: -0.0048410, -0.0034761, -0.0047559, -0.0034273, -0.0009726, 0.0008298
4: -0.0002739, 0.0012037, -0.0003267, 0.0011115, -0.0008983, 0.0010529
5: 0.0029743, 0.0043725, 0.0030615, 0.0044225, -0.0009964, 0.0008501
6: -0.0104993, -0.0049514, -0.0101534, -0.0047530, -0.0039534, 0.0033729
7: 0.0041866, 0.0117424, 0.0039165, 0.0112713, -0.0045935, 0.0053842
8: 0.9921631, 0.9974855, 0.9919728, 0.9971536, -0.0032358, 0.0037927
9: -0.0136047, -0.0087734, -0.0133035, -0.0086007, -0.0034428, 0.0029372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0020179
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0021619
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0063750, 0.0087825, 0.0062349, 0.0086704, -0.0014549, 0.0017524
1: 0.0022433, 0.0025911, 0.0022231, 0.0025749, -0.0002102, 0.0002532
2: 0.0095043, 0.0108353, 0.0095662, 0.0109127, -0.0009689, 0.0008044
3: -0.0048507, -0.0034741, -0.0047866, -0.0033940, -0.0010021, 0.0008319
4: -0.0002761, 0.0012142, -0.0003628, 0.0011448, -0.0009006, 0.0010848
5: 0.0029643, 0.0043746, 0.0030299, 0.0044566, -0.0010266, 0.0008523
6: -0.0105389, -0.0049432, -0.0102784, -0.0046177, -0.0040731, 0.0033816
7: 0.0041755, 0.0117963, 0.0037322, 0.0114416, -0.0046055, 0.0055473
8: 0.9921552, 0.9975234, 0.9918429, 0.9972736, -0.0032442, 0.0039076
9: -0.0136392, -0.0087663, -0.0134124, -0.0084828, -0.0035471, 0.0029449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021577, upper bound: 0.0020179
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021577, upper bound: 0.0021619
time: 0.70 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0063785, 0.0087654, 0.0063083, 0.0087857, -0.0016464, 0.0016938
1: 0.0022438, 0.0025887, 0.0022337, 0.0025916, -0.0002379, 0.0002447
2: 0.0095137, 0.0108334, 0.0095025, 0.0108722, -0.0009365, 0.0009103
3: -0.0048410, -0.0034761, -0.0048525, -0.0034360, -0.0009685, 0.0009414
4: -0.0002739, 0.0012037, -0.0003173, 0.0012162, -0.0010192, 0.0010485
5: 0.0029743, 0.0043725, 0.0029624, 0.0044136, -0.0009922, 0.0009645
6: -0.0104993, -0.0049514, -0.0105463, -0.0047883, -0.0039369, 0.0038267
7: 0.0041866, 0.0117424, 0.0039645, 0.0118064, -0.0052117, 0.0053617
8: 0.9921631, 0.9974855, 0.9920066, 0.9975305, -0.0036712, 0.0037769
9: -0.0136047, -0.0087734, -0.0136457, -0.0086314, -0.0034284, 0.0033325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0019662
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021434
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0063750, 0.0087825, 0.0062482, 0.0088389, -0.0016501, 0.0017510
1: 0.0022433, 0.0025911, 0.0022250, 0.0025993, -0.0002384, 0.0002530
2: 0.0095043, 0.0108353, 0.0094731, 0.0109054, -0.0009681, 0.0009123
3: -0.0048507, -0.0034741, -0.0048830, -0.0034016, -0.0010013, 0.0009435
4: -0.0002761, 0.0012142, -0.0003545, 0.0012491, -0.0010214, 0.0010839
5: 0.0029643, 0.0043746, 0.0029312, 0.0044488, -0.0010258, 0.0009666
6: -0.0105389, -0.0049432, -0.0106700, -0.0046486, -0.0040699, 0.0038352
7: 0.0041755, 0.0117963, 0.0037743, 0.0119750, -0.0052232, 0.0055428
8: 0.9921552, 0.9975234, 0.9918726, 0.9976493, -0.0036793, 0.0039045
9: -0.0136392, -0.0087663, -0.0137535, -0.0085097, -0.0035442, 0.0033399

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0019662
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021434
time: 0.60 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0063885, 0.0089608, 0.0062349, 0.0086704, -0.0014683, 0.0019680
1: 0.0022453, 0.0026169, 0.0022231, 0.0025749, -0.0002121, 0.0002843
2: 0.0094057, 0.0108278, 0.0095662, 0.0109127, -0.0010880, 0.0008118
3: -0.0049527, -0.0034818, -0.0047866, -0.0033940, -0.0011253, 0.0008396
4: -0.0002677, 0.0013246, -0.0003628, 0.0011448, -0.0009089, 0.0012182
5: 0.0028598, 0.0043667, 0.0030299, 0.0044566, -0.0011528, 0.0008601
6: -0.0109535, -0.0049747, -0.0102784, -0.0046177, -0.0045741, 0.0034128
7: 0.0042184, 0.0123609, 0.0037322, 0.0114416, -0.0046479, 0.0062295
8: 0.9921855, 0.9979211, 0.9918429, 0.9972736, -0.0032741, 0.0043882
9: -0.0140003, -0.0087937, -0.0134124, -0.0084828, -0.0039833, 0.0029720

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0019032
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
time: 0.77 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0063885, 0.0089608, 0.0062482, 0.0088389, -0.0015200, 0.0018250
1: 0.0022453, 0.0026169, 0.0022250, 0.0025993, -0.0002196, 0.0002637
2: 0.0094057, 0.0108278, 0.0094731, 0.0109054, -0.0010090, 0.0008404
3: -0.0049527, -0.0034818, -0.0048830, -0.0034016, -0.0010435, 0.0008691
4: -0.0002677, 0.0013246, -0.0003545, 0.0012491, -0.0009409, 0.0011297
5: 0.0028598, 0.0043667, 0.0029312, 0.0044488, -0.0010691, 0.0008904
6: -0.0109535, -0.0049747, -0.0106700, -0.0046486, -0.0042417, 0.0035329
7: 0.0042184, 0.0123609, 0.0037743, 0.0119750, -0.0048115, 0.0057769
8: 0.9921855, 0.9979211, 0.9918726, 0.9976493, -0.0033893, 0.0040694
9: -0.0140003, -0.0087937, -0.0137535, -0.0085097, -0.0036939, 0.0030766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0019032
time: 0.79 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0064408, 0.0087028, 0.0060771, 0.0085830, -0.0015225, 0.0019697
1: 0.0022528, 0.0025796, 0.0022003, 0.0025623, -0.0002200, 0.0002846
2: 0.0095483, 0.0107989, 0.0096145, 0.0110000, -0.0010890, 0.0008417
3: -0.0048052, -0.0035117, -0.0047367, -0.0033037, -0.0011263, 0.0008706
4: -0.0002353, 0.0011649, -0.0004605, 0.0010908, -0.0009424, 0.0012193
5: 0.0030109, 0.0043360, 0.0030811, 0.0045491, -0.0011538, 0.0008919
6: -0.0103538, -0.0050962, -0.0100753, -0.0042508, -0.0045781, 0.0035387
7: 0.0043839, 0.0115443, 0.0032325, 0.0111650, -0.0048194, 0.0062350
8: 0.9923019, 0.9973459, 0.9914909, 0.9970787, -0.0033949, 0.0043921
9: -0.0134781, -0.0088995, -0.0132356, -0.0081633, -0.0039868, 0.0030816

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021255
time: 0.72 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021255
time: 0.66 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0063796, 0.0087597, 0.0060739, 0.0086003, -0.0015875, 0.0019996
1: 0.0022440, 0.0025878, 0.0021998, 0.0025648, -0.0002293, 0.0002889
2: 0.0095168, 0.0108327, 0.0096050, 0.0110017, -0.0011055, 0.0008777
3: -0.0048377, -0.0034767, -0.0047465, -0.0033019, -0.0011434, 0.0009077
4: -0.0002732, 0.0012001, -0.0004624, 0.0011015, -0.0009827, 0.0012378
5: 0.0029776, 0.0043719, 0.0030710, 0.0045510, -0.0011714, 0.0009299
6: -0.0104860, -0.0049540, -0.0101155, -0.0042435, -0.0046477, 0.0036897
7: 0.0041902, 0.0117243, 0.0032225, 0.0112197, -0.0050250, 0.0063298
8: 0.9921655, 0.9974728, 0.9914839, 0.9971172, -0.0035397, 0.0044588
9: -0.0135932, -0.0087757, -0.0132705, -0.0081569, -0.0040474, 0.0032131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0022905
time: 0.63 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0022905
time: 0.64 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0064493, 0.0088852, 0.0060771, 0.0085830, -0.0015320, 0.0021903
1: 0.0022540, 0.0026060, 0.0022003, 0.0025623, -0.0002213, 0.0003164
2: 0.0094475, 0.0107942, 0.0096145, 0.0110000, -0.0012110, 0.0008470
3: -0.0049094, -0.0035166, -0.0047367, -0.0033037, -0.0012524, 0.0008760
4: -0.0002301, 0.0012778, -0.0004605, 0.0010908, -0.0009483, 0.0013558
5: 0.0029041, 0.0043311, 0.0030811, 0.0045491, -0.0012831, 0.0008974
6: -0.0107776, -0.0051160, -0.0100753, -0.0042508, -0.0050908, 0.0035607
7: 0.0044108, 0.0121215, 0.0032325, 0.0111650, -0.0048494, 0.0069333
8: 0.9923210, 0.9977525, 0.9914909, 0.9970787, -0.0034160, 0.0048839
9: -0.0138472, -0.0089167, -0.0132356, -0.0081633, -0.0044333, 0.0031008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0020703
time: 0.61 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0020703
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0063940, 0.0089355, 0.0060739, 0.0086003, -0.0015985, 0.0022168
1: 0.0022460, 0.0026132, 0.0021998, 0.0025648, -0.0002309, 0.0003203
2: 0.0094197, 0.0108248, 0.0096050, 0.0110017, -0.0012256, 0.0008838
3: -0.0049382, -0.0034849, -0.0047465, -0.0033019, -0.0012676, 0.0009140
4: -0.0002643, 0.0013089, -0.0004624, 0.0011015, -0.0009895, 0.0013723
5: 0.0028746, 0.0043635, 0.0030710, 0.0045510, -0.0012986, 0.0009364
6: -0.0108946, -0.0049873, -0.0101155, -0.0042435, -0.0051525, 0.0037153
7: 0.0042356, 0.0122808, 0.0032225, 0.0112197, -0.0050599, 0.0070173
8: 0.9921975, 0.9978647, 0.9914839, 0.9971172, -0.0035643, 0.0049431
9: -0.0139490, -0.0088047, -0.0132705, -0.0081569, -0.0044871, 0.0032354

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021980
time: 0.60 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021982
time: 0.72 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0064408, 0.0087028, 0.0059402, 0.0084922, -0.0014895, 0.0021741
1: 0.0022528, 0.0025796, 0.0021805, 0.0025492, -0.0002152, 0.0003141
2: 0.0095483, 0.0107989, 0.0096647, 0.0110757, -0.0012020, 0.0008235
3: -0.0048052, -0.0035117, -0.0046848, -0.0032255, -0.0012432, 0.0008517
4: -0.0002353, 0.0011649, -0.0005452, 0.0010346, -0.0009220, 0.0013458
5: 0.0030109, 0.0043360, 0.0031343, 0.0046293, -0.0012736, 0.0008726
6: -0.0103538, -0.0050962, -0.0098643, -0.0039327, -0.0050531, 0.0034621
7: 0.0043839, 0.0115443, 0.0027993, 0.0108776, -0.0047151, 0.0068819
8: 0.9923019, 0.9973459, 0.9911858, 0.9968763, -0.0033214, 0.0048478
9: -0.0134781, -0.0088995, -0.0130518, -0.0078863, -0.0044005, 0.0030149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0020996
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0020996
time: 0.69 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0063796, 0.0087597, 0.0059367, 0.0085085, -0.0015540, 0.0022087
1: 0.0022440, 0.0025878, 0.0021800, 0.0025515, -0.0002245, 0.0003191
2: 0.0095168, 0.0108327, 0.0096557, 0.0110776, -0.0012211, 0.0008592
3: -0.0048377, -0.0034767, -0.0046940, -0.0032235, -0.0012630, 0.0008886
4: -0.0002732, 0.0012001, -0.0005473, 0.0010446, -0.0009620, 0.0013672
5: 0.0029776, 0.0043719, 0.0031248, 0.0046313, -0.0012939, 0.0009103
6: -0.0104860, -0.0049540, -0.0099021, -0.0039246, -0.0051337, 0.0036119
7: 0.0041902, 0.0117243, 0.0027883, 0.0109291, -0.0049191, 0.0069916
8: 0.9921655, 0.9974728, 0.9911780, 0.9969126, -0.0034651, 0.0049251
9: -0.0135932, -0.0087757, -0.0130847, -0.0078793, -0.0044706, 0.0031454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0022642
time: 0.73 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0022642
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0064493, 0.0088852, 0.0059402, 0.0084922, -0.0014990, 0.0023947
1: 0.0022540, 0.0026060, 0.0021805, 0.0025492, -0.0002166, 0.0003460
2: 0.0094475, 0.0107942, 0.0096647, 0.0110757, -0.0013239, 0.0008288
3: -0.0049094, -0.0035166, -0.0046848, -0.0032255, -0.0013693, 0.0008571
4: -0.0002301, 0.0012778, -0.0005452, 0.0010346, -0.0009279, 0.0014823
5: 0.0029041, 0.0043311, 0.0031343, 0.0046293, -0.0014028, 0.0008781
6: -0.0107776, -0.0051160, -0.0098643, -0.0039327, -0.0055658, 0.0034841
7: 0.0044108, 0.0121215, 0.0027993, 0.0108776, -0.0047450, 0.0075802
8: 0.9923210, 0.9977525, 0.9911858, 0.9968763, -0.0033425, 0.0053396
9: -0.0138472, -0.0089167, -0.0130518, -0.0078863, -0.0048470, 0.0030341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0020229
time: 0.64 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0020229
time: 0.73 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0063940, 0.0089355, 0.0059367, 0.0085085, -0.0015650, 0.0024259
1: 0.0022460, 0.0026132, 0.0021800, 0.0025515, -0.0002261, 0.0003505
2: 0.0094197, 0.0108248, 0.0096557, 0.0110776, -0.0013412, 0.0008653
3: -0.0049382, -0.0034849, -0.0046940, -0.0032235, -0.0013872, 0.0008949
4: -0.0002643, 0.0013089, -0.0005473, 0.0010446, -0.0009688, 0.0015017
5: 0.0028746, 0.0043635, 0.0031248, 0.0046313, -0.0014211, 0.0009168
6: -0.0108946, -0.0049873, -0.0099021, -0.0039246, -0.0056385, 0.0036375
7: 0.0042356, 0.0122808, 0.0027883, 0.0109291, -0.0049540, 0.0076792
8: 0.9921975, 0.9978647, 0.9911780, 0.9969126, -0.0034897, 0.0054094
9: -0.0139490, -0.0088047, -0.0130847, -0.0078793, -0.0049103, 0.0031677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0021559
time: 0.62 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0021558
time: 0.74 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0064408, 0.0087028, 0.0061001, 0.0087421, -0.0016920, 0.0019504
1: 0.0022528, 0.0025796, 0.0022036, 0.0025853, -0.0002444, 0.0002818
2: 0.0095483, 0.0107989, 0.0095266, 0.0109873, -0.0010783, 0.0009354
3: -0.0048052, -0.0035117, -0.0048277, -0.0033169, -0.0011152, 0.0009675
4: -0.0002353, 0.0011649, -0.0004462, 0.0011893, -0.0010474, 0.0012073
5: 0.0030109, 0.0043360, 0.0029879, 0.0045356, -0.0011425, 0.0009912
6: -0.0103538, -0.0050962, -0.0104452, -0.0043043, -0.0045332, 0.0039326
7: 0.0043839, 0.0115443, 0.0033054, 0.0116687, -0.0053559, 0.0061738
8: 0.9923019, 0.9973459, 0.9915423, 0.9974335, -0.0037728, 0.0043489
9: -0.0134781, -0.0088995, -0.0135576, -0.0082099, -0.0039477, 0.0034247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0021347
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0021347
time: 0.75 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0064493, 0.0088852, 0.0061001, 0.0087421, -0.0015828, 0.0020410
1: 0.0022540, 0.0026060, 0.0022036, 0.0025853, -0.0002287, 0.0002949
2: 0.0094475, 0.0107942, 0.0095266, 0.0109873, -0.0011284, 0.0008751
3: -0.0049094, -0.0035166, -0.0048277, -0.0033169, -0.0011670, 0.0009051
4: -0.0002301, 0.0012778, -0.0004462, 0.0011893, -0.0009798, 0.0012634
5: 0.0029041, 0.0043311, 0.0029879, 0.0045356, -0.0011956, 0.0009272
6: -0.0107776, -0.0051160, -0.0104452, -0.0043043, -0.0047438, 0.0036789
7: 0.0044108, 0.0121215, 0.0033054, 0.0116687, -0.0050103, 0.0064606
8: 0.9923210, 0.9977525, 0.9915423, 0.9974335, -0.0035294, 0.0045510
9: -0.0138472, -0.0089167, -0.0135576, -0.0082099, -0.0041311, 0.0032037

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0020703
time: 0.78 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0020703
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0063796, 0.0087597, 0.0060969, 0.0087605, -0.0017588, 0.0019748
1: 0.0022440, 0.0025878, 0.0022031, 0.0025879, -0.0002541, 0.0002853
2: 0.0095168, 0.0108327, 0.0095164, 0.0109890, -0.0010918, 0.0009724
3: -0.0048377, -0.0034767, -0.0048382, -0.0033151, -0.0011292, 0.0010057
4: -0.0002732, 0.0012001, -0.0004482, 0.0012006, -0.0010887, 0.0012224
5: 0.0029776, 0.0043719, 0.0029771, 0.0045375, -0.0011568, 0.0010303
6: -0.0104860, -0.0049540, -0.0104879, -0.0042970, -0.0045899, 0.0040878
7: 0.0041902, 0.0117243, 0.0032954, 0.0117269, -0.0055673, 0.0062510
8: 0.9921655, 0.9974728, 0.9915352, 0.9974745, -0.0039217, 0.0044034
9: -0.0135932, -0.0087757, -0.0135948, -0.0082035, -0.0039971, 0.0035599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0023245
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0023245
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0063940, 0.0089355, 0.0060969, 0.0087605, -0.0016523, 0.0020684
1: 0.0022460, 0.0026132, 0.0022031, 0.0025879, -0.0002387, 0.0002988
2: 0.0094197, 0.0108248, 0.0095164, 0.0109890, -0.0011436, 0.0009135
3: -0.0049382, -0.0034849, -0.0048382, -0.0033151, -0.0011827, 0.0009448
4: -0.0002643, 0.0013089, -0.0004482, 0.0012006, -0.0010228, 0.0012804
5: 0.0028746, 0.0043635, 0.0029771, 0.0045375, -0.0012117, 0.0009679
6: -0.0108946, -0.0049873, -0.0104879, -0.0042970, -0.0048075, 0.0038403
7: 0.0042356, 0.0122808, 0.0032954, 0.0117269, -0.0052302, 0.0065474
8: 0.9921975, 0.9978647, 0.9915352, 0.9974745, -0.0036842, 0.0046121
9: -0.0139490, -0.0088047, -0.0135948, -0.0082035, -0.0041866, 0.0033443

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0021980
time: 0.69 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0021982
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0064408, 0.0087028, 0.0059528, 0.0086626, -0.0016524, 0.0021416
1: 0.0022528, 0.0025796, 0.0021823, 0.0025738, -0.0002387, 0.0003094
2: 0.0095483, 0.0107989, 0.0095705, 0.0110687, -0.0011840, 0.0009136
3: -0.0048052, -0.0035117, -0.0047822, -0.0032327, -0.0012246, 0.0009449
4: -0.0002353, 0.0011649, -0.0005374, 0.0011400, -0.0010229, 0.0013257
5: 0.0030109, 0.0043360, 0.0030345, 0.0046219, -0.0012545, 0.0009680
6: -0.0103538, -0.0050962, -0.0102602, -0.0039619, -0.0049776, 0.0038406
7: 0.0043839, 0.0115443, 0.0028391, 0.0114168, -0.0052306, 0.0067791
8: 0.9923019, 0.9973459, 0.9912138, 0.9972562, -0.0036845, 0.0047754
9: -0.0134781, -0.0088995, -0.0133966, -0.0079117, -0.0043348, 0.0033446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019378, upper bound: 0.0020835
time: 0.66 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019378, upper bound: 0.0020835
time: 0.67 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0063796, 0.0087597, 0.0059493, 0.0086791, -0.0017182, 0.0021747
1: 0.0022440, 0.0025878, 0.0021818, 0.0025762, -0.0002482, 0.0003142
2: 0.0095168, 0.0108327, 0.0095614, 0.0110706, -0.0012023, 0.0009499
3: -0.0048377, -0.0034767, -0.0047916, -0.0032307, -0.0012435, 0.0009825
4: -0.0002732, 0.0012001, -0.0005396, 0.0011502, -0.0010636, 0.0013462
5: 0.0029776, 0.0043719, 0.0030249, 0.0046240, -0.0012739, 0.0010065
6: -0.0104860, -0.0049540, -0.0102985, -0.0039538, -0.0050545, 0.0039935
7: 0.0041902, 0.0117243, 0.0028280, 0.0114690, -0.0054388, 0.0068839
8: 0.9921655, 0.9974728, 0.9912060, 0.9972929, -0.0038312, 0.0048491
9: -0.0135932, -0.0087757, -0.0134299, -0.0079047, -0.0044017, 0.0034777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0022504
time: 0.67 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0022504
time: 0.76 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0063940, 0.0089355, 0.0059493, 0.0086791, -0.0016193, 0.0022773
1: 0.0022460, 0.0026132, 0.0021818, 0.0025762, -0.0002339, 0.0003290
2: 0.0094197, 0.0108248, 0.0095614, 0.0110706, -0.0012590, 0.0008952
3: -0.0049382, -0.0034849, -0.0047916, -0.0032307, -0.0013022, 0.0009259
4: -0.0002643, 0.0013089, -0.0005396, 0.0011502, -0.0010023, 0.0014097
5: 0.0028746, 0.0043635, 0.0030249, 0.0046240, -0.0013340, 0.0009486
6: -0.0108946, -0.0049873, -0.0102985, -0.0039538, -0.0052930, 0.0037636
7: 0.0042356, 0.0122808, 0.0028280, 0.0114690, -0.0051257, 0.0072086
8: 0.9921975, 0.9978647, 0.9912060, 0.9972929, -0.0036106, 0.0050779
9: -0.0139490, -0.0088047, -0.0134299, -0.0079047, -0.0046093, 0.0032775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021558
time: 0.74 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021559
time: 0.76 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0063785, 0.0087654, -0.0017009, 0.0014511
1: 0.0022315, 0.0025672, 0.0022438, 0.0025887, -0.0002457, 0.0002096
2: 0.0095960, 0.0108805, 0.0095137, 0.0108334, -0.0008023, 0.0009404
3: -0.0047559, -0.0034273, -0.0048410, -0.0034761, -0.0008298, 0.0009726
4: -0.0003267, 0.0011115, -0.0002739, 0.0012037, -0.0010529, 0.0008983
5: 0.0030615, 0.0044225, 0.0029743, 0.0043725, -0.0008501, 0.0009964
6: -0.0101534, -0.0047530, -0.0104993, -0.0049514, -0.0033729, 0.0039534
7: 0.0039165, 0.0112713, 0.0041866, 0.0117424, -0.0053842, 0.0045935
8: 0.9919728, 0.9971536, 0.9921631, 0.9974855, -0.0037927, 0.0032358
9: -0.0133035, -0.0086007, -0.0136047, -0.0087734, -0.0029372, 0.0034428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0062337, 0.0086784, -0.0015065, 0.0015007
1: 0.0022315, 0.0025672, 0.0022229, 0.0025761, -0.0002176, 0.0002168
2: 0.0095960, 0.0108805, 0.0095618, 0.0109134, -0.0008297, 0.0008329
3: -0.0047559, -0.0034273, -0.0047912, -0.0033933, -0.0008581, 0.0008614
4: -0.0003267, 0.0011115, -0.0003635, 0.0011498, -0.0009325, 0.0009290
5: 0.0030615, 0.0044225, 0.0030252, 0.0044574, -0.0008791, 0.0008825
6: -0.0101534, -0.0047530, -0.0102970, -0.0046148, -0.0034880, 0.0035015
7: 0.0039165, 0.0112713, 0.0037283, 0.0114670, -0.0047687, 0.0047504
8: 0.9919728, 0.9971536, 0.9918402, 0.9972915, -0.0033592, 0.0033463
9: -0.0133035, -0.0086007, -0.0134286, -0.0084803, -0.0030375, 0.0030492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
time: 0.72 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0063750, 0.0087825, -0.0017524, 0.0014549
1: 0.0022231, 0.0025749, 0.0022433, 0.0025911, -0.0002532, 0.0002102
2: 0.0095662, 0.0109127, 0.0095043, 0.0108353, -0.0008044, 0.0009689
3: -0.0047866, -0.0033940, -0.0048507, -0.0034741, -0.0008319, 0.0010021
4: -0.0003628, 0.0011448, -0.0002761, 0.0012142, -0.0010848, 0.0009006
5: 0.0030299, 0.0044566, 0.0029643, 0.0043746, -0.0008523, 0.0010266
6: -0.0102784, -0.0046177, -0.0105389, -0.0049432, -0.0033816, 0.0040731
7: 0.0037322, 0.0114416, 0.0041755, 0.0117963, -0.0055473, 0.0046055
8: 0.9918429, 0.9972736, 0.9921552, 0.9975234, -0.0039076, 0.0032442
9: -0.0134124, -0.0084828, -0.0136392, -0.0087663, -0.0029449, 0.0035471

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
time: 0.65 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
time: 0.60 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0062301, 0.0086951, -0.0015741, 0.0015074
1: 0.0022231, 0.0025749, 0.0022224, 0.0025785, -0.0002274, 0.0002178
2: 0.0095662, 0.0109127, 0.0095525, 0.0109154, -0.0008334, 0.0008703
3: -0.0047866, -0.0033940, -0.0048008, -0.0033912, -0.0008620, 0.0009001
4: -0.0003628, 0.0011448, -0.0003658, 0.0011602, -0.0009744, 0.0009331
5: 0.0030299, 0.0044566, 0.0030154, 0.0044595, -0.0008830, 0.0009221
6: -0.0102784, -0.0046177, -0.0103359, -0.0046064, -0.0035037, 0.0036587
7: 0.0037322, 0.0114416, 0.0037168, 0.0115199, -0.0049828, 0.0047717
8: 0.9918429, 0.9972736, 0.9918320, 0.9973287, -0.0035100, 0.0033613
9: -0.0134124, -0.0084828, -0.0134625, -0.0084730, -0.0030511, 0.0031862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063083, 0.0087857, 0.0063785, 0.0087654, -0.0016938, 0.0016464
1: 0.0022337, 0.0025916, 0.0022438, 0.0025887, -0.0002447, 0.0002379
2: 0.0095025, 0.0108722, 0.0095137, 0.0108334, -0.0009103, 0.0009365
3: -0.0048525, -0.0034360, -0.0048410, -0.0034761, -0.0009414, 0.0009685
4: -0.0003173, 0.0012162, -0.0002739, 0.0012037, -0.0010485, 0.0010192
5: 0.0029624, 0.0044136, 0.0029743, 0.0043725, -0.0009645, 0.0009922
6: -0.0105463, -0.0047883, -0.0104993, -0.0049514, -0.0038267, 0.0039369
7: 0.0039645, 0.0118064, 0.0041866, 0.0117424, -0.0053617, 0.0052117
8: 0.9920066, 0.9975305, 0.9921631, 0.9974855, -0.0037769, 0.0036712
9: -0.0136457, -0.0086314, -0.0136047, -0.0087734, -0.0033325, 0.0034284

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
time: 0.73 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063083, 0.0087857, 0.0062337, 0.0086784, -0.0015145, 0.0017224
1: 0.0022337, 0.0025916, 0.0022229, 0.0025761, -0.0002188, 0.0002488
2: 0.0095025, 0.0108722, 0.0095618, 0.0109134, -0.0009523, 0.0008373
3: -0.0048525, -0.0034360, -0.0047912, -0.0033933, -0.0009849, 0.0008660
4: -0.0003173, 0.0012162, -0.0003635, 0.0011498, -0.0009375, 0.0010662
5: 0.0029624, 0.0044136, 0.0030252, 0.0044574, -0.0010090, 0.0008872
6: -0.0105463, -0.0047883, -0.0102970, -0.0046148, -0.0040034, 0.0035201
7: 0.0039645, 0.0118064, 0.0037283, 0.0114670, -0.0047941, 0.0054523
8: 0.9920066, 0.9975305, 0.9918402, 0.9972915, -0.0033771, 0.0038407
9: -0.0136457, -0.0086314, -0.0134286, -0.0084803, -0.0034864, 0.0030655

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
time: 0.73 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0063750, 0.0087825, -0.0017510, 0.0016501
1: 0.0022250, 0.0025993, 0.0022433, 0.0025911, -0.0002530, 0.0002384
2: 0.0094731, 0.0109054, 0.0095043, 0.0108353, -0.0009123, 0.0009681
3: -0.0048830, -0.0034016, -0.0048507, -0.0034741, -0.0009435, 0.0010013
4: -0.0003545, 0.0012491, -0.0002761, 0.0012142, -0.0010839, 0.0010214
5: 0.0029312, 0.0044488, 0.0029643, 0.0043746, -0.0009666, 0.0010258
6: -0.0106700, -0.0046486, -0.0105389, -0.0049432, -0.0038352, 0.0040699
7: 0.0037743, 0.0119750, 0.0041755, 0.0117963, -0.0055428, 0.0052232
8: 0.9918726, 0.9976493, 0.9921552, 0.9975234, -0.0039045, 0.0036793
9: -0.0137535, -0.0085097, -0.0136392, -0.0087663, -0.0033399, 0.0035442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
time: 0.66 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
time: 0.77 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0062301, 0.0086951, -0.0015874, 0.0017263
1: 0.0022250, 0.0025993, 0.0022224, 0.0025785, -0.0002293, 0.0002494
2: 0.0094731, 0.0109054, 0.0095525, 0.0109154, -0.0009544, 0.0008777
3: -0.0048830, -0.0034016, -0.0048008, -0.0033912, -0.0009871, 0.0009077
4: -0.0003545, 0.0012491, -0.0003658, 0.0011602, -0.0009826, 0.0010686
5: 0.0029312, 0.0044488, 0.0030154, 0.0044595, -0.0010112, 0.0009299
6: -0.0106700, -0.0046486, -0.0103359, -0.0046064, -0.0040123, 0.0036896
7: 0.0037743, 0.0119750, 0.0037168, 0.0115199, -0.0050250, 0.0054644
8: 0.9918726, 0.9976493, 0.9918320, 0.9973287, -0.0035397, 0.0038493
9: -0.0137535, -0.0085097, -0.0134625, -0.0084730, -0.0034941, 0.0032131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
time: 0.81 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B1_B1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0063885, 0.0089608, -0.0019680, 0.0014683
1: 0.0022231, 0.0025749, 0.0022453, 0.0026169, -0.0002843, 0.0002121
2: 0.0095662, 0.0109127, 0.0094057, 0.0108278, -0.0008118, 0.0010880
3: -0.0047866, -0.0033940, -0.0049527, -0.0034818, -0.0008396, 0.0011253
4: -0.0003628, 0.0011448, -0.0002677, 0.0013246, -0.0012182, 0.0009089
5: 0.0030299, 0.0044566, 0.0028598, 0.0043667, -0.0008601, 0.0011528
6: -0.0102784, -0.0046177, -0.0109535, -0.0049747, -0.0034128, 0.0045741
7: 0.0037322, 0.0114416, 0.0042184, 0.0123609, -0.0062295, 0.0046479
8: 0.9918429, 0.9972736, 0.9921855, 0.9979211, -0.0043882, 0.0032741
9: -0.0134124, -0.0084828, -0.0140003, -0.0087937, -0.0029720, 0.0039833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0020780
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0020780
time: 0.74 seconds

## BFS IS instance: IS_A1_A2_B1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0062428, 0.0088631, -0.0017908, 0.0015224
1: 0.0022231, 0.0025749, 0.0022242, 0.0026028, -0.0002587, 0.0002199
2: 0.0095662, 0.0109127, 0.0094597, 0.0109084, -0.0008417, 0.0009901
3: -0.0047866, -0.0033940, -0.0048968, -0.0033985, -0.0008705, 0.0010240
4: -0.0003628, 0.0011448, -0.0003579, 0.0012641, -0.0011085, 0.0009424
5: 0.0030299, 0.0044566, 0.0029170, 0.0044520, -0.0008918, 0.0010490
6: -0.0102784, -0.0046177, -0.0107264, -0.0046359, -0.0035385, 0.0041623
7: 0.0037322, 0.0114416, 0.0037570, 0.0120517, -0.0056687, 0.0048192
8: 0.9918429, 0.9972736, 0.9918604, 0.9977034, -0.0039931, 0.0033947
9: -0.0134124, -0.0084828, -0.0138025, -0.0084987, -0.0030815, 0.0036247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0020773
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0020773
time: 0.83 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0060771, 0.0085830, -0.0017576, 0.0019586
1: 0.0022315, 0.0025672, 0.0022003, 0.0025623, -0.0002539, 0.0002830
2: 0.0095960, 0.0108805, 0.0096145, 0.0110000, -0.0010829, 0.0009717
3: -0.0047559, -0.0034273, -0.0047367, -0.0033037, -0.0011200, 0.0010050
4: -0.0003267, 0.0011115, -0.0004605, 0.0010908, -0.0010880, 0.0012124
5: 0.0030615, 0.0044225, 0.0030811, 0.0045491, -0.0011474, 0.0010296
6: -0.0101534, -0.0047530, -0.0100753, -0.0042508, -0.0045524, 0.0040852
7: 0.0039165, 0.0112713, 0.0032325, 0.0111650, -0.0055637, 0.0062000
8: 0.9919728, 0.9971536, 0.9914909, 0.9970787, -0.0039192, 0.0043674
9: -0.0133035, -0.0086007, -0.0132356, -0.0081633, -0.0039644, 0.0035576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0059402, 0.0084922, -0.0015645, 0.0020095
1: 0.0022315, 0.0025672, 0.0021805, 0.0025492, -0.0002260, 0.0002903
2: 0.0095960, 0.0108805, 0.0096647, 0.0110757, -0.0011110, 0.0008650
3: -0.0047559, -0.0034273, -0.0046848, -0.0032255, -0.0011490, 0.0008946
4: -0.0003267, 0.0011115, -0.0005452, 0.0010346, -0.0009685, 0.0012439
5: 0.0030615, 0.0044225, 0.0031343, 0.0046293, -0.0011772, 0.0009165
6: -0.0101534, -0.0047530, -0.0098643, -0.0039327, -0.0046706, 0.0036364
7: 0.0039165, 0.0112713, 0.0027993, 0.0108776, -0.0049524, 0.0063610
8: 0.9919728, 0.9971536, 0.9911858, 0.9968763, -0.0034886, 0.0044808
9: -0.0133035, -0.0086007, -0.0130518, -0.0078863, -0.0040674, 0.0031667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
time: 0.62 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0060739, 0.0086003, -0.0018034, 0.0019854
1: 0.0022231, 0.0025749, 0.0021998, 0.0025648, -0.0002605, 0.0002868
2: 0.0095662, 0.0109127, 0.0096050, 0.0110017, -0.0010977, 0.0009970
3: -0.0047866, -0.0033940, -0.0047465, -0.0033019, -0.0011353, 0.0010312
4: -0.0003628, 0.0011448, -0.0004624, 0.0011015, -0.0011163, 0.0012290
5: 0.0030299, 0.0044566, 0.0030710, 0.0045510, -0.0011631, 0.0010564
6: -0.0102784, -0.0046177, -0.0101155, -0.0042435, -0.0046147, 0.0041915
7: 0.0037322, 0.0114416, 0.0032225, 0.0112197, -0.0057085, 0.0062848
8: 0.9918429, 0.9972736, 0.9914839, 0.9971172, -0.0040212, 0.0044271
9: -0.0134124, -0.0084828, -0.0132705, -0.0081569, -0.0040187, 0.0036501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0059367, 0.0085085, -0.0016265, 0.0020382
1: 0.0022231, 0.0025749, 0.0021800, 0.0025515, -0.0002350, 0.0002945
2: 0.0095662, 0.0109127, 0.0096557, 0.0110776, -0.0011268, 0.0008993
3: -0.0047866, -0.0033940, -0.0046940, -0.0032235, -0.0011654, 0.0009300
4: -0.0003628, 0.0011448, -0.0005473, 0.0010446, -0.0010068, 0.0012617
5: 0.0030299, 0.0044566, 0.0031248, 0.0046313, -0.0011940, 0.0009528
6: -0.0102784, -0.0046177, -0.0099021, -0.0039246, -0.0047373, 0.0037804
7: 0.0037322, 0.0114416, 0.0027883, 0.0109291, -0.0051486, 0.0064517
8: 0.9918429, 0.9972736, 0.9911780, 0.9969126, -0.0036268, 0.0045447
9: -0.0134124, -0.0084828, -0.0130847, -0.0078793, -0.0041254, 0.0032922

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
time: 0.69 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
time: 0.65 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063083, 0.0087857, 0.0060771, 0.0085830, -0.0017505, 0.0021539
1: 0.0022337, 0.0025916, 0.0022003, 0.0025623, -0.0002529, 0.0003112
2: 0.0095025, 0.0108722, 0.0096145, 0.0110000, -0.0011908, 0.0009678
3: -0.0048525, -0.0034360, -0.0047367, -0.0033037, -0.0012316, 0.0010010
4: -0.0003173, 0.0012162, -0.0004605, 0.0010908, -0.0010836, 0.0013333
5: 0.0029624, 0.0044136, 0.0030811, 0.0045491, -0.0012618, 0.0010255
6: -0.0105463, -0.0047883, -0.0100753, -0.0042508, -0.0050063, 0.0040687
7: 0.0039645, 0.0118064, 0.0032325, 0.0111650, -0.0055413, 0.0068181
8: 0.9920066, 0.9975305, 0.9914909, 0.9970787, -0.0039034, 0.0048028
9: -0.0136457, -0.0086314, -0.0132356, -0.0081633, -0.0043597, 0.0035432

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
time: 0.75 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
time: 0.61 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063083, 0.0087857, 0.0059402, 0.0084922, -0.0015725, 0.0022313
1: 0.0022337, 0.0025916, 0.0021805, 0.0025492, -0.0002272, 0.0003224
2: 0.0095025, 0.0108722, 0.0096647, 0.0110757, -0.0012336, 0.0008694
3: -0.0048525, -0.0034360, -0.0046848, -0.0032255, -0.0012759, 0.0008992
4: -0.0003173, 0.0012162, -0.0005452, 0.0010346, -0.0009734, 0.0013812
5: 0.0029624, 0.0044136, 0.0031343, 0.0046293, -0.0013071, 0.0009212
6: -0.0105463, -0.0047883, -0.0098643, -0.0039327, -0.0051861, 0.0036550
7: 0.0039645, 0.0118064, 0.0027993, 0.0108776, -0.0049778, 0.0070630
8: 0.9920066, 0.9975305, 0.9911858, 0.9968763, -0.0035065, 0.0049753
9: -0.0136457, -0.0086314, -0.0130518, -0.0078863, -0.0045162, 0.0031830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
time: 0.61 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
time: 0.79 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0060739, 0.0086003, -0.0018020, 0.0021806
1: 0.0022250, 0.0025993, 0.0021998, 0.0025648, -0.0002603, 0.0003150
2: 0.0094731, 0.0109054, 0.0096050, 0.0110017, -0.0012056, 0.0009963
3: -0.0048830, -0.0034016, -0.0047465, -0.0033019, -0.0012469, 0.0010304
4: -0.0003545, 0.0012491, -0.0004624, 0.0011015, -0.0011154, 0.0013498
5: 0.0029312, 0.0044488, 0.0030710, 0.0045510, -0.0012774, 0.0010556
6: -0.0106700, -0.0046486, -0.0101155, -0.0042435, -0.0050682, 0.0041882
7: 0.0037743, 0.0119750, 0.0032225, 0.0112197, -0.0057040, 0.0069025
8: 0.9918726, 0.9976493, 0.9914839, 0.9971172, -0.0040180, 0.0048622
9: -0.0137535, -0.0085097, -0.0132705, -0.0081569, -0.0044136, 0.0036473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
time: 0.67 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0059367, 0.0085085, -0.0016398, 0.0022570
1: 0.0022250, 0.0025993, 0.0021800, 0.0025515, -0.0002369, 0.0003261
2: 0.0094731, 0.0109054, 0.0096557, 0.0110776, -0.0012478, 0.0009066
3: -0.0048830, -0.0034016, -0.0046940, -0.0032235, -0.0012906, 0.0009377
4: -0.0003545, 0.0012491, -0.0005473, 0.0010446, -0.0010151, 0.0013971
5: 0.0029312, 0.0044488, 0.0031248, 0.0046313, -0.0013222, 0.0009606
6: -0.0106700, -0.0046486, -0.0099021, -0.0039246, -0.0052459, 0.0038114
7: 0.0037743, 0.0119750, 0.0027883, 0.0109291, -0.0051908, 0.0071445
8: 0.9918726, 0.9976493, 0.9911780, 0.9969126, -0.0036565, 0.0050327
9: -0.0137535, -0.0085097, -0.0130847, -0.0078793, -0.0045684, 0.0033191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
time: 0.74 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
time: 0.82 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0061001, 0.0087421, -0.0019271, 0.0019393
1: 0.0022315, 0.0025672, 0.0022036, 0.0025853, -0.0002784, 0.0002802
2: 0.0095960, 0.0108805, 0.0095266, 0.0109873, -0.0010722, 0.0010655
3: -0.0047559, -0.0034273, -0.0048277, -0.0033169, -0.0011089, 0.0011019
4: -0.0003267, 0.0011115, -0.0004462, 0.0011893, -0.0011929, 0.0012004
5: 0.0030615, 0.0044225, 0.0029879, 0.0045356, -0.0011360, 0.0011289
6: -0.0101534, -0.0047530, -0.0104452, -0.0043043, -0.0045074, 0.0044791
7: 0.0039165, 0.0112713, 0.0033054, 0.0116687, -0.0061002, 0.0061387
8: 0.9919728, 0.9971536, 0.9915423, 0.9974335, -0.0042971, 0.0043242
9: -0.0133035, -0.0086007, -0.0135576, -0.0082099, -0.0039253, 0.0039006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019559, upper bound: 0.0020943
time: 0.62 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019559, upper bound: 0.0020943
time: 0.71 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062932, 0.0086166, 0.0059528, 0.0086626, -0.0017342, 0.0019894
1: 0.0022315, 0.0025672, 0.0021823, 0.0025738, -0.0002505, 0.0002874
2: 0.0095960, 0.0108805, 0.0095705, 0.0110687, -0.0010999, 0.0009588
3: -0.0047559, -0.0034273, -0.0047822, -0.0032327, -0.0011376, 0.0009916
4: -0.0003267, 0.0011115, -0.0005374, 0.0011400, -0.0010735, 0.0012315
5: 0.0030615, 0.0044225, 0.0030345, 0.0046219, -0.0011654, 0.0010159
6: -0.0101534, -0.0047530, -0.0102602, -0.0039619, -0.0046239, 0.0040307
7: 0.0039165, 0.0112713, 0.0028391, 0.0114168, -0.0054894, 0.0062973
8: 0.9919728, 0.9971536, 0.9912138, 0.9972562, -0.0038669, 0.0044360
9: -0.0133035, -0.0086007, -0.0133966, -0.0079117, -0.0040267, 0.0035101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019324, upper bound: 0.0020943
time: 0.78 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019324, upper bound: 0.0020943
time: 0.78 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0060969, 0.0087605, -0.0019747, 0.0019606
1: 0.0022231, 0.0025749, 0.0022031, 0.0025879, -0.0002853, 0.0002832
2: 0.0095662, 0.0109127, 0.0095164, 0.0109890, -0.0010839, 0.0010917
3: -0.0047866, -0.0033940, -0.0048382, -0.0033151, -0.0011211, 0.0011291
4: -0.0003628, 0.0011448, -0.0004482, 0.0012006, -0.0012223, 0.0012136
5: 0.0030299, 0.0044566, 0.0029771, 0.0045375, -0.0011485, 0.0011568
6: -0.0102784, -0.0046177, -0.0104879, -0.0042970, -0.0045569, 0.0045897
7: 0.0037322, 0.0114416, 0.0032954, 0.0117269, -0.0062507, 0.0062061
8: 0.9918429, 0.9972736, 0.9915352, 0.9974745, -0.0044031, 0.0043717
9: -0.0134124, -0.0084828, -0.0135948, -0.0082035, -0.0039683, 0.0039969

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0022341
time: 0.63 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0022341
time: 0.66 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0060969, 0.0087605, -0.0018717, 0.0020539
1: 0.0022250, 0.0025993, 0.0022031, 0.0025879, -0.0002704, 0.0002967
2: 0.0094731, 0.0109054, 0.0095164, 0.0109890, -0.0011355, 0.0010348
3: -0.0048830, -0.0034016, -0.0048382, -0.0033151, -0.0011744, 0.0010703
4: -0.0003545, 0.0012491, -0.0004482, 0.0012006, -0.0011586, 0.0012714
5: 0.0029312, 0.0044488, 0.0029771, 0.0045375, -0.0012032, 0.0010964
6: -0.0106700, -0.0046486, -0.0104879, -0.0042970, -0.0047738, 0.0043504
7: 0.0037743, 0.0119750, 0.0032954, 0.0117269, -0.0059248, 0.0065014
8: 0.9918726, 0.9976493, 0.9915352, 0.9974745, -0.0041736, 0.0045798
9: -0.0137535, -0.0085097, -0.0135948, -0.0082035, -0.0041572, 0.0037885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0021456
time: 0.76 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0021456
time: 0.61 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062349, 0.0086704, 0.0059493, 0.0086791, -0.0017979, 0.0020132
1: 0.0022231, 0.0025749, 0.0021818, 0.0025762, -0.0002597, 0.0002909
2: 0.0095662, 0.0109127, 0.0095614, 0.0110706, -0.0011131, 0.0009940
3: -0.0047866, -0.0033940, -0.0047916, -0.0032307, -0.0011512, 0.0010280
4: -0.0003628, 0.0011448, -0.0005396, 0.0011502, -0.0011129, 0.0012462
5: 0.0030299, 0.0044566, 0.0030249, 0.0046240, -0.0011794, 0.0010532
6: -0.0102784, -0.0046177, -0.0102985, -0.0039538, -0.0046793, 0.0041787
7: 0.0037322, 0.0114416, 0.0028280, 0.0114690, -0.0056910, 0.0063729
8: 0.9918429, 0.9972736, 0.9912060, 0.9972929, -0.0040089, 0.0044892
9: -0.0134124, -0.0084828, -0.0134299, -0.0079047, -0.0040750, 0.0036390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0022317
time: 0.68 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0022317
time: 0.81 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062482, 0.0088389, 0.0059493, 0.0086791, -0.0016890, 0.0021055
1: 0.0022250, 0.0025993, 0.0021818, 0.0025762, -0.0002440, 0.0003042
2: 0.0094731, 0.0109054, 0.0095614, 0.0110706, -0.0011641, 0.0009338
3: -0.0048830, -0.0034016, -0.0047916, -0.0032307, -0.0012040, 0.0009658
4: -0.0003545, 0.0012491, -0.0005396, 0.0011502, -0.0010455, 0.0013034
5: 0.0029312, 0.0044488, 0.0030249, 0.0046240, -0.0012334, 0.0009894
6: -0.0106700, -0.0046486, -0.0102985, -0.0039538, -0.0048938, 0.0039256
7: 0.0037743, 0.0119750, 0.0028280, 0.0114690, -0.0053464, 0.0066649
8: 0.9918726, 0.9976493, 0.9912060, 0.9972929, -0.0037661, 0.0046949
9: -0.0137535, -0.0085097, -0.0134299, -0.0079047, -0.0042617, 0.0034186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0021456
time: 0.79 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_A2_B2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0021456
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0060771, 0.0085830, 0.0064408, 0.0087028, -0.0019697, 0.0015225
1: 0.0022003, 0.0025623, 0.0022528, 0.0025796, -0.0002846, 0.0002200
2: 0.0096145, 0.0110000, 0.0095483, 0.0107989, -0.0008417, 0.0010890
3: -0.0047367, -0.0033037, -0.0048052, -0.0035117, -0.0008706, 0.0011263
4: -0.0004605, 0.0010908, -0.0002353, 0.0011649, -0.0012193, 0.0009424
5: 0.0030811, 0.0045491, 0.0030109, 0.0043360, -0.0008919, 0.0011538
6: -0.0100753, -0.0042508, -0.0103538, -0.0050962, -0.0035387, 0.0045781
7: 0.0032325, 0.0111650, 0.0043839, 0.0115443, -0.0062350, 0.0048194
8: 0.9914909, 0.9970787, 0.9923019, 0.9973459, -0.0043921, 0.0033949
9: -0.0132356, -0.0081633, -0.0134781, -0.0088995, -0.0030816, 0.0039868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0020470
time: 0.80 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0021954
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0060739, 0.0086003, 0.0063796, 0.0087597, -0.0019996, 0.0015875
1: 0.0021998, 0.0025648, 0.0022440, 0.0025878, -0.0002889, 0.0002293
2: 0.0096050, 0.0110017, 0.0095168, 0.0108327, -0.0008777, 0.0011055
3: -0.0047465, -0.0033019, -0.0048377, -0.0034767, -0.0009077, 0.0011434
4: -0.0004624, 0.0011015, -0.0002732, 0.0012001, -0.0012378, 0.0009827
5: 0.0030710, 0.0045510, 0.0029776, 0.0043719, -0.0009299, 0.0011714
6: -0.0101155, -0.0042435, -0.0104860, -0.0049540, -0.0036897, 0.0046477
7: 0.0032225, 0.0112197, 0.0041902, 0.0117243, -0.0063298, 0.0050250
8: 0.9914839, 0.9971172, 0.9921655, 0.9974728, -0.0044588, 0.0035397
9: -0.0132705, -0.0081569, -0.0135932, -0.0087757, -0.0032131, 0.0040474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022905, upper bound: 0.0020470
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022905, upper bound: 0.0021953
time: 0.67 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0060771, 0.0085830, 0.0064493, 0.0088852, -0.0021903, 0.0015320
1: 0.0022003, 0.0025623, 0.0022540, 0.0026060, -0.0003164, 0.0002213
2: 0.0096145, 0.0110000, 0.0094475, 0.0107942, -0.0008470, 0.0012110
3: -0.0047367, -0.0033037, -0.0049094, -0.0035166, -0.0008760, 0.0012524
4: -0.0004605, 0.0010908, -0.0002301, 0.0012778, -0.0013558, 0.0009483
5: 0.0030811, 0.0045491, 0.0029041, 0.0043311, -0.0008974, 0.0012831
6: -0.0100753, -0.0042508, -0.0107776, -0.0051160, -0.0035607, 0.0050908
7: 0.0032325, 0.0111650, 0.0044108, 0.0121215, -0.0069333, 0.0048494
8: 0.9914909, 0.9970787, 0.9923210, 0.9977525, -0.0048839, 0.0034160
9: -0.0132356, -0.0081633, -0.0138472, -0.0089167, -0.0031008, 0.0044333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0020293
time: 0.73 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0022108
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0060739, 0.0086003, 0.0063940, 0.0089355, -0.0022168, 0.0015985
1: 0.0021998, 0.0025648, 0.0022460, 0.0026132, -0.0003203, 0.0002309
2: 0.0096050, 0.0110017, 0.0094197, 0.0108248, -0.0008838, 0.0012256
3: -0.0047465, -0.0033019, -0.0049382, -0.0034849, -0.0009140, 0.0012676
4: -0.0004624, 0.0011015, -0.0002643, 0.0013089, -0.0013723, 0.0009895
5: 0.0030710, 0.0045510, 0.0028746, 0.0043635, -0.0009364, 0.0012986
6: -0.0101155, -0.0042435, -0.0108946, -0.0049873, -0.0037153, 0.0051525
7: 0.0032225, 0.0112197, 0.0042356, 0.0122808, -0.0070173, 0.0050599
8: 0.9914839, 0.9971172, 0.9921975, 0.9978647, -0.0049431, 0.0035643
9: -0.0132705, -0.0081569, -0.0139490, -0.0088047, -0.0032354, 0.0044871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021980, upper bound: 0.0020293
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021980, upper bound: 0.0022110
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0059402, 0.0084922, 0.0064408, 0.0087028, -0.0021741, 0.0014895
1: 0.0021805, 0.0025492, 0.0022528, 0.0025796, -0.0003141, 0.0002152
2: 0.0096647, 0.0110757, 0.0095483, 0.0107989, -0.0008235, 0.0012020
3: -0.0046848, -0.0032255, -0.0048052, -0.0035117, -0.0008517, 0.0012432
4: -0.0005452, 0.0010346, -0.0002353, 0.0011649, -0.0013458, 0.0009220
5: 0.0031343, 0.0046293, 0.0030109, 0.0043360, -0.0008726, 0.0012736
6: -0.0098643, -0.0039327, -0.0103538, -0.0050962, -0.0034621, 0.0050531
7: 0.0027993, 0.0108776, 0.0043839, 0.0115443, -0.0068819, 0.0047151
8: 0.9911858, 0.9968763, 0.9923019, 0.9973459, -0.0048478, 0.0033214
9: -0.0130518, -0.0078863, -0.0134781, -0.0088995, -0.0030149, 0.0044005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020996, upper bound: 0.0020339
time: 0.77 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020996, upper bound: 0.0021577
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0059367, 0.0085085, 0.0063796, 0.0087597, -0.0022087, 0.0015540
1: 0.0021800, 0.0025515, 0.0022440, 0.0025878, -0.0003191, 0.0002245
2: 0.0096557, 0.0110776, 0.0095168, 0.0108327, -0.0008592, 0.0012211
3: -0.0046940, -0.0032235, -0.0048377, -0.0034767, -0.0008886, 0.0012630
4: -0.0005473, 0.0010446, -0.0002732, 0.0012001, -0.0013672, 0.0009620
5: 0.0031248, 0.0046313, 0.0029776, 0.0043719, -0.0009103, 0.0012939
6: -0.0099021, -0.0039246, -0.0104860, -0.0049540, -0.0036119, 0.0051337
7: 0.0027883, 0.0109291, 0.0041902, 0.0117243, -0.0069916, 0.0049191
8: 0.9911780, 0.9969126, 0.9921655, 0.9974728, -0.0049251, 0.0034651
9: -0.0130847, -0.0078793, -0.0135932, -0.0087757, -0.0031454, 0.0044706

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022642, upper bound: 0.0020339
time: 0.76 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022642, upper bound: 0.0021577
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0059402, 0.0084922, 0.0064493, 0.0088852, -0.0023947, 0.0014990
1: 0.0021805, 0.0025492, 0.0022540, 0.0026060, -0.0003460, 0.0002166
2: 0.0096647, 0.0110757, 0.0094475, 0.0107942, -0.0008288, 0.0013239
3: -0.0046848, -0.0032255, -0.0049094, -0.0035166, -0.0008571, 0.0013693
4: -0.0005452, 0.0010346, -0.0002301, 0.0012778, -0.0014823, 0.0009279
5: 0.0031343, 0.0046293, 0.0029041, 0.0043311, -0.0008781, 0.0014028
6: -0.0098643, -0.0039327, -0.0107776, -0.0051160, -0.0034841, 0.0055658
7: 0.0027993, 0.0108776, 0.0044108, 0.0121215, -0.0075802, 0.0047450
8: 0.9911858, 0.9968763, 0.9923210, 0.9977525, -0.0053396, 0.0033425
9: -0.0130518, -0.0078863, -0.0138472, -0.0089167, -0.0030341, 0.0048470

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0019672
time: 0.68 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0021143
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0059367, 0.0085085, 0.0063940, 0.0089355, -0.0024259, 0.0015650
1: 0.0021800, 0.0025515, 0.0022460, 0.0026132, -0.0003505, 0.0002261
2: 0.0096557, 0.0110776, 0.0094197, 0.0108248, -0.0008653, 0.0013412
3: -0.0046940, -0.0032235, -0.0049382, -0.0034849, -0.0008949, 0.0013872
4: -0.0005473, 0.0010446, -0.0002643, 0.0013089, -0.0015017, 0.0009688
5: 0.0031248, 0.0046313, 0.0028746, 0.0043635, -0.0009168, 0.0014211
6: -0.0099021, -0.0039246, -0.0108946, -0.0049873, -0.0036375, 0.0056385
7: 0.0027883, 0.0109291, 0.0042356, 0.0122808, -0.0076792, 0.0049540
8: 0.9911780, 0.9969126, 0.9921975, 0.9978647, -0.0054094, 0.0034897
9: -0.0130847, -0.0078793, -0.0139490, -0.0088047, -0.0031677, 0.0049103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0019672
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0021143
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0061001, 0.0087421, 0.0064408, 0.0087028, -0.0019504, 0.0016920
1: 0.0022036, 0.0025853, 0.0022528, 0.0025796, -0.0002818, 0.0002444
2: 0.0095266, 0.0109873, 0.0095483, 0.0107989, -0.0009354, 0.0010783
3: -0.0048277, -0.0033169, -0.0048052, -0.0035117, -0.0009675, 0.0011152
4: -0.0004462, 0.0011893, -0.0002353, 0.0011649, -0.0012073, 0.0010474
5: 0.0029879, 0.0045356, 0.0030109, 0.0043360, -0.0009912, 0.0011425
6: -0.0104452, -0.0043043, -0.0103538, -0.0050962, -0.0039326, 0.0045332
7: 0.0033054, 0.0116687, 0.0043839, 0.0115443, -0.0061738, 0.0053559
8: 0.9915423, 0.9974335, 0.9923019, 0.9973459, -0.0043489, 0.0037728
9: -0.0135576, -0.0082099, -0.0134781, -0.0088995, -0.0034247, 0.0039477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021346, upper bound: 0.0020145
time: 0.63 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021346, upper bound: 0.0021100
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0061001, 0.0087421, 0.0064493, 0.0088852, -0.0020410, 0.0015828
1: 0.0022036, 0.0025853, 0.0022540, 0.0026060, -0.0002949, 0.0002287
2: 0.0095266, 0.0109873, 0.0094475, 0.0107942, -0.0008751, 0.0011284
3: -0.0048277, -0.0033169, -0.0049094, -0.0035166, -0.0009051, 0.0011670
4: -0.0004462, 0.0011893, -0.0002301, 0.0012778, -0.0012634, 0.0009798
5: 0.0029879, 0.0045356, 0.0029041, 0.0043311, -0.0009272, 0.0011956
6: -0.0104452, -0.0043043, -0.0107776, -0.0051160, -0.0036789, 0.0047438
7: 0.0033054, 0.0116687, 0.0044108, 0.0121215, -0.0064606, 0.0050103
8: 0.9915423, 0.9974335, 0.9923210, 0.9977525, -0.0045510, 0.0035294
9: -0.0135576, -0.0082099, -0.0138472, -0.0089167, -0.0032037, 0.0041311

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021347, upper bound: 0.0020145
time: 0.67 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021347, upper bound: 0.0021100
time: 0.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.19 seconds
IS_A1_A1_B1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0020470
IS_A1_A1_B1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0020470
IS_A1_A1_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021954
IS_A1_A1_B1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021954
IS_A1_A1_B1_B1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0020293
IS_A1_A1_B1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0022108
IS_A1_A1_B1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0020293
IS_A1_A1_B1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021100, upper bound: 0.0022110
IS_A1_A1_B1_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
IS_A1_A1_B1_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
IS_A1_A1_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
IS_A1_A1_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021062, upper bound: 0.0019637
IS_A1_A1_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
IS_A1_A1_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
IS_A1_A1_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
IS_A1_A1_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021100
IS_A1_A1_B1_B2_A1_B1_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0020179
IS_A1_A1_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0021619
IS_A1_A1_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021577, upper bound: 0.0020179
IS_A1_A1_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021577, upper bound: 0.0021619
IS_A1_A1_B1_B2_A1_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0019662
IS_A1_A1_B1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021434
IS_A1_A1_B1_B2_A1_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0019662
IS_A1_A1_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020130, upper bound: 0.0021434
IS_A1_A1_B1_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0019032
IS_A1_A1_B1_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
IS_A1_A1_B1_B2_A2_B2_B2_A1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0019032
IS_A1_A1_B1_B2_A2_B2_B2_A2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019579, upper bound: 0.0020392
IS_A1_A1_B2_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021255
IS_A1_A1_B2_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0021255
IS_A1_A1_B2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0022905
IS_A1_A1_B2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020470, upper bound: 0.0022905
IS_A1_A1_B2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0020703
IS_A1_A1_B2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0020703
IS_A1_A1_B2_B1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021980
IS_A1_A1_B2_B1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020293, upper bound: 0.0021982
IS_A1_A1_B2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0020996
IS_A1_A1_B2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0020996
IS_A1_A1_B2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0022642
IS_A1_A1_B2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020339, upper bound: 0.0022642
IS_A1_A1_B2_B1_B2_A2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0020229
IS_A1_A1_B2_B1_B2_A2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0020229
IS_A1_A1_B2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0021559
IS_A1_A1_B2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019672, upper bound: 0.0021558
IS_A1_A1_B2_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0021347
IS_A1_A1_B2_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0021347
IS_A1_A1_B2_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0020703
IS_A1_A1_B2_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020145, upper bound: 0.0020703
IS_A1_A1_B2_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0023245
IS_A1_A1_B2_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0023245
IS_A1_A1_B2_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0021980
IS_A1_A1_B2_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019637, upper bound: 0.0021982
IS_A1_A1_B2_B2_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019378, upper bound: 0.0020835
IS_A1_A1_B2_B2_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019378, upper bound: 0.0020835
IS_A1_A1_B2_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0022504
IS_A1_A1_B2_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0022504
IS_A1_A1_B2_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021558
IS_A1_A1_B2_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019090, upper bound: 0.0021559
IS_A1_A2_B1_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
IS_A1_A2_B1_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
IS_A1_A2_B1_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
IS_A1_A2_B1_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0020339
IS_A1_A2_B1_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
IS_A1_A2_B1_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
IS_A1_A2_B1_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
IS_A1_A2_B1_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0021577
IS_A1_A2_B1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
IS_A1_A2_B1_B1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
IS_A1_A2_B1_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
IS_A1_A2_B1_B1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0019070
IS_A1_A2_B1_B1_A2_A2_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
IS_A1_A2_B1_B1_A2_A2_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
IS_A1_A2_B1_B1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
IS_A1_A2_B1_B1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0020111
IS_A1_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0020780
IS_A1_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0020780
IS_A1_A2_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0020773
IS_A1_A2_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0020773
IS_A1_A2_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
IS_A1_A2_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
IS_A1_A2_B2_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
IS_A1_A2_B2_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020701, upper bound: 0.0021375
IS_A1_A2_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
IS_A1_A2_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
IS_A1_A2_B2_B1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
IS_A1_A2_B2_B1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020150, upper bound: 0.0022570
IS_A1_A2_B2_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
IS_A1_A2_B2_B1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
IS_A1_A2_B2_B1_A2_A1_B2_B1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
IS_A1_A2_B2_B1_A2_A1_B2_B2, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020216, upper bound: 0.0020435
IS_A1_A2_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
IS_A1_A2_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
IS_A1_A2_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
IS_A1_A2_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019594, upper bound: 0.0021456
IS_A1_A2_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019559, upper bound: 0.0020943
IS_A1_A2_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019559, upper bound: 0.0020943
IS_A1_A2_B2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019324, upper bound: 0.0020943
IS_A1_A2_B2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019324, upper bound: 0.0020943
IS_A1_A2_B2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0022341
IS_A1_A2_B2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0022341
IS_A1_A2_B2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0021456
IS_A1_A2_B2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0019032, upper bound: 0.0021456
IS_A1_A2_B2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0022317
IS_A1_A2_B2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0022317
IS_A1_A2_B2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0021456
IS_A1_A2_B2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0018974, upper bound: 0.0021456
IS_A2_B1_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0020470
IS_A2_B1_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0021954
IS_A2_B1_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0022905, upper bound: 0.0020470
IS_A2_B1_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0022905, upper bound: 0.0021953
IS_A2_B1_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0020293
IS_A2_B1_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020703, upper bound: 0.0022108
IS_A2_B1_B1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021980, upper bound: 0.0020293
IS_A2_B1_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021980, upper bound: 0.0022110
IS_A2_B1_B1_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020996, upper bound: 0.0020339
IS_A2_B1_B1_A1_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020996, upper bound: 0.0021577
IS_A2_B1_B1_A1_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0022642, upper bound: 0.0020339
IS_A2_B1_B1_A1_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0022642, upper bound: 0.0021577
IS_A2_B1_B1_A1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0019672
IS_A2_B1_B1_A1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0020229, upper bound: 0.0021143
IS_A2_B1_B1_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0019672
IS_A2_B1_B1_A1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021559, upper bound: 0.0021143
IS_A2_B1_B1_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021346, upper bound: 0.0020145
IS_A2_B1_B1_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021346, upper bound: 0.0021100
IS_A2_B1_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021347, upper bound: 0.0020145
IS_A2_B1_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.19
Output dim: 8, lower bound: -0.0021347, upper bound: 0.0021100
IS_A2_B1_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023245, upper bound: 0.0021100
IS_A2_B1_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023245, upper bound: 0.0021100
IS_A2_B1_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020130
IS_A2_B1_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020835, upper bound: 0.0020130
IS_A2_B1_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020130
IS_A2_B1_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022504, upper bound: 0.0020130
IS_A2_B1_B2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020989, upper bound: 0.0021497
IS_A2_B1_B2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020989, upper bound: 0.0021577
IS_A2_B1_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022493, upper bound: 0.0021497
IS_A2_B1_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022493, upper bound: 0.0021577
IS_A2_B1_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020220, upper bound: 0.0021143
IS_A2_B1_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020220, upper bound: 0.0021143
IS_A2_B1_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021426, upper bound: 0.0021143
IS_A2_B1_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021426, upper bound: 0.0021143
IS_A2_B1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020107
IS_A2_B1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020107
IS_A2_B1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020111
IS_A2_B1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0020824, upper bound: 0.0020111
IS_A2_B1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020107
IS_A2_B1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020107
IS_A2_B1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020111
IS_A2_B1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022291, upper bound: 0.0020111
IS_A2_B2_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0020507
IS_A2_B2_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0021607
IS_A2_B2_A1_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0020507
IS_A2_B2_A1_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023074, upper bound: 0.0021607
IS_A2_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021642, upper bound: 0.0021351
IS_A2_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021351
IS_A2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021642, upper bound: 0.0021351
IS_A2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022805, upper bound: 0.0021351
IS_A2_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021308, upper bound: 0.0022910
IS_A2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021308, upper bound: 0.0021607
IS_A2_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022394, upper bound: 0.0022910
IS_A2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022394, upper bound: 0.0021607
IS_A2_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0022639
IS_A2_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0021351
IS_A2_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022153, upper bound: 0.0022639
IS_A2_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022153, upper bound: 0.0021351
IS_A2_B2_A2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0020894
IS_A2_B2_A2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0020894
IS_A2_B2_A2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0021953
IS_A2_B2_A2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022834, upper bound: 0.0021953
IS_A2_B2_A2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0020511
IS_A2_B2_A2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0020511
IS_A2_B2_A2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021362
IS_A2_B2_A2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0023406, upper bound: 0.0021362
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021269
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0020511
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021269
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0020511
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0022554
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021362
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0022554
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.19
Output dim: 8, lower bound: -0.0022065, upper bound: 0.0021362

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.28 + 597.30 = 600.58 seconds
