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
Threshold: 0.00079709


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0027922, -0.0017280, -0.0027922, -0.0017280, -0.0007764, 0.0007764)
1: (-0.0113961, -0.0086956, -0.0113961, -0.0086956, -0.0019703, 0.0019703)
2: (0.0279598, 0.0296353, 0.0279598, 0.0296353, -0.0012224, 0.0012224)
3: (0.0042260, 0.0073545, 0.0042260, 0.0073545, -0.0022825, 0.0022825)
4: (-0.0104848, -0.0077379, -0.0104848, -0.0077379, -0.0020041, 0.0020041)
5: (0.0097668, 0.0108073, 0.0097668, 0.0108073, -0.0007591, 0.0007591)
6: (0.0057514, 0.0097219, 0.0057514, 0.0097219, -0.0028968, 0.0028968)
7: (0.9820838, 0.9848623, 0.9820838, 0.9848623, -0.0020270, 0.0020270)
8: (-0.0057732, -0.0027944, -0.0057732, -0.0027944, -0.0021733, 0.0021733)
9: (-0.0031538, -0.0011861, -0.0031538, -0.0011861, -0.0014356, 0.0014356)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.73 + 1.53 = 3.27 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.0011923, upper bound: 0.0011924

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 105
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 105

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011384, upper bound: 0.0011343
time: 0.63 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011458, upper bound: 0.0011458
time: 0.67 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.49 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 7, lower bound: -0.0011384, upper bound: 0.0011343
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 7, lower bound: -0.0011458, upper bound: 0.0011458

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0028114, -0.0017713, -0.0027913, -0.0017417, -0.0007601, 0.0007262
1: -0.0114449, -0.0088055, -0.0113939, -0.0087304, -0.0019288, 0.0018427
2: 0.0279295, 0.0295671, 0.0279612, 0.0296137, -0.0011966, 0.0011432
3: 0.0043533, 0.0074110, 0.0042663, 0.0073518, -0.0021347, 0.0022344
4: -0.0105344, -0.0078497, -0.0104825, -0.0077733, -0.0019619, 0.0018744
5: 0.0097480, 0.0107649, 0.0097677, 0.0107939, -0.0007431, 0.0007100
6: 0.0059130, 0.0097936, 0.0058026, 0.0097186, -0.0027092, 0.0028357
7: 0.9821970, 0.9849124, 0.9821197, 0.9848599, -0.0018958, 0.0019843
8: -0.0056519, -0.0027406, -0.0057348, -0.0027969, -0.0020326, 0.0021275
9: -0.0031893, -0.0012662, -0.0031521, -0.0012115, -0.0014053, 0.0013426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0010939
time: 0.64 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011059, upper bound: 0.0011003
time: 0.67 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0027908, -0.0017554, -0.0027920, -0.0017322, -0.0007611, 0.0007536
1: -0.0113926, -0.0087651, -0.0113956, -0.0087063, -0.0019313, 0.0019123
2: 0.0279620, 0.0295921, 0.0279601, 0.0296286, -0.0011982, 0.0011864
3: 0.0043065, 0.0073504, 0.0042384, 0.0073539, -0.0022153, 0.0022373
4: -0.0104812, -0.0078086, -0.0104843, -0.0077488, -0.0019644, 0.0019451
5: 0.0097682, 0.0107805, 0.0097670, 0.0108031, -0.0007441, 0.0007367
6: 0.0058537, 0.0097167, 0.0057672, 0.0097211, -0.0028114, 0.0028394
7: 0.9821554, 0.9848586, 0.9820949, 0.9848616, -0.0019673, 0.0019869
8: -0.0056965, -0.0027982, -0.0057613, -0.0027949, -0.0021093, 0.0021303
9: -0.0031512, -0.0012368, -0.0031534, -0.0011939, -0.0014072, 0.0013933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011089, upper bound: 0.0011083
time: 0.66 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0011121, upper bound: 0.0011121
time: 0.63 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.03 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 7, lower bound: -0.0010999, upper bound: 0.0010939
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 7, lower bound: -0.0011059, upper bound: 0.0011003
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 7, lower bound: -0.0011089, upper bound: 0.0011083
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.03
Output dim: 7, lower bound: -0.0011121, upper bound: 0.0011121

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028104, -0.0017854, -0.0027956, -0.0017840, -0.0007040, 0.0006779
1: -0.0114423, -0.0088413, -0.0114047, -0.0088378, -0.0017865, 0.0017203
2: 0.0279312, 0.0295448, 0.0279545, 0.0295470, -0.0011084, 0.0010673
3: 0.0043948, 0.0074079, 0.0043908, 0.0073644, -0.0019929, 0.0020696
4: -0.0105317, -0.0078862, -0.0104936, -0.0078825, -0.0018172, 0.0017499
5: 0.0097490, 0.0107511, 0.0097635, 0.0107525, -0.0006883, 0.0006628
6: 0.0059658, 0.0097897, 0.0059606, 0.0097345, -0.0025293, 0.0026266
7: 0.9822338, 0.9849097, 0.9822302, 0.9848710, -0.0017698, 0.0018380
8: -0.0056124, -0.0027435, -0.0056163, -0.0027849, -0.0018976, 0.0019706
9: -0.0031874, -0.0012923, -0.0031600, -0.0012897, -0.0013017, 0.0012534

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010337
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010381, upper bound: 0.0010337
time: 0.66 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028111, -0.0017763, -0.0027899, -0.0017618, -0.0007001, 0.0007067
1: -0.0114441, -0.0088181, -0.0113903, -0.0087814, -0.0017766, 0.0017933
2: 0.0279301, 0.0295592, 0.0279634, 0.0295820, -0.0011022, 0.0011126
3: 0.0043679, 0.0074100, 0.0043254, 0.0073477, -0.0020775, 0.0020581
4: -0.0105336, -0.0078625, -0.0104789, -0.0078252, -0.0018071, 0.0018241
5: 0.0097483, 0.0107601, 0.0097691, 0.0107742, -0.0006845, 0.0006909
6: 0.0059316, 0.0097924, 0.0058776, 0.0097133, -0.0026366, 0.0026120
7: 0.9822099, 0.9849116, 0.9821721, 0.9848561, -0.0018449, 0.0018278
8: -0.0056380, -0.0027415, -0.0056785, -0.0028008, -0.0019781, 0.0019596
9: -0.0031887, -0.0012754, -0.0031495, -0.0012486, -0.0012945, 0.0013066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010377, upper bound: 0.0010381
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010438, upper bound: 0.0010381
time: 0.66 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027897, -0.0017694, -0.0027962, -0.0017741, -0.0007037, 0.0007103
1: -0.0113898, -0.0088008, -0.0114063, -0.0088127, -0.0017858, 0.0018024
2: 0.0279638, 0.0295700, 0.0279535, 0.0295626, -0.0011079, 0.0011182
3: 0.0043478, 0.0073471, 0.0043617, 0.0073662, -0.0020880, 0.0020688
4: -0.0104783, -0.0078449, -0.0104951, -0.0078570, -0.0018165, 0.0018334
5: 0.0097693, 0.0107668, 0.0097629, 0.0107622, -0.0006880, 0.0006944
6: 0.0059061, 0.0097125, 0.0059236, 0.0097368, -0.0026500, 0.0026256
7: 0.9821920, 0.9848556, 0.9822043, 0.9848726, -0.0018543, 0.0018372
8: -0.0056571, -0.0028014, -0.0056440, -0.0027832, -0.0019881, 0.0019698
9: -0.0031491, -0.0012627, -0.0031612, -0.0012714, -0.0013012, 0.0013133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010405, upper bound: 0.0010423
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010417, upper bound: 0.0010423
time: 0.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027905, -0.0017606, -0.0027905, -0.0017525, -0.0006970, 0.0007344
1: -0.0113918, -0.0087785, -0.0113920, -0.0087577, -0.0017688, 0.0018636
2: 0.0279625, 0.0295838, 0.0279624, 0.0295967, -0.0010974, 0.0011562
3: 0.0043220, 0.0073494, 0.0042980, 0.0073496, -0.0021589, 0.0020491
4: -0.0104804, -0.0078222, -0.0104806, -0.0078011, -0.0017992, 0.0018956
5: 0.0097685, 0.0107753, 0.0097684, 0.0107833, -0.0006815, 0.0007180
6: 0.0058733, 0.0097155, 0.0058428, 0.0097158, -0.0027400, 0.0026006
7: 0.9821692, 0.9848577, 0.9821478, 0.9848579, -0.0019173, 0.0018197
8: -0.0056817, -0.0027992, -0.0057046, -0.0027990, -0.0020556, 0.0019511
9: -0.0031506, -0.0012465, -0.0031507, -0.0012314, -0.0012888, 0.0013579

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010449, upper bound: 0.0010461
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010461, upper bound: 0.0010461
time: 0.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.03 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010337
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010381, upper bound: 0.0010337
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010377, upper bound: 0.0010381
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010438, upper bound: 0.0010381
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010405, upper bound: 0.0010423
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010417, upper bound: 0.0010423
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010449, upper bound: 0.0010461
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.03
Output dim: 7, lower bound: -0.0010461, upper bound: 0.0010461

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028095, -0.0018045, -0.0027956, -0.0017840, -0.0007034, 0.0006579
1: -0.0114400, -0.0088897, -0.0114047, -0.0088378, -0.0017850, 0.0016696
2: 0.0279326, 0.0295148, 0.0279545, 0.0295470, -0.0011074, 0.0010358
3: 0.0044509, 0.0074053, 0.0043908, 0.0073644, -0.0019341, 0.0020678
4: -0.0105294, -0.0079353, -0.0104936, -0.0078825, -0.0018156, 0.0016982
5: 0.0097499, 0.0107325, 0.0097635, 0.0107525, -0.0006877, 0.0006432
6: 0.0060368, 0.0097863, 0.0059606, 0.0097345, -0.0024546, 0.0026243
7: 0.9822836, 0.9849072, 0.9822302, 0.9848710, -0.0017176, 0.0018364
8: -0.0055590, -0.0027460, -0.0056163, -0.0027849, -0.0018416, 0.0019689
9: -0.0031857, -0.0013275, -0.0031600, -0.0012897, -0.0013005, 0.0012165

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010313, upper bound: 0.0010269
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010313, upper bound: 0.0010337
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028664, -0.0018268, -0.0027948, -0.0017983, -0.0007639, 0.0006599
1: -0.0115845, -0.0089463, -0.0114027, -0.0088740, -0.0019384, 0.0016745
2: 0.0278430, 0.0294797, 0.0279557, 0.0295246, -0.0012026, 0.0010389
3: 0.0045164, 0.0075727, 0.0044327, 0.0073621, -0.0019398, 0.0022456
4: -0.0106764, -0.0079929, -0.0104915, -0.0079194, -0.0019717, 0.0017032
5: 0.0096942, 0.0107107, 0.0097643, 0.0107385, -0.0007468, 0.0006451
6: 0.0061201, 0.0099988, 0.0060137, 0.0097315, -0.0024619, 0.0028499
7: 0.9823419, 0.9850560, 0.9822674, 0.9848689, -0.0017227, 0.0019942
8: -0.0054966, -0.0025866, -0.0055764, -0.0027871, -0.0018470, 0.0021381
9: -0.0032910, -0.0013688, -0.0031585, -0.0013161, -0.0014124, 0.0012201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010336
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010337
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028102, -0.0017954, -0.0027899, -0.0017618, -0.0006994, 0.0006868
1: -0.0114418, -0.0088665, -0.0113903, -0.0087814, -0.0017749, 0.0017429
2: 0.0279315, 0.0295292, 0.0279634, 0.0295820, -0.0011012, 0.0010813
3: 0.0044240, 0.0074074, 0.0043254, 0.0073477, -0.0020190, 0.0020562
4: -0.0105313, -0.0079118, -0.0104789, -0.0078252, -0.0018054, 0.0017728
5: 0.0097492, 0.0107414, 0.0097691, 0.0107742, -0.0006838, 0.0006715
6: 0.0060028, 0.0097890, 0.0058776, 0.0097133, -0.0025624, 0.0026095
7: 0.9822598, 0.9849092, 0.9821721, 0.9848561, -0.0017931, 0.0018260
8: -0.0055846, -0.0027440, -0.0056785, -0.0028008, -0.0019224, 0.0019578
9: -0.0031870, -0.0013107, -0.0031495, -0.0012486, -0.0012932, 0.0012699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010370, upper bound: 0.0010332
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010370, upper bound: 0.0010381
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028677, -0.0018179, -0.0027891, -0.0017770, -0.0007663, 0.0006886
1: -0.0115878, -0.0089237, -0.0113883, -0.0088199, -0.0019446, 0.0017474
2: 0.0278409, 0.0294937, 0.0279646, 0.0295582, -0.0012064, 0.0010841
3: 0.0044903, 0.0075766, 0.0043700, 0.0073454, -0.0020243, 0.0022527
4: -0.0106798, -0.0079699, -0.0104769, -0.0078643, -0.0019780, 0.0017774
5: 0.0096930, 0.0107194, 0.0097698, 0.0107594, -0.0007492, 0.0006732
6: 0.0060869, 0.0100037, 0.0059342, 0.0097104, -0.0025691, 0.0028590
7: 0.9823186, 0.9850594, 0.9822117, 0.9848542, -0.0017977, 0.0020006
8: -0.0055215, -0.0025829, -0.0056361, -0.0028030, -0.0019275, 0.0021450
9: -0.0032934, -0.0013523, -0.0031481, -0.0012766, -0.0014169, 0.0012732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010438, upper bound: 0.0010381
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010438, upper bound: 0.0010381
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027888, -0.0017854, -0.0027962, -0.0017741, -0.0007030, 0.0006891
1: -0.0113874, -0.0088412, -0.0114063, -0.0088127, -0.0017840, 0.0017487
2: 0.0279652, 0.0295449, 0.0279535, 0.0295626, -0.0011068, 0.0010849
3: 0.0043947, 0.0073444, 0.0043617, 0.0073662, -0.0020258, 0.0020667
4: -0.0104760, -0.0078860, -0.0104951, -0.0078570, -0.0018147, 0.0017788
5: 0.0097702, 0.0107512, 0.0097629, 0.0107622, -0.0006873, 0.0006737
6: 0.0059655, 0.0097091, 0.0059236, 0.0097368, -0.0025710, 0.0026229
7: 0.9822337, 0.9848533, 0.9822043, 0.9848726, -0.0017991, 0.0018354
8: -0.0056126, -0.0028040, -0.0056440, -0.0027832, -0.0019289, 0.0019678
9: -0.0031474, -0.0012922, -0.0031612, -0.0012714, -0.0012999, 0.0012741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010362, upper bound: 0.0010359
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010362, upper bound: 0.0010423
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028463, -0.0018090, -0.0027954, -0.0017887, -0.0007572, 0.0006888
1: -0.0115336, -0.0089012, -0.0114042, -0.0088496, -0.0019214, 0.0017479
2: 0.0278745, 0.0295077, 0.0279548, 0.0295397, -0.0011921, 0.0010844
3: 0.0044643, 0.0075137, 0.0044044, 0.0073638, -0.0020249, 0.0022259
4: -0.0106246, -0.0079471, -0.0104930, -0.0078945, -0.0019544, 0.0017779
5: 0.0097139, 0.0107280, 0.0097637, 0.0107480, -0.0007403, 0.0006734
6: 0.0060538, 0.0099239, 0.0059778, 0.0097338, -0.0025698, 0.0028249
7: 0.9822955, 0.9850035, 0.9822423, 0.9848705, -0.0017982, 0.0019768
8: -0.0055463, -0.0026428, -0.0056033, -0.0027854, -0.0019280, 0.0021194
9: -0.0032539, -0.0013360, -0.0031597, -0.0012983, -0.0014000, 0.0012735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010329, upper bound: 0.0010337
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010329, upper bound: 0.0010423
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027896, -0.0017766, -0.0027905, -0.0017525, -0.0006963, 0.0007142
1: -0.0113895, -0.0088188, -0.0113920, -0.0087577, -0.0017670, 0.0018124
2: 0.0279639, 0.0295588, 0.0279624, 0.0295967, -0.0010963, 0.0011244
3: 0.0043688, 0.0073467, 0.0042980, 0.0073496, -0.0020996, 0.0020470
4: -0.0104780, -0.0078633, -0.0104806, -0.0078011, -0.0017973, 0.0018435
5: 0.0097694, 0.0107598, 0.0097684, 0.0107833, -0.0006808, 0.0006983
6: 0.0059327, 0.0097121, 0.0058428, 0.0097158, -0.0026647, 0.0025979
7: 0.9822106, 0.9848554, 0.9821478, 0.9848579, -0.0018646, 0.0018179
8: -0.0056372, -0.0028017, -0.0057046, -0.0027990, -0.0019991, 0.0019491
9: -0.0031489, -0.0012759, -0.0031507, -0.0012314, -0.0012875, 0.0013205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010408
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010460
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028472, -0.0018010, -0.0027898, -0.0017677, -0.0007590, 0.0007142
1: -0.0115359, -0.0088810, -0.0113900, -0.0087965, -0.0019259, 0.0018123
2: 0.0278731, 0.0295202, 0.0279636, 0.0295727, -0.0011949, 0.0011244
3: 0.0044408, 0.0075163, 0.0043429, 0.0073473, -0.0020995, 0.0022311
4: -0.0106269, -0.0079265, -0.0104786, -0.0078405, -0.0019590, 0.0018434
5: 0.0097130, 0.0107358, 0.0097692, 0.0107684, -0.0007420, 0.0006982
6: 0.0060240, 0.0099273, 0.0058998, 0.0097128, -0.0026645, 0.0028316
7: 0.9822746, 0.9850059, 0.9821876, 0.9848558, -0.0018645, 0.0019814
8: -0.0055687, -0.0026402, -0.0056619, -0.0028011, -0.0019990, 0.0021244
9: -0.0032556, -0.0013212, -0.0031493, -0.0012596, -0.0014033, 0.0013205

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010381, upper bound: 0.0010386
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010381, upper bound: 0.0010461
time: 0.78 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.52 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010313, upper bound: 0.0010269
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010313, upper bound: 0.0010337
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010336
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010337
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010370, upper bound: 0.0010332
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010370, upper bound: 0.0010381
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010438, upper bound: 0.0010381
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010438, upper bound: 0.0010381
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010362, upper bound: 0.0010359
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010362, upper bound: 0.0010423
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010329, upper bound: 0.0010337
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010329, upper bound: 0.0010423
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010408
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010460
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010381, upper bound: 0.0010386
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.52
Output dim: 7, lower bound: -0.0010381, upper bound: 0.0010461

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028095, -0.0018045, -0.0027946, -0.0018013, -0.0006837, 0.0006572
1: -0.0114400, -0.0088897, -0.0114023, -0.0088817, -0.0017349, 0.0016678
2: 0.0279326, 0.0295148, 0.0279560, 0.0295198, -0.0010763, 0.0010347
3: 0.0044509, 0.0074053, 0.0044416, 0.0073616, -0.0019320, 0.0020098
4: -0.0105294, -0.0079353, -0.0104910, -0.0079272, -0.0017647, 0.0016964
5: 0.0097499, 0.0107325, 0.0097645, 0.0107356, -0.0006684, 0.0006426
6: 0.0060368, 0.0097863, 0.0060251, 0.0097309, -0.0024520, 0.0025507
7: 0.9822836, 0.9849072, 0.9822754, 0.9848685, -0.0017158, 0.0017848
8: -0.0055590, -0.0027460, -0.0055678, -0.0027876, -0.0018396, 0.0019136
9: -0.0031857, -0.0013275, -0.0031582, -0.0013217, -0.0012641, 0.0012152

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010272
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010272
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028095, -0.0018045, -0.0028518, -0.0018228, -0.0006722, 0.0007304
1: -0.0114400, -0.0088897, -0.0115474, -0.0089361, -0.0017059, 0.0018536
2: 0.0279326, 0.0295148, 0.0278659, 0.0294860, -0.0010583, 0.0011500
3: 0.0044509, 0.0074053, 0.0045047, 0.0075297, -0.0021473, 0.0019762
4: -0.0105294, -0.0079353, -0.0106387, -0.0079826, -0.0017352, 0.0018854
5: 0.0097499, 0.0107325, 0.0097085, 0.0107146, -0.0006572, 0.0007141
6: 0.0060368, 0.0097863, 0.0061051, 0.0099443, -0.0027252, 0.0025080
7: 0.9822836, 0.9849072, 0.9823313, 0.9850178, -0.0019070, 0.0017550
8: -0.0055590, -0.0027460, -0.0055078, -0.0026275, -0.0020446, 0.0018816
9: -0.0031857, -0.0013275, -0.0032640, -0.0013614, -0.0012429, 0.0013505

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010337
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010337
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028664, -0.0018268, -0.0028129, -0.0018279, -0.0007366, 0.0006680
1: -0.0115845, -0.0089463, -0.0114487, -0.0089490, -0.0018692, 0.0016950
2: 0.0278430, 0.0294797, 0.0279272, 0.0294780, -0.0011597, 0.0010516
3: 0.0045164, 0.0075727, 0.0045196, 0.0074154, -0.0019636, 0.0021654
4: -0.0106764, -0.0079929, -0.0105383, -0.0079957, -0.0019013, 0.0017241
5: 0.0096942, 0.0107107, 0.0097466, 0.0107096, -0.0007202, 0.0006531
6: 0.0061201, 0.0099988, 0.0061241, 0.0097992, -0.0024921, 0.0027482
7: 0.9823419, 0.9850560, 0.9823446, 0.9849162, -0.0017438, 0.0019230
8: -0.0054966, -0.0025866, -0.0054936, -0.0027364, -0.0018697, 0.0020618
9: -0.0032910, -0.0013688, -0.0031921, -0.0013708, -0.0013619, 0.0012350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010336
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010336
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028664, -0.0018268, -0.0027943, -0.0018116, -0.0007499, 0.0006536
1: -0.0115845, -0.0089463, -0.0114015, -0.0089078, -0.0019029, 0.0016586
2: 0.0278430, 0.0294797, 0.0279565, 0.0295036, -0.0011806, 0.0010290
3: 0.0045164, 0.0075727, 0.0044718, 0.0073607, -0.0019214, 0.0022044
4: -0.0106764, -0.0079929, -0.0104903, -0.0079538, -0.0019356, 0.0016871
5: 0.0096942, 0.0107107, 0.0097647, 0.0107255, -0.0007331, 0.0006390
6: 0.0061201, 0.0099988, 0.0060635, 0.0097298, -0.0024385, 0.0027977
7: 0.9823419, 0.9850560, 0.9823022, 0.9848677, -0.0017064, 0.0019577
8: -0.0054966, -0.0025866, -0.0055391, -0.0027884, -0.0018295, 0.0020990
9: -0.0032910, -0.0013688, -0.0031577, -0.0013407, -0.0013865, 0.0012085

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010337
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010337
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028102, -0.0017954, -0.0027890, -0.0017796, -0.0006788, 0.0006861
1: -0.0114418, -0.0088665, -0.0113879, -0.0088266, -0.0017226, 0.0017411
2: 0.0279315, 0.0295292, 0.0279649, 0.0295540, -0.0010687, 0.0010802
3: 0.0044240, 0.0074074, 0.0043777, 0.0073450, -0.0020170, 0.0019956
4: -0.0105313, -0.0079118, -0.0104765, -0.0078711, -0.0017522, 0.0017710
5: 0.0097492, 0.0107414, 0.0097700, 0.0107568, -0.0006637, 0.0006708
6: 0.0060028, 0.0097890, 0.0059440, 0.0097098, -0.0025599, 0.0025326
7: 0.9822598, 0.9849092, 0.9822186, 0.9848537, -0.0017913, 0.0017722
8: -0.0055846, -0.0027440, -0.0056287, -0.0028034, -0.0019205, 0.0019001
9: -0.0031870, -0.0013107, -0.0031478, -0.0012815, -0.0012551, 0.0012686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010274
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010319
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028102, -0.0017954, -0.0028460, -0.0018046, -0.0006698, 0.0007537
1: -0.0114418, -0.0088665, -0.0115327, -0.0088900, -0.0016997, 0.0019127
2: 0.0279315, 0.0295292, 0.0278751, 0.0295146, -0.0010545, 0.0011866
3: 0.0044240, 0.0074074, 0.0044512, 0.0075126, -0.0022157, 0.0019690
4: -0.0105313, -0.0079118, -0.0106237, -0.0079357, -0.0017289, 0.0019455
5: 0.0097492, 0.0107414, 0.0097142, 0.0107324, -0.0006549, 0.0007369
6: 0.0060028, 0.0097890, 0.0060373, 0.0099226, -0.0028121, 0.0024990
7: 0.9822598, 0.9849092, 0.9822839, 0.9850026, -0.0019677, 0.0017487
8: -0.0055846, -0.0027440, -0.0055587, -0.0026438, -0.0021097, 0.0018748
9: -0.0031870, -0.0013107, -0.0032532, -0.0013278, -0.0012384, 0.0013936

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010328
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010377
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028677, -0.0018179, -0.0028094, -0.0018055, -0.0007390, 0.0006935
1: -0.0115878, -0.0089237, -0.0114398, -0.0088924, -0.0018753, 0.0017599
2: 0.0278409, 0.0294937, 0.0279327, 0.0295132, -0.0011635, 0.0010919
3: 0.0044903, 0.0075766, 0.0044540, 0.0074050, -0.0020388, 0.0021725
4: -0.0106798, -0.0079699, -0.0105292, -0.0079381, -0.0019075, 0.0017901
5: 0.0096930, 0.0107194, 0.0097500, 0.0107315, -0.0007225, 0.0006781
6: 0.0060869, 0.0100037, 0.0060408, 0.0097860, -0.0025875, 0.0027572
7: 0.9823186, 0.9850594, 0.9822863, 0.9849070, -0.0018106, 0.0019293
8: -0.0055215, -0.0025829, -0.0055561, -0.0027462, -0.0019413, 0.0020685
9: -0.0032934, -0.0013523, -0.0031855, -0.0013295, -0.0013664, 0.0012823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010328
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010377
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028677, -0.0018179, -0.0027886, -0.0017913, -0.0007528, 0.0006813
1: -0.0115878, -0.0089237, -0.0113870, -0.0088561, -0.0019104, 0.0017290
2: 0.0278409, 0.0294937, 0.0279654, 0.0295356, -0.0011852, 0.0010727
3: 0.0044903, 0.0075766, 0.0044120, 0.0073439, -0.0020030, 0.0022131
4: -0.0106798, -0.0079699, -0.0104756, -0.0079012, -0.0019432, 0.0017587
5: 0.0096930, 0.0107194, 0.0097703, 0.0107454, -0.0007360, 0.0006661
6: 0.0060869, 0.0100037, 0.0059875, 0.0097085, -0.0025420, 0.0028088
7: 0.9823186, 0.9850594, 0.9822491, 0.9848528, -0.0017788, 0.0019654
8: -0.0055215, -0.0025829, -0.0055961, -0.0028044, -0.0019071, 0.0021073
9: -0.0032934, -0.0013523, -0.0031471, -0.0013031, -0.0013920, 0.0012598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010329
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010377
time: 0.73 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027888, -0.0017854, -0.0027952, -0.0017907, -0.0006840, 0.0006884
1: -0.0113874, -0.0088412, -0.0114038, -0.0088547, -0.0017358, 0.0017470
2: 0.0279652, 0.0295449, 0.0279551, 0.0295365, -0.0010769, 0.0010839
3: 0.0043947, 0.0073444, 0.0044103, 0.0073633, -0.0020238, 0.0020109
4: -0.0104760, -0.0078860, -0.0104926, -0.0078997, -0.0017656, 0.0017770
5: 0.0097702, 0.0107512, 0.0097639, 0.0107460, -0.0006688, 0.0006731
6: 0.0059655, 0.0097091, 0.0059854, 0.0097331, -0.0025685, 0.0025521
7: 0.9822337, 0.9848533, 0.9822476, 0.9848701, -0.0017973, 0.0017858
8: -0.0056126, -0.0028040, -0.0055976, -0.0027859, -0.0019270, 0.0019147
9: -0.0031474, -0.0012922, -0.0031593, -0.0013020, -0.0012648, 0.0012729

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010368
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010368
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027888, -0.0017854, -0.0028524, -0.0018128, -0.0006707, 0.0007669
1: -0.0113874, -0.0088412, -0.0115490, -0.0089107, -0.0017021, 0.0019461
2: 0.0279652, 0.0295449, 0.0278650, 0.0295018, -0.0010560, 0.0012073
3: 0.0043947, 0.0073444, 0.0044752, 0.0075315, -0.0022544, 0.0019718
4: -0.0104760, -0.0078860, -0.0106403, -0.0079567, -0.0017313, 0.0019795
5: 0.0097702, 0.0107512, 0.0097079, 0.0107244, -0.0006558, 0.0007498
6: 0.0059655, 0.0097091, 0.0060678, 0.0099466, -0.0028611, 0.0025024
7: 0.9822337, 0.9848533, 0.9823052, 0.9850194, -0.0020021, 0.0017511
8: -0.0056126, -0.0028040, -0.0055358, -0.0026258, -0.0021466, 0.0018774
9: -0.0031474, -0.0012922, -0.0032651, -0.0013429, -0.0012401, 0.0014179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010423
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010423
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028463, -0.0018090, -0.0028129, -0.0018279, -0.0007205, 0.0006894
1: -0.0115336, -0.0089012, -0.0114487, -0.0089490, -0.0018283, 0.0017494
2: 0.0278745, 0.0295077, 0.0279272, 0.0294780, -0.0011343, 0.0010853
3: 0.0044643, 0.0075137, 0.0045196, 0.0074154, -0.0020265, 0.0021180
4: -0.0106246, -0.0079471, -0.0105383, -0.0079957, -0.0018597, 0.0017794
5: 0.0097139, 0.0107280, 0.0097466, 0.0107096, -0.0007044, 0.0006740
6: 0.0060538, 0.0099239, 0.0061241, 0.0097992, -0.0025719, 0.0026881
7: 0.9822955, 0.9850035, 0.9823446, 0.9849162, -0.0017997, 0.0018810
8: -0.0055463, -0.0026428, -0.0054936, -0.0027364, -0.0019296, 0.0020167
9: -0.0032539, -0.0013360, -0.0031921, -0.0013708, -0.0013321, 0.0012746

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010337
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010337
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028463, -0.0018090, -0.0027943, -0.0018116, -0.0007549, 0.0006828
1: -0.0115336, -0.0089012, -0.0114015, -0.0089078, -0.0019157, 0.0017327
2: 0.0278745, 0.0295077, 0.0279565, 0.0295036, -0.0011885, 0.0010749
3: 0.0044643, 0.0075137, 0.0044718, 0.0073607, -0.0020072, 0.0022192
4: -0.0106246, -0.0079471, -0.0104903, -0.0079538, -0.0019486, 0.0017624
5: 0.0097139, 0.0107280, 0.0097647, 0.0107255, -0.0007381, 0.0006675
6: 0.0060538, 0.0099239, 0.0060635, 0.0097298, -0.0025474, 0.0028165
7: 0.9822955, 0.9850035, 0.9823022, 0.9848677, -0.0017825, 0.0019708
8: -0.0055463, -0.0026428, -0.0055391, -0.0027884, -0.0019112, 0.0021130
9: -0.0032539, -0.0013360, -0.0031577, -0.0013407, -0.0013958, 0.0012624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010423
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010423
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027896, -0.0017766, -0.0027896, -0.0017695, -0.0006765, 0.0007136
1: -0.0113895, -0.0088188, -0.0113896, -0.0088009, -0.0017168, 0.0018109
2: 0.0279639, 0.0295588, 0.0279639, 0.0295699, -0.0010651, 0.0011235
3: 0.0043688, 0.0073467, 0.0043480, 0.0073469, -0.0020978, 0.0019888
4: -0.0104780, -0.0078633, -0.0104782, -0.0078450, -0.0017463, 0.0018420
5: 0.0097694, 0.0107598, 0.0097693, 0.0107667, -0.0006614, 0.0006977
6: 0.0059327, 0.0097121, 0.0059063, 0.0097123, -0.0026624, 0.0025241
7: 0.9822106, 0.9848554, 0.9821922, 0.9848555, -0.0018630, 0.0017662
8: -0.0056372, -0.0028017, -0.0056570, -0.0028016, -0.0019974, 0.0018937
9: -0.0031489, -0.0012759, -0.0031490, -0.0012629, -0.0012509, 0.0013194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010369
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010397
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027896, -0.0017766, -0.0028472, -0.0017952, -0.0006665, 0.0007877
1: -0.0113895, -0.0088188, -0.0115357, -0.0088662, -0.0016913, 0.0019989
2: 0.0279639, 0.0295588, 0.0278733, 0.0295294, -0.0010493, 0.0012401
3: 0.0043688, 0.0073467, 0.0044236, 0.0075161, -0.0023156, 0.0019593
4: -0.0104780, -0.0078633, -0.0106267, -0.0079114, -0.0017203, 0.0020332
5: 0.0097694, 0.0107598, 0.0097131, 0.0107415, -0.0006516, 0.0007701
6: 0.0059327, 0.0097121, 0.0060023, 0.0099270, -0.0029388, 0.0024866
7: 0.9822106, 0.9848554, 0.9822594, 0.9850056, -0.0020565, 0.0017400
8: -0.0056372, -0.0028017, -0.0055850, -0.0026405, -0.0022048, 0.0018655
9: -0.0031489, -0.0012759, -0.0032554, -0.0013104, -0.0012323, 0.0014564

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010417
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010458
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028472, -0.0018010, -0.0028094, -0.0018055, -0.0007220, 0.0007168
1: -0.0115359, -0.0088810, -0.0114398, -0.0088924, -0.0018323, 0.0018190
2: 0.0278731, 0.0295202, 0.0279327, 0.0295132, -0.0011368, 0.0011285
3: 0.0044408, 0.0075163, 0.0044540, 0.0074050, -0.0021072, 0.0021226
4: -0.0106269, -0.0079265, -0.0105292, -0.0079381, -0.0018638, 0.0018502
5: 0.0097130, 0.0107358, 0.0097500, 0.0107315, -0.0007059, 0.0007008
6: 0.0060240, 0.0099273, 0.0060408, 0.0097860, -0.0026743, 0.0026939
7: 0.9822746, 0.9850059, 0.9822863, 0.9849070, -0.0018714, 0.0018851
8: -0.0055687, -0.0026402, -0.0055561, -0.0027462, -0.0020064, 0.0020211
9: -0.0032556, -0.0013212, -0.0031855, -0.0013295, -0.0013350, 0.0013253

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010332
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010382
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028472, -0.0018010, -0.0027886, -0.0017913, -0.0007594, 0.0007082
1: -0.0115359, -0.0088810, -0.0113871, -0.0088561, -0.0019270, 0.0017972
2: 0.0278731, 0.0295202, 0.0279654, 0.0295356, -0.0011955, 0.0011150
3: 0.0044408, 0.0075163, 0.0044120, 0.0073439, -0.0020820, 0.0022324
4: -0.0106269, -0.0079265, -0.0104756, -0.0079012, -0.0019601, 0.0018281
5: 0.0097130, 0.0107358, 0.0097703, 0.0107454, -0.0007424, 0.0006924
6: 0.0060240, 0.0099273, 0.0059875, 0.0097085, -0.0026423, 0.0028331
7: 0.9822746, 0.9850059, 0.9822491, 0.9848528, -0.0018490, 0.0019825
8: -0.0055687, -0.0026402, -0.0055961, -0.0028044, -0.0019824, 0.0021255
9: -0.0032556, -0.0013212, -0.0031471, -0.0013031, -0.0014040, 0.0013095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010417
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010458
time: 0.65 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.05 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010272
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010272
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010337
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010337
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010336
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010336
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010337
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010337
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010274
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010319
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010328
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010282, upper bound: 0.0010377
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010328
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010377
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010329
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010377
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010368
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010368
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010423
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010423
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010337
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010337
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010423
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010423
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010369
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010397
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010417
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010380, upper bound: 0.0010458
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010332
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010382
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010417
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.05
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010458

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0027946, -0.0018013, -0.0006630, 0.0006264
1: -0.0114483, -0.0089602, -0.0114023, -0.0088817, -0.0016825, 0.0015895
2: 0.0279274, 0.0294711, 0.0279560, 0.0295198, -0.0010438, 0.0009862
3: 0.0045325, 0.0074150, 0.0044416, 0.0073616, -0.0018414, 0.0019490
4: -0.0105379, -0.0080071, -0.0104910, -0.0079272, -0.0017113, 0.0016168
5: 0.0097467, 0.0107053, 0.0097645, 0.0107356, -0.0006482, 0.0006124
6: 0.0061405, 0.0097987, 0.0060251, 0.0097309, -0.0023370, 0.0024736
7: 0.9823561, 0.9849159, 0.9822754, 0.9848685, -0.0016353, 0.0017309
8: -0.0054813, -0.0027368, -0.0055678, -0.0027876, -0.0017533, 0.0018558
9: -0.0031918, -0.0013789, -0.0031582, -0.0013217, -0.0012259, 0.0011582

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010448
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010467
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0027946, -0.0018013, -0.0006715, 0.0006624
1: -0.0114394, -0.0089014, -0.0114023, -0.0088817, -0.0017041, 0.0016808
2: 0.0279330, 0.0295076, 0.0279560, 0.0295198, -0.0010572, 0.0010428
3: 0.0044644, 0.0074046, 0.0044416, 0.0073616, -0.0019472, 0.0019742
4: -0.0105288, -0.0079472, -0.0104910, -0.0079272, -0.0017334, 0.0017097
5: 0.0097502, 0.0107280, 0.0097645, 0.0107356, -0.0006566, 0.0006476
6: 0.0060540, 0.0097855, 0.0060251, 0.0097309, -0.0024712, 0.0025055
7: 0.9822956, 0.9849067, 0.9822754, 0.9848685, -0.0017292, 0.0017532
8: -0.0055462, -0.0027466, -0.0055678, -0.0027876, -0.0018540, 0.0018797
9: -0.0031853, -0.0013360, -0.0031582, -0.0013217, -0.0012416, 0.0012247

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010448
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010467
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028518, -0.0018228, -0.0006516, 0.0006996
1: -0.0114483, -0.0089602, -0.0115474, -0.0089361, -0.0016535, 0.0017754
2: 0.0279274, 0.0294711, 0.0278659, 0.0294860, -0.0010258, 0.0011014
3: 0.0045325, 0.0074150, 0.0045047, 0.0075297, -0.0020567, 0.0019155
4: -0.0105379, -0.0080071, -0.0106387, -0.0079826, -0.0016818, 0.0018058
5: 0.0097467, 0.0107053, 0.0097085, 0.0107146, -0.0006370, 0.0006840
6: 0.0061405, 0.0097987, 0.0061051, 0.0099443, -0.0026102, 0.0024310
7: 0.9823561, 0.9849159, 0.9823313, 0.9850178, -0.0018265, 0.0017011
8: -0.0054813, -0.0027368, -0.0055078, -0.0026275, -0.0019583, 0.0018238
9: -0.0031918, -0.0013789, -0.0032640, -0.0013614, -0.0012047, 0.0012935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010336
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010337
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028518, -0.0018228, -0.0006601, 0.0007356
1: -0.0114394, -0.0089014, -0.0115474, -0.0089361, -0.0016751, 0.0018666
2: 0.0279330, 0.0295076, 0.0278659, 0.0294860, -0.0010393, 0.0011581
3: 0.0044644, 0.0074046, 0.0045047, 0.0075297, -0.0021624, 0.0019406
4: -0.0105288, -0.0079472, -0.0106387, -0.0079826, -0.0017039, 0.0018987
5: 0.0097502, 0.0107280, 0.0097085, 0.0107146, -0.0006454, 0.0007192
6: 0.0060540, 0.0097855, 0.0061051, 0.0099443, -0.0027444, 0.0024628
7: 0.9822956, 0.9849067, 0.9823313, 0.9850178, -0.0019204, 0.0017234
8: -0.0055462, -0.0027466, -0.0055078, -0.0026275, -0.0020590, 0.0018477
9: -0.0031853, -0.0013360, -0.0032640, -0.0013614, -0.0012205, 0.0013601

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010336
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010337
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028129, -0.0018279, -0.0007213, 0.0006370
1: -0.0115958, -0.0090144, -0.0114487, -0.0089490, -0.0018304, 0.0016164
2: 0.0278359, 0.0294374, 0.0279272, 0.0294780, -0.0011356, 0.0010028
3: 0.0045954, 0.0075858, 0.0045196, 0.0074154, -0.0018725, 0.0021204
4: -0.0106879, -0.0080622, -0.0105383, -0.0079957, -0.0018618, 0.0016441
5: 0.0096899, 0.0106844, 0.0097466, 0.0107096, -0.0007052, 0.0006227
6: 0.0062202, 0.0100155, 0.0061241, 0.0097992, -0.0023764, 0.0026911
7: 0.9824119, 0.9850676, 0.9823446, 0.9849162, -0.0016629, 0.0018831
8: -0.0054215, -0.0025741, -0.0054936, -0.0027364, -0.0017829, 0.0020190
9: -0.0032993, -0.0014184, -0.0031921, -0.0013708, -0.0013337, 0.0011777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018343, -0.0028129, -0.0018279, -0.0007257, 0.0006729
1: -0.0115838, -0.0089654, -0.0114487, -0.0089490, -0.0018416, 0.0017076
2: 0.0278434, 0.0294679, 0.0279272, 0.0294780, -0.0011425, 0.0010594
3: 0.0045385, 0.0075719, 0.0045196, 0.0074154, -0.0019782, 0.0021334
4: -0.0106757, -0.0080123, -0.0105383, -0.0079957, -0.0018732, 0.0017369
5: 0.0096945, 0.0107033, 0.0097466, 0.0107096, -0.0007095, 0.0006579
6: 0.0061481, 0.0099978, 0.0061241, 0.0097992, -0.0025105, 0.0027076
7: 0.9823614, 0.9850553, 0.9823446, 0.9849162, -0.0017567, 0.0018946
8: -0.0054756, -0.0025874, -0.0054936, -0.0027364, -0.0018835, 0.0020313
9: -0.0032905, -0.0013827, -0.0031921, -0.0013708, -0.0013418, 0.0012442

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0027943, -0.0018116, -0.0007346, 0.0006226
1: -0.0115958, -0.0090144, -0.0114015, -0.0089078, -0.0018641, 0.0015799
2: 0.0278359, 0.0294374, 0.0279565, 0.0295036, -0.0011565, 0.0009802
3: 0.0045954, 0.0075858, 0.0044718, 0.0073607, -0.0018303, 0.0021595
4: -0.0106879, -0.0080622, -0.0104903, -0.0079538, -0.0018961, 0.0016071
5: 0.0096899, 0.0106844, 0.0097647, 0.0107255, -0.0007182, 0.0006087
6: 0.0062202, 0.0100155, 0.0060635, 0.0097298, -0.0023229, 0.0027407
7: 0.9824119, 0.9850676, 0.9823022, 0.9848677, -0.0016254, 0.0019178
8: -0.0054215, -0.0025741, -0.0055391, -0.0027884, -0.0017427, 0.0020562
9: -0.0032993, -0.0014184, -0.0031577, -0.0013407, -0.0013582, 0.0011512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018343, -0.0027943, -0.0018116, -0.0007390, 0.0006585
1: -0.0115838, -0.0089654, -0.0114015, -0.0089078, -0.0018753, 0.0016712
2: 0.0278434, 0.0294679, 0.0279565, 0.0295036, -0.0011635, 0.0010368
3: 0.0045385, 0.0075719, 0.0044718, 0.0073607, -0.0019360, 0.0021725
4: -0.0106757, -0.0080123, -0.0104903, -0.0079538, -0.0019075, 0.0016998
5: 0.0096945, 0.0107033, 0.0097647, 0.0107255, -0.0007225, 0.0006439
6: 0.0061481, 0.0099978, 0.0060635, 0.0097298, -0.0024570, 0.0027572
7: 0.9823614, 0.9850553, 0.9823022, 0.9848677, -0.0017193, 0.0019293
8: -0.0054756, -0.0025874, -0.0055391, -0.0027884, -0.0018433, 0.0020685
9: -0.0032905, -0.0013827, -0.0031577, -0.0013407, -0.0013664, 0.0012176

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0027890, -0.0017796, -0.0007009, 0.0006381
1: -0.0114483, -0.0089602, -0.0113879, -0.0088266, -0.0017786, 0.0016192
2: 0.0279274, 0.0294711, 0.0279649, 0.0295540, -0.0011035, 0.0010046
3: 0.0045325, 0.0074150, 0.0043777, 0.0073450, -0.0018758, 0.0020604
4: -0.0105379, -0.0080071, -0.0104765, -0.0078711, -0.0018091, 0.0016470
5: 0.0097467, 0.0107053, 0.0097700, 0.0107568, -0.0006853, 0.0006239
6: 0.0061405, 0.0097987, 0.0059440, 0.0097098, -0.0023806, 0.0026150
7: 0.9823561, 0.9849159, 0.9822186, 0.9848537, -0.0016659, 0.0018298
8: -0.0054813, -0.0027368, -0.0056287, -0.0028034, -0.0017861, 0.0019619
9: -0.0031918, -0.0013789, -0.0031478, -0.0012815, -0.0012959, 0.0011798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010458
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010477
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0027890, -0.0017796, -0.0006718, 0.0006350
1: -0.0114394, -0.0089014, -0.0113879, -0.0088266, -0.0017047, 0.0016115
2: 0.0279330, 0.0295076, 0.0279649, 0.0295540, -0.0010576, 0.0009998
3: 0.0044644, 0.0074046, 0.0043777, 0.0073450, -0.0018668, 0.0019748
4: -0.0105288, -0.0079472, -0.0104765, -0.0078711, -0.0017340, 0.0016391
5: 0.0097502, 0.0107280, 0.0097700, 0.0107568, -0.0006568, 0.0006209
6: 0.0060540, 0.0097855, 0.0059440, 0.0097098, -0.0023692, 0.0025063
7: 0.9822956, 0.9849067, 0.9822186, 0.9848537, -0.0016579, 0.0017538
8: -0.0055462, -0.0027466, -0.0056287, -0.0028034, -0.0017775, 0.0018804
9: -0.0031853, -0.0013360, -0.0031478, -0.0012815, -0.0012421, 0.0011741

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010502
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010517
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028460, -0.0018046, -0.0006868, 0.0007057
1: -0.0114483, -0.0089602, -0.0115327, -0.0088900, -0.0017429, 0.0017908
2: 0.0279274, 0.0294711, 0.0278751, 0.0295146, -0.0010813, 0.0011110
3: 0.0045325, 0.0074150, 0.0044512, 0.0075126, -0.0020745, 0.0020191
4: -0.0105379, -0.0080071, -0.0106237, -0.0079357, -0.0017728, 0.0018215
5: 0.0097467, 0.0107053, 0.0097142, 0.0107324, -0.0006715, 0.0006899
6: 0.0061405, 0.0097987, 0.0060373, 0.0099226, -0.0026328, 0.0025625
7: 0.9823561, 0.9849159, 0.9822839, 0.9850026, -0.0018423, 0.0017931
8: -0.0054813, -0.0027368, -0.0055587, -0.0026438, -0.0019753, 0.0019225
9: -0.0031918, -0.0013789, -0.0032532, -0.0013278, -0.0012699, 0.0013048

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010328
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010329
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028460, -0.0018046, -0.0006633, 0.0007092
1: -0.0114394, -0.0089014, -0.0115327, -0.0088900, -0.0016831, 0.0017998
2: 0.0279330, 0.0295076, 0.0278751, 0.0295146, -0.0010442, 0.0011166
3: 0.0044644, 0.0074046, 0.0044512, 0.0075126, -0.0020850, 0.0019498
4: -0.0105288, -0.0079472, -0.0106237, -0.0079357, -0.0017120, 0.0018307
5: 0.0097502, 0.0107280, 0.0097142, 0.0107324, -0.0006485, 0.0006934
6: 0.0060540, 0.0097855, 0.0060373, 0.0099226, -0.0026461, 0.0024745
7: 0.9822956, 0.9849067, 0.9822839, 0.9850026, -0.0018516, 0.0017316
8: -0.0055462, -0.0027466, -0.0055587, -0.0026438, -0.0019852, 0.0018565
9: -0.0031853, -0.0013360, -0.0032532, -0.0013278, -0.0012263, 0.0013113

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010377
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010377
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028094, -0.0018055, -0.0007552, 0.0006455
1: -0.0115960, -0.0090144, -0.0114398, -0.0088924, -0.0019165, 0.0016381
2: 0.0278358, 0.0294374, 0.0279327, 0.0295132, -0.0011890, 0.0010163
3: 0.0045954, 0.0075860, 0.0044540, 0.0074050, -0.0018977, 0.0022202
4: -0.0106881, -0.0080622, -0.0105292, -0.0079381, -0.0019495, 0.0016662
5: 0.0096898, 0.0106844, 0.0097500, 0.0107315, -0.0007384, 0.0006311
6: 0.0062202, 0.0100157, 0.0060408, 0.0097860, -0.0024084, 0.0028178
7: 0.9824119, 0.9850678, 0.9822863, 0.9849070, -0.0016853, 0.0019717
8: -0.0054215, -0.0025739, -0.0055561, -0.0027462, -0.0018069, 0.0021140
9: -0.0032994, -0.0014184, -0.0031855, -0.0013295, -0.0013964, 0.0011935

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010314
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010313
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028665, -0.0018343, -0.0028094, -0.0018055, -0.0007318, 0.0006452
1: -0.0115847, -0.0089654, -0.0114398, -0.0088924, -0.0018570, 0.0016373
2: 0.0278428, 0.0294679, 0.0279327, 0.0295132, -0.0011521, 0.0010158
3: 0.0045385, 0.0075729, 0.0044540, 0.0074050, -0.0018968, 0.0021513
4: -0.0106766, -0.0080123, -0.0105292, -0.0079381, -0.0018889, 0.0016654
5: 0.0096942, 0.0107033, 0.0097500, 0.0107315, -0.0007155, 0.0006308
6: 0.0061481, 0.0099991, 0.0060408, 0.0097860, -0.0024072, 0.0027302
7: 0.9823614, 0.9850562, 0.9822863, 0.9849070, -0.0016845, 0.0019105
8: -0.0054756, -0.0025863, -0.0055561, -0.0027462, -0.0018060, 0.0020483
9: -0.0032912, -0.0013827, -0.0031855, -0.0013295, -0.0013530, 0.0011930

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010345
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010345
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0027886, -0.0017913, -0.0007746, 0.0006333
1: -0.0115960, -0.0090144, -0.0113870, -0.0088561, -0.0019657, 0.0016072
2: 0.0278358, 0.0294374, 0.0279654, 0.0295356, -0.0012196, 0.0009971
3: 0.0045954, 0.0075860, 0.0044120, 0.0073439, -0.0018618, 0.0022772
4: -0.0106881, -0.0080622, -0.0104756, -0.0079012, -0.0019995, 0.0016347
5: 0.0096898, 0.0106844, 0.0097703, 0.0107454, -0.0007574, 0.0006192
6: 0.0062202, 0.0100157, 0.0059875, 0.0097085, -0.0023629, 0.0028901
7: 0.9824119, 0.9850678, 0.9822491, 0.9848528, -0.0016534, 0.0020223
8: -0.0054215, -0.0025739, -0.0055961, -0.0028044, -0.0017727, 0.0021683
9: -0.0032994, -0.0014184, -0.0031471, -0.0013031, -0.0014323, 0.0011710

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010272
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010272
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028665, -0.0018343, -0.0027886, -0.0017913, -0.0007456, 0.0006306
1: -0.0115847, -0.0089654, -0.0113870, -0.0088561, -0.0018921, 0.0016002
2: 0.0278428, 0.0294679, 0.0279654, 0.0295356, -0.0011739, 0.0009928
3: 0.0045385, 0.0075729, 0.0044120, 0.0073439, -0.0018537, 0.0021919
4: -0.0106766, -0.0080123, -0.0104756, -0.0079012, -0.0019246, 0.0016276
5: 0.0096942, 0.0107033, 0.0097703, 0.0107454, -0.0007290, 0.0006165
6: 0.0061481, 0.0099991, 0.0059875, 0.0097085, -0.0023526, 0.0027818
7: 0.9823614, 0.9850562, 0.9822491, 0.9848528, -0.0016462, 0.0019466
8: -0.0054756, -0.0025863, -0.0055961, -0.0028044, -0.0017650, 0.0020871
9: -0.0032912, -0.0013827, -0.0031471, -0.0013031, -0.0013786, 0.0011659

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010315
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010315
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0027952, -0.0017907, -0.0006604, 0.0006575
1: -0.0114011, -0.0089103, -0.0114038, -0.0088547, -0.0016758, 0.0016685
2: 0.0279567, 0.0295020, 0.0279551, 0.0295365, -0.0010397, 0.0010351
3: 0.0044748, 0.0073603, 0.0044103, 0.0073633, -0.0019328, 0.0019413
4: -0.0104899, -0.0079563, -0.0104926, -0.0078997, -0.0017046, 0.0016971
5: 0.0097649, 0.0107245, 0.0097639, 0.0107460, -0.0006456, 0.0006428
6: 0.0060672, 0.0097292, 0.0059854, 0.0097331, -0.0024530, 0.0024638
7: 0.9823048, 0.9848673, 0.9822476, 0.9848701, -0.0017165, 0.0017240
8: -0.0055363, -0.0027888, -0.0055976, -0.0027859, -0.0018404, 0.0018485
9: -0.0031574, -0.0013426, -0.0031593, -0.0013020, -0.0012210, 0.0012157

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010448
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010675
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0027952, -0.0017907, -0.0006719, 0.0006928
1: -0.0113867, -0.0088574, -0.0114038, -0.0088547, -0.0017051, 0.0017580
2: 0.0279657, 0.0295349, 0.0279551, 0.0295365, -0.0010579, 0.0010906
3: 0.0044134, 0.0073435, 0.0044103, 0.0073633, -0.0020365, 0.0019753
4: -0.0104752, -0.0079025, -0.0104926, -0.0078997, -0.0017344, 0.0017881
5: 0.0097705, 0.0107449, 0.0097639, 0.0107460, -0.0006569, 0.0006773
6: 0.0059893, 0.0097080, 0.0059854, 0.0097331, -0.0025846, 0.0025069
7: 0.9822503, 0.9848524, 0.9822476, 0.9848701, -0.0018086, 0.0017542
8: -0.0055947, -0.0028048, -0.0055976, -0.0027859, -0.0019391, 0.0018808
9: -0.0031469, -0.0013040, -0.0031593, -0.0013020, -0.0012424, 0.0012809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010448
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010675
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028524, -0.0018128, -0.0006476, 0.0007359
1: -0.0114011, -0.0089103, -0.0115490, -0.0089107, -0.0016433, 0.0018675
2: 0.0279567, 0.0295020, 0.0278650, 0.0295018, -0.0010195, 0.0011586
3: 0.0044748, 0.0073603, 0.0044752, 0.0075315, -0.0021634, 0.0019037
4: -0.0104899, -0.0079563, -0.0106403, -0.0079567, -0.0016716, 0.0018996
5: 0.0097649, 0.0107245, 0.0097079, 0.0107244, -0.0006331, 0.0007195
6: 0.0060672, 0.0097292, 0.0060678, 0.0099466, -0.0027457, 0.0024161
7: 0.9823048, 0.9848673, 0.9823052, 0.9850194, -0.0019213, 0.0016907
8: -0.0055363, -0.0027888, -0.0055358, -0.0026258, -0.0020599, 0.0018126
9: -0.0031574, -0.0013426, -0.0032651, -0.0013429, -0.0011974, 0.0013607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010337
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010423
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028524, -0.0018128, -0.0006586, 0.0007712
1: -0.0113867, -0.0088574, -0.0115490, -0.0089107, -0.0016714, 0.0019570
2: 0.0279657, 0.0295349, 0.0278650, 0.0295018, -0.0010369, 0.0012141
3: 0.0044134, 0.0073435, 0.0044752, 0.0075315, -0.0022671, 0.0019362
4: -0.0104752, -0.0079025, -0.0106403, -0.0079567, -0.0017001, 0.0019906
5: 0.0097705, 0.0107449, 0.0097079, 0.0107244, -0.0006439, 0.0007540
6: 0.0059893, 0.0097080, 0.0060678, 0.0099466, -0.0028772, 0.0024573
7: 0.9822503, 0.9848524, 0.9823052, 0.9850194, -0.0020134, 0.0017195
8: -0.0055947, -0.0028048, -0.0055358, -0.0026258, -0.0021586, 0.0018436
9: -0.0031469, -0.0013040, -0.0032651, -0.0013429, -0.0012178, 0.0014259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010337
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010423
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028129, -0.0018279, -0.0007041, 0.0006562
1: -0.0115469, -0.0089699, -0.0114487, -0.0089490, -0.0017866, 0.0016651
2: 0.0278663, 0.0294651, 0.0279272, 0.0294780, -0.0011084, 0.0010331
3: 0.0045438, 0.0075291, 0.0045196, 0.0074154, -0.0019290, 0.0020697
4: -0.0106381, -0.0080169, -0.0105383, -0.0079957, -0.0018173, 0.0016937
5: 0.0097087, 0.0107016, 0.0097466, 0.0107096, -0.0006883, 0.0006415
6: 0.0061547, 0.0099435, 0.0061241, 0.0097992, -0.0024481, 0.0026268
7: 0.9823660, 0.9850173, 0.9823446, 0.9849162, -0.0017131, 0.0018381
8: -0.0054706, -0.0026281, -0.0054936, -0.0027364, -0.0018367, 0.0019707
9: -0.0032636, -0.0013860, -0.0031921, -0.0013708, -0.0013018, 0.0012132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028129, -0.0018279, -0.0007101, 0.0006974
1: -0.0115328, -0.0089255, -0.0114487, -0.0089490, -0.0018019, 0.0017698
2: 0.0278750, 0.0294926, 0.0279272, 0.0294780, -0.0011179, 0.0010980
3: 0.0044924, 0.0075128, 0.0045196, 0.0074154, -0.0020502, 0.0020874
4: -0.0106238, -0.0079718, -0.0105383, -0.0079957, -0.0018329, 0.0018002
5: 0.0097142, 0.0107187, 0.0097466, 0.0107096, -0.0006942, 0.0006819
6: 0.0060895, 0.0099228, 0.0061241, 0.0097992, -0.0026020, 0.0026492
7: 0.9823204, 0.9850028, 0.9823446, 0.9849162, -0.0018207, 0.0018538
8: -0.0055195, -0.0026436, -0.0054936, -0.0027364, -0.0019521, 0.0019876
9: -0.0032533, -0.0013536, -0.0031921, -0.0013708, -0.0013129, 0.0012895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0027943, -0.0018116, -0.0007402, 0.0006518
1: -0.0115469, -0.0089699, -0.0114015, -0.0089078, -0.0018783, 0.0016541
2: 0.0278663, 0.0294651, 0.0279565, 0.0295036, -0.0011653, 0.0010262
3: 0.0045438, 0.0075291, 0.0044718, 0.0073607, -0.0019162, 0.0021759
4: -0.0106381, -0.0080169, -0.0104903, -0.0079538, -0.0019105, 0.0016825
5: 0.0097087, 0.0107016, 0.0097647, 0.0107255, -0.0007236, 0.0006373
6: 0.0061547, 0.0099435, 0.0060635, 0.0097298, -0.0024319, 0.0027615
7: 0.9823660, 0.9850173, 0.9823022, 0.9848677, -0.0017017, 0.0019323
8: -0.0054706, -0.0026281, -0.0055391, -0.0027884, -0.0018245, 0.0020718
9: -0.0032636, -0.0013860, -0.0031577, -0.0013407, -0.0013685, 0.0012052

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010356
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010357
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0027943, -0.0018116, -0.0007450, 0.0006869
1: -0.0115328, -0.0089255, -0.0114015, -0.0089078, -0.0018906, 0.0017430
2: 0.0278750, 0.0294926, 0.0279565, 0.0295036, -0.0011729, 0.0010814
3: 0.0044924, 0.0075128, 0.0044718, 0.0073607, -0.0020192, 0.0021902
4: -0.0106238, -0.0079718, -0.0104903, -0.0079538, -0.0019231, 0.0017730
5: 0.0097142, 0.0107187, 0.0097647, 0.0107255, -0.0007284, 0.0006715
6: 0.0060895, 0.0099228, 0.0060635, 0.0097298, -0.0025627, 0.0027796
7: 0.9823204, 0.9850028, 0.9823022, 0.9848677, -0.0017932, 0.0019450
8: -0.0055195, -0.0026436, -0.0055391, -0.0027884, -0.0019226, 0.0020854
9: -0.0032533, -0.0013536, -0.0031577, -0.0013407, -0.0013775, 0.0012700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010357
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010357
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0027896, -0.0017695, -0.0006995, 0.0006656
1: -0.0114011, -0.0089103, -0.0113896, -0.0088009, -0.0017750, 0.0016890
2: 0.0279567, 0.0295020, 0.0279639, 0.0295699, -0.0011012, 0.0010479
3: 0.0044748, 0.0073603, 0.0043480, 0.0073469, -0.0019566, 0.0020563
4: -0.0104899, -0.0079563, -0.0104782, -0.0078450, -0.0018055, 0.0017180
5: 0.0097649, 0.0107245, 0.0097693, 0.0107667, -0.0006839, 0.0006507
6: 0.0060672, 0.0097292, 0.0059063, 0.0097123, -0.0024832, 0.0026097
7: 0.9823048, 0.9848673, 0.9821922, 0.9848555, -0.0017376, 0.0018261
8: -0.0055363, -0.0027888, -0.0056570, -0.0028016, -0.0018630, 0.0019579
9: -0.0031574, -0.0013426, -0.0031490, -0.0012629, -0.0012933, 0.0012306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010458
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010685
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0027896, -0.0017695, -0.0006688, 0.0006657
1: -0.0113867, -0.0088574, -0.0113896, -0.0088009, -0.0016972, 0.0016892
2: 0.0279657, 0.0295349, 0.0279639, 0.0295699, -0.0010529, 0.0010480
3: 0.0044134, 0.0073435, 0.0043480, 0.0073469, -0.0019569, 0.0019661
4: -0.0104752, -0.0079025, -0.0104782, -0.0078450, -0.0017263, 0.0017182
5: 0.0097705, 0.0107449, 0.0097693, 0.0107667, -0.0006539, 0.0006508
6: 0.0059893, 0.0097080, 0.0059063, 0.0097123, -0.0024835, 0.0024953
7: 0.9822503, 0.9848524, 0.9821922, 0.9848555, -0.0017378, 0.0017461
8: -0.0055947, -0.0028048, -0.0056570, -0.0028016, -0.0018632, 0.0018721
9: -0.0031469, -0.0013040, -0.0031490, -0.0012629, -0.0012366, 0.0012308

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010501
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010704
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028472, -0.0017952, -0.0006839, 0.0007397
1: -0.0114011, -0.0089103, -0.0115357, -0.0088662, -0.0017354, 0.0018770
2: 0.0279567, 0.0295020, 0.0278733, 0.0295294, -0.0010766, 0.0011645
3: 0.0044748, 0.0073603, 0.0044236, 0.0075161, -0.0021744, 0.0020103
4: -0.0104899, -0.0079563, -0.0106267, -0.0079114, -0.0017652, 0.0019092
5: 0.0097649, 0.0107245, 0.0097131, 0.0107415, -0.0006686, 0.0007232
6: 0.0060672, 0.0097292, 0.0060023, 0.0099270, -0.0027596, 0.0025514
7: 0.9823048, 0.9848673, 0.9822594, 0.9850056, -0.0019311, 0.0017853
8: -0.0055363, -0.0027888, -0.0055850, -0.0026405, -0.0020704, 0.0019142
9: -0.0031574, -0.0013426, -0.0032554, -0.0013104, -0.0012644, 0.0013676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010332
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010417
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028472, -0.0017952, -0.0006588, 0.0007454
1: -0.0113867, -0.0088574, -0.0115357, -0.0088662, -0.0016717, 0.0018915
2: 0.0279657, 0.0295349, 0.0278733, 0.0295294, -0.0010371, 0.0011735
3: 0.0044134, 0.0073435, 0.0044236, 0.0075161, -0.0021912, 0.0019366
4: -0.0104752, -0.0079025, -0.0106267, -0.0079114, -0.0017004, 0.0019240
5: 0.0097705, 0.0107449, 0.0097131, 0.0107415, -0.0006441, 0.0007288
6: 0.0059893, 0.0097080, 0.0060023, 0.0099270, -0.0027809, 0.0024577
7: 0.9822503, 0.9848524, 0.9822594, 0.9850056, -0.0019460, 0.0017198
8: -0.0055947, -0.0028048, -0.0055850, -0.0026405, -0.0020864, 0.0018439
9: -0.0031469, -0.0013040, -0.0032554, -0.0013104, -0.0012180, 0.0013782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 105
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 105

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010382
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010458
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028094, -0.0018055, -0.0007385, 0.0006647
1: -0.0115469, -0.0089699, -0.0114398, -0.0088924, -0.0018740, 0.0016868
2: 0.0278663, 0.0294651, 0.0279327, 0.0295132, -0.0011626, 0.0010465
3: 0.0045438, 0.0075291, 0.0044540, 0.0074050, -0.0019541, 0.0021709
4: -0.0106381, -0.0080169, -0.0105292, -0.0079381, -0.0019061, 0.0017158
5: 0.0097087, 0.0107016, 0.0097500, 0.0107315, -0.0007220, 0.0006499
6: 0.0061547, 0.0099435, 0.0060408, 0.0097860, -0.0024800, 0.0027551
7: 0.9823660, 0.9850173, 0.9822863, 0.9849070, -0.0017354, 0.0019279
8: -0.0054706, -0.0026281, -0.0055561, -0.0027462, -0.0018606, 0.0020670
9: -0.0032636, -0.0013860, -0.0031855, -0.0013295, -0.0013654, 0.0012291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010314
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010313
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028094, -0.0018055, -0.0007148, 0.0006643
1: -0.0115328, -0.0089255, -0.0114398, -0.0088924, -0.0018140, 0.0016857
2: 0.0278750, 0.0294926, 0.0279327, 0.0295132, -0.0011254, 0.0010458
3: 0.0044924, 0.0075128, 0.0044540, 0.0074050, -0.0019528, 0.0021014
4: -0.0106238, -0.0079718, -0.0105292, -0.0079381, -0.0018451, 0.0017146
5: 0.0097142, 0.0107187, 0.0097500, 0.0107315, -0.0006989, 0.0006495
6: 0.0060895, 0.0099228, 0.0060408, 0.0097860, -0.0024783, 0.0026670
7: 0.9823204, 0.9850028, 0.9822863, 0.9849070, -0.0017342, 0.0018662
8: -0.0055195, -0.0026436, -0.0055561, -0.0027462, -0.0018594, 0.0020009
9: -0.0032533, -0.0013536, -0.0031855, -0.0013295, -0.0013217, 0.0012282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010345
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010345
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0027886, -0.0017913, -0.0007735, 0.0006605
1: -0.0115469, -0.0089699, -0.0113871, -0.0088561, -0.0019630, 0.0016761
2: 0.0278663, 0.0294651, 0.0279654, 0.0295356, -0.0012178, 0.0010398
3: 0.0045438, 0.0075291, 0.0044120, 0.0073439, -0.0019416, 0.0022740
4: -0.0106381, -0.0080169, -0.0104756, -0.0079012, -0.0019967, 0.0017048
5: 0.0097087, 0.0107016, 0.0097703, 0.0107454, -0.0007563, 0.0006457
6: 0.0061547, 0.0099435, 0.0059875, 0.0097085, -0.0024642, 0.0028860
7: 0.9823660, 0.9850173, 0.9822491, 0.9848528, -0.0017243, 0.0020195
8: -0.0054706, -0.0026281, -0.0055961, -0.0028044, -0.0018487, 0.0021652
9: -0.0032636, -0.0013860, -0.0031471, -0.0013031, -0.0014303, 0.0012212

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010362
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010362
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0027886, -0.0017913, -0.0007522, 0.0006590
1: -0.0115328, -0.0089255, -0.0113871, -0.0088561, -0.0019087, 0.0016722
2: 0.0278750, 0.0294926, 0.0279654, 0.0295356, -0.0011842, 0.0010375
3: 0.0044924, 0.0075128, 0.0044120, 0.0073439, -0.0019372, 0.0022112
4: -0.0106238, -0.0079718, -0.0104756, -0.0079012, -0.0019415, 0.0017009
5: 0.0097142, 0.0107187, 0.0097703, 0.0107454, -0.0007354, 0.0006443
6: 0.0060895, 0.0099228, 0.0059875, 0.0097085, -0.0024586, 0.0028062
7: 0.9823204, 0.9850028, 0.9822491, 0.9848528, -0.0017204, 0.0019637
8: -0.0055195, -0.0026436, -0.0055961, -0.0028044, -0.0018445, 0.0021054
9: -0.0032533, -0.0013536, -0.0031471, -0.0013031, -0.0013907, 0.0012184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 140

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010389
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010390
time: 0.85 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.73 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010448
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010467
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010448
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010467
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010336
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010337
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010336
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010337
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010278
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010269
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010458
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010477
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010502
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010460, upper bound: 0.0010517
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010328
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010329
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010377
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010283, upper bound: 0.0010377
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010314
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010313
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010345
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010345
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010272
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010272
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010315
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010357, upper bound: 0.0010315
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010448
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010675
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010448
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010675
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010337
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010423
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010337
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010423
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010278
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010356
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010357
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010357
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010357
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010458
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010685
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010501
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010450, upper bound: 0.0010704
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010332
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010417
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010382
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010286, upper bound: 0.0010458
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010314
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010313
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010345
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010345
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010362
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010362
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010389
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.73
Output dim: 7, lower bound: -0.0010407, upper bound: 0.0010390

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028128, -0.0018323, -0.0006345, 0.0006345
1: -0.0114483, -0.0089602, -0.0114483, -0.0089602, -0.0016102, 0.0016102
2: 0.0279274, 0.0294711, 0.0279274, 0.0294711, -0.0009990, 0.0009990
3: 0.0045325, 0.0074150, 0.0045325, 0.0074150, -0.0018653, 0.0018653
4: -0.0105379, -0.0080071, -0.0105379, -0.0080071, -0.0016378, 0.0016378
5: 0.0097467, 0.0107053, 0.0097467, 0.0107053, -0.0006204, 0.0006204
6: 0.0061405, 0.0097987, 0.0061405, 0.0097987, -0.0023674, 0.0023674
7: 0.9823561, 0.9849159, 0.9823561, 0.9849159, -0.0016566, 0.0016566
8: -0.0054813, -0.0027368, -0.0054813, -0.0027368, -0.0017761, 0.0017761
9: -0.0031918, -0.0013789, -0.0031918, -0.0013789, -0.0011732, 0.0011732

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010003
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010188
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0027942, -0.0018126, -0.0006530, 0.0006201
1: -0.0114483, -0.0089602, -0.0114011, -0.0089103, -0.0016571, 0.0015737
2: 0.0279274, 0.0294711, 0.0279567, 0.0295020, -0.0010281, 0.0009763
3: 0.0045325, 0.0074150, 0.0044748, 0.0073603, -0.0018230, 0.0019197
4: -0.0105379, -0.0080071, -0.0104899, -0.0079563, -0.0016856, 0.0016007
5: 0.0097467, 0.0107053, 0.0097649, 0.0107245, -0.0006385, 0.0006063
6: 0.0061405, 0.0097987, 0.0060672, 0.0097292, -0.0023137, 0.0024364
7: 0.9823561, 0.9849159, 0.9823048, 0.9848673, -0.0016190, 0.0017049
8: -0.0054813, -0.0027368, -0.0055363, -0.0027888, -0.0017358, 0.0018279
9: -0.0031918, -0.0013789, -0.0031574, -0.0013426, -0.0012074, 0.0011466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010013
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010205
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028128, -0.0018323, -0.0006431, 0.0006705
1: -0.0114394, -0.0089014, -0.0114483, -0.0089602, -0.0016319, 0.0017015
2: 0.0279330, 0.0295076, 0.0279274, 0.0294711, -0.0010124, 0.0010556
3: 0.0044644, 0.0074046, 0.0045325, 0.0074150, -0.0019711, 0.0018905
4: -0.0105288, -0.0079472, -0.0105379, -0.0080071, -0.0016599, 0.0017307
5: 0.0097502, 0.0107280, 0.0097467, 0.0107053, -0.0006287, 0.0006555
6: 0.0060540, 0.0097855, 0.0061405, 0.0097987, -0.0025016, 0.0023992
7: 0.9822956, 0.9849067, 0.9823561, 0.9849159, -0.0017505, 0.0016789
8: -0.0055462, -0.0027466, -0.0054813, -0.0027368, -0.0018768, 0.0018000
9: -0.0031853, -0.0013360, -0.0031918, -0.0013789, -0.0011890, 0.0012397

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010340, upper bound: 0.0010001
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010265, upper bound: 0.0010188
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0027942, -0.0018126, -0.0006616, 0.0006561
1: -0.0114394, -0.0089014, -0.0114011, -0.0089103, -0.0016788, 0.0016650
2: 0.0279330, 0.0295076, 0.0279567, 0.0295020, -0.0010416, 0.0010330
3: 0.0044644, 0.0074046, 0.0044748, 0.0073603, -0.0019288, 0.0019448
4: -0.0105288, -0.0079472, -0.0104899, -0.0079563, -0.0017076, 0.0016936
5: 0.0097502, 0.0107280, 0.0097649, 0.0107245, -0.0006468, 0.0006415
6: 0.0060540, 0.0097855, 0.0060672, 0.0097292, -0.0024479, 0.0024683
7: 0.9822956, 0.9849067, 0.9823048, 0.9848673, -0.0017129, 0.0017272
8: -0.0055462, -0.0027466, -0.0055363, -0.0027888, -0.0018365, 0.0018518
9: -0.0031853, -0.0013360, -0.0031574, -0.0013426, -0.0012232, 0.0012131

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010340, upper bound: 0.0010010
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010265, upper bound: 0.0010205
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028709, -0.0018536, -0.0006244, 0.0007114
1: -0.0114483, -0.0089602, -0.0115958, -0.0090144, -0.0015845, 0.0018052
2: 0.0279274, 0.0294711, 0.0278359, 0.0294374, -0.0009831, 0.0011200
3: 0.0045325, 0.0074150, 0.0045954, 0.0075858, -0.0020912, 0.0018356
4: -0.0105379, -0.0080071, -0.0106879, -0.0080622, -0.0016117, 0.0018362
5: 0.0097467, 0.0107053, 0.0096899, 0.0106844, -0.0006105, 0.0006955
6: 0.0061405, 0.0097987, 0.0062202, 0.0100155, -0.0026541, 0.0023296
7: 0.9823561, 0.9849159, 0.9824119, 0.9850676, -0.0018572, 0.0016302
8: -0.0054813, -0.0027368, -0.0054215, -0.0025741, -0.0019912, 0.0017478
9: -0.0031918, -0.0013789, -0.0032993, -0.0014184, -0.0011545, 0.0013153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0009846
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010079
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028514, -0.0018361, -0.0006384, 0.0006940
1: -0.0114483, -0.0089602, -0.0115463, -0.0089699, -0.0016199, 0.0017612
2: 0.0279274, 0.0294711, 0.0278666, 0.0294651, -0.0010050, 0.0010927
3: 0.0045325, 0.0074150, 0.0045438, 0.0075285, -0.0020403, 0.0018766
4: -0.0105379, -0.0080071, -0.0106376, -0.0080169, -0.0016477, 0.0017915
5: 0.0097467, 0.0107053, 0.0097090, 0.0107016, -0.0006241, 0.0006786
6: 0.0061405, 0.0097987, 0.0061547, 0.0099427, -0.0025894, 0.0023817
7: 0.9823561, 0.9849159, 0.9823660, 0.9850166, -0.0018120, 0.0016666
8: -0.0054813, -0.0027368, -0.0054706, -0.0026287, -0.0019427, 0.0017868
9: -0.0031918, -0.0013789, -0.0032632, -0.0013860, -0.0011803, 0.0012833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0009853
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010079
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028709, -0.0018536, -0.0006330, 0.0007473
1: -0.0114394, -0.0089014, -0.0115958, -0.0090144, -0.0016062, 0.0018965
2: 0.0279330, 0.0295076, 0.0278359, 0.0294374, -0.0009965, 0.0011766
3: 0.0044644, 0.0074046, 0.0045954, 0.0075858, -0.0021970, 0.0018607
4: -0.0105288, -0.0079472, -0.0106879, -0.0080622, -0.0016338, 0.0019291
5: 0.0097502, 0.0107280, 0.0096899, 0.0106844, -0.0006188, 0.0007307
6: 0.0060540, 0.0097855, 0.0062202, 0.0100155, -0.0027883, 0.0023615
7: 0.9822956, 0.9849067, 0.9824119, 0.9850676, -0.0019511, 0.0016525
8: -0.0055462, -0.0027466, -0.0054215, -0.0025741, -0.0020919, 0.0017717
9: -0.0031853, -0.0013360, -0.0032993, -0.0014184, -0.0011703, 0.0013818

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009843
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010078, upper bound: 0.0010079
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028514, -0.0018361, -0.0006469, 0.0007300
1: -0.0114394, -0.0089014, -0.0115463, -0.0089699, -0.0016416, 0.0018525
2: 0.0279330, 0.0295076, 0.0278666, 0.0294651, -0.0010185, 0.0011493
3: 0.0044644, 0.0074046, 0.0045438, 0.0075285, -0.0021461, 0.0019017
4: -0.0105288, -0.0079472, -0.0106376, -0.0080169, -0.0016698, 0.0018843
5: 0.0097502, 0.0107280, 0.0097090, 0.0107016, -0.0006325, 0.0007137
6: 0.0060540, 0.0097855, 0.0061547, 0.0099427, -0.0027236, 0.0024135
7: 0.9822956, 0.9849067, 0.9823660, 0.9850166, -0.0019059, 0.0016889
8: -0.0055462, -0.0027466, -0.0054706, -0.0026287, -0.0020434, 0.0018108
9: -0.0031853, -0.0013360, -0.0032632, -0.0013860, -0.0011961, 0.0013498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009848
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010078, upper bound: 0.0010079
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028128, -0.0018323, -0.0007114, 0.0006244
1: -0.0115958, -0.0090144, -0.0114483, -0.0089602, -0.0018052, 0.0015845
2: 0.0278359, 0.0294374, 0.0279274, 0.0294711, -0.0011200, 0.0009831
3: 0.0045954, 0.0075858, 0.0045325, 0.0074150, -0.0018356, 0.0020912
4: -0.0106879, -0.0080622, -0.0105379, -0.0080071, -0.0018362, 0.0016117
5: 0.0096899, 0.0106844, 0.0097467, 0.0107053, -0.0006955, 0.0006105
6: 0.0062202, 0.0100155, 0.0061405, 0.0097987, -0.0023296, 0.0026541
7: 0.9824119, 0.9850676, 0.9823561, 0.9849159, -0.0016302, 0.0018572
8: -0.0054215, -0.0025741, -0.0054813, -0.0027368, -0.0017478, 0.0019912
9: -0.0032993, -0.0014184, -0.0031918, -0.0013789, -0.0013153, 0.0011545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009740
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010027
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028709, -0.0018536, -0.0006427, 0.0006427
1: -0.0115958, -0.0090144, -0.0115958, -0.0090144, -0.0016309, 0.0016309
2: 0.0278359, 0.0294374, 0.0278359, 0.0294374, -0.0010118, 0.0010118
3: 0.0045954, 0.0075858, 0.0045954, 0.0075858, -0.0018893, 0.0018893
4: -0.0106879, -0.0080622, -0.0106879, -0.0080622, -0.0016589, 0.0016589
5: 0.0096899, 0.0106844, 0.0096899, 0.0106844, -0.0006283, 0.0006283
6: 0.0062202, 0.0100155, 0.0062202, 0.0100155, -0.0023977, 0.0023977
7: 0.9824119, 0.9850676, 0.9824119, 0.9850676, -0.0016778, 0.0016778
8: -0.0054215, -0.0025741, -0.0054215, -0.0025741, -0.0017989, 0.0017989
9: -0.0032993, -0.0014184, -0.0032993, -0.0014184, -0.0011883, 0.0011883

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009740
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010027
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018343, -0.0028128, -0.0018323, -0.0007158, 0.0006572
1: -0.0115838, -0.0089654, -0.0114483, -0.0089602, -0.0018164, 0.0016677
2: 0.0278434, 0.0294679, 0.0279274, 0.0294711, -0.0011269, 0.0010347
3: 0.0045385, 0.0075719, 0.0045325, 0.0074150, -0.0019320, 0.0021042
4: -0.0106757, -0.0080123, -0.0105379, -0.0080071, -0.0018476, 0.0016964
5: 0.0096945, 0.0107033, 0.0097467, 0.0107053, -0.0006998, 0.0006425
6: 0.0061481, 0.0099978, 0.0061405, 0.0097987, -0.0024520, 0.0026705
7: 0.9823614, 0.9850553, 0.9823561, 0.9849159, -0.0017158, 0.0018687
8: -0.0054756, -0.0025874, -0.0054813, -0.0027368, -0.0018396, 0.0020036
9: -0.0032905, -0.0013827, -0.0031918, -0.0013789, -0.0013235, 0.0012151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009734
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010144, upper bound: 0.0010027
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018343, -0.0028709, -0.0018536, -0.0006510, 0.0006786
1: -0.0115838, -0.0089654, -0.0115958, -0.0090144, -0.0016519, 0.0017221
2: 0.0278434, 0.0294679, 0.0278359, 0.0294374, -0.0010249, 0.0010684
3: 0.0045385, 0.0075719, 0.0045954, 0.0075858, -0.0019949, 0.0019137
4: -0.0106757, -0.0080123, -0.0106879, -0.0080622, -0.0016803, 0.0017516
5: 0.0096945, 0.0107033, 0.0096899, 0.0106844, -0.0006365, 0.0006635
6: 0.0061481, 0.0099978, 0.0062202, 0.0100155, -0.0025318, 0.0024287
7: 0.9823614, 0.9850553, 0.9824119, 0.9850676, -0.0017717, 0.0016995
8: -0.0054756, -0.0025874, -0.0054215, -0.0025741, -0.0018995, 0.0018221
9: -0.0032905, -0.0013827, -0.0032993, -0.0014184, -0.0012036, 0.0012547

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009734
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010144, upper bound: 0.0010027
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0027942, -0.0018126, -0.0007299, 0.0006100
1: -0.0115958, -0.0090144, -0.0114011, -0.0089103, -0.0018522, 0.0015480
2: 0.0278359, 0.0294374, 0.0279567, 0.0295020, -0.0011491, 0.0009604
3: 0.0045954, 0.0075858, 0.0044748, 0.0073603, -0.0017933, 0.0021456
4: -0.0106879, -0.0080622, -0.0104899, -0.0079563, -0.0018840, 0.0015746
5: 0.0096899, 0.0106844, 0.0097649, 0.0107245, -0.0007136, 0.0005964
6: 0.0062202, 0.0100155, 0.0060672, 0.0097292, -0.0022760, 0.0027231
7: 0.9824119, 0.9850676, 0.9823048, 0.9848673, -0.0015926, 0.0019055
8: -0.0054215, -0.0025741, -0.0055363, -0.0027888, -0.0017075, 0.0020430
9: -0.0032993, -0.0014184, -0.0031574, -0.0013426, -0.0013495, 0.0011279

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009728
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010007
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028514, -0.0018361, -0.0006619, 0.0006282
1: -0.0115958, -0.0090144, -0.0115463, -0.0089699, -0.0016796, 0.0015942
2: 0.0278359, 0.0294374, 0.0278666, 0.0294651, -0.0010420, 0.0009890
3: 0.0045954, 0.0075858, 0.0045438, 0.0075285, -0.0018468, 0.0019458
4: -0.0106879, -0.0080622, -0.0106376, -0.0080169, -0.0017085, 0.0016216
5: 0.0096899, 0.0106844, 0.0097090, 0.0107016, -0.0006471, 0.0006142
6: 0.0062202, 0.0100155, 0.0061547, 0.0099427, -0.0023438, 0.0024694
7: 0.9824119, 0.9850676, 0.9823660, 0.9850166, -0.0016401, 0.0017280
8: -0.0054215, -0.0025741, -0.0054706, -0.0026287, -0.0017584, 0.0018527
9: -0.0032993, -0.0014184, -0.0032632, -0.0013860, -0.0012238, 0.0011616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009728
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010007
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018343, -0.0027942, -0.0018126, -0.0007343, 0.0006428
1: -0.0115838, -0.0089654, -0.0114011, -0.0089103, -0.0018634, 0.0016312
2: 0.0278434, 0.0294679, 0.0279567, 0.0295020, -0.0011560, 0.0010120
3: 0.0045385, 0.0075719, 0.0044748, 0.0073603, -0.0018897, 0.0021586
4: -0.0106757, -0.0080123, -0.0104899, -0.0079563, -0.0018954, 0.0016592
5: 0.0096945, 0.0107033, 0.0097649, 0.0107245, -0.0007179, 0.0006285
6: 0.0061481, 0.0099978, 0.0060672, 0.0097292, -0.0023983, 0.0027396
7: 0.9823614, 0.9850553, 0.9823048, 0.9848673, -0.0016782, 0.0019170
8: -0.0054756, -0.0025874, -0.0055363, -0.0027888, -0.0017993, 0.0020553
9: -0.0032905, -0.0013827, -0.0031574, -0.0013426, -0.0013577, 0.0011885

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009721
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010143, upper bound: 0.0010007
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018343, -0.0028514, -0.0018361, -0.0006702, 0.0006642
1: -0.0115838, -0.0089654, -0.0115463, -0.0089699, -0.0017007, 0.0016854
2: 0.0278434, 0.0294679, 0.0278666, 0.0294651, -0.0010551, 0.0010456
3: 0.0045385, 0.0075719, 0.0045438, 0.0075285, -0.0019525, 0.0019702
4: -0.0106757, -0.0080123, -0.0106376, -0.0080169, -0.0017299, 0.0017143
5: 0.0096945, 0.0107033, 0.0097090, 0.0107016, -0.0006552, 0.0006493
6: 0.0061481, 0.0099978, 0.0061547, 0.0099427, -0.0024779, 0.0025004
7: 0.9823614, 0.9850553, 0.9823660, 0.9850166, -0.0017339, 0.0017497
8: -0.0054756, -0.0025874, -0.0054706, -0.0026287, -0.0018591, 0.0018759
9: -0.0032905, -0.0013827, -0.0032632, -0.0013860, -0.0012392, 0.0012280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009721
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010143, upper bound: 0.0010007
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028092, -0.0018091, -0.0006705, 0.0006431
1: -0.0114483, -0.0089602, -0.0114394, -0.0089014, -0.0017015, 0.0016319
2: 0.0279274, 0.0294711, 0.0279330, 0.0295076, -0.0010556, 0.0010124
3: 0.0045325, 0.0074150, 0.0044644, 0.0074046, -0.0018905, 0.0019711
4: -0.0105379, -0.0080071, -0.0105288, -0.0079472, -0.0017307, 0.0016599
5: 0.0097467, 0.0107053, 0.0097502, 0.0107280, -0.0006555, 0.0006287
6: 0.0061405, 0.0097987, 0.0060540, 0.0097855, -0.0023992, 0.0025016
7: 0.9823561, 0.9849159, 0.9822956, 0.9849067, -0.0016789, 0.0017505
8: -0.0054813, -0.0027368, -0.0055462, -0.0027466, -0.0018000, 0.0018768
9: -0.0031918, -0.0013789, -0.0031853, -0.0013360, -0.0012397, 0.0011890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010091
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010215
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0027885, -0.0017917, -0.0006939, 0.0006308
1: -0.0114483, -0.0089602, -0.0113867, -0.0088574, -0.0017609, 0.0016008
2: 0.0279274, 0.0294711, 0.0279657, 0.0295349, -0.0010925, 0.0009932
3: 0.0045325, 0.0074150, 0.0044134, 0.0073435, -0.0018545, 0.0020399
4: -0.0105379, -0.0080071, -0.0104752, -0.0079025, -0.0017911, 0.0016283
5: 0.0097467, 0.0107053, 0.0097705, 0.0107449, -0.0006784, 0.0006168
6: 0.0061405, 0.0097987, 0.0059893, 0.0097080, -0.0023536, 0.0025889
7: 0.9823561, 0.9849159, 0.9822503, 0.9848524, -0.0016469, 0.0018116
8: -0.0054813, -0.0027368, -0.0055947, -0.0028048, -0.0017658, 0.0019423
9: -0.0031918, -0.0013789, -0.0031469, -0.0013040, -0.0012830, 0.0011664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010110
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010231
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028092, -0.0018091, -0.0006434, 0.0006434
1: -0.0114394, -0.0089014, -0.0114394, -0.0089014, -0.0016328, 0.0016328
2: 0.0279330, 0.0295076, 0.0279330, 0.0295076, -0.0010130, 0.0010130
3: 0.0044644, 0.0074046, 0.0044644, 0.0074046, -0.0018916, 0.0018916
4: -0.0105288, -0.0079472, -0.0105288, -0.0079472, -0.0016609, 0.0016609
5: 0.0097502, 0.0107280, 0.0097502, 0.0107280, -0.0006291, 0.0006291
6: 0.0060540, 0.0097855, 0.0060540, 0.0097855, -0.0024006, 0.0024006
7: 0.9822956, 0.9849067, 0.9822956, 0.9849067, -0.0016798, 0.0016798
8: -0.0055462, -0.0027466, -0.0055462, -0.0027466, -0.0018011, 0.0018011
9: -0.0031853, -0.0013360, -0.0031853, -0.0013360, -0.0011897, 0.0011897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010398, upper bound: 0.0010066
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010327, upper bound: 0.0010250
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0027885, -0.0017917, -0.0006620, 0.0006288
1: -0.0114394, -0.0089014, -0.0113867, -0.0088574, -0.0016798, 0.0015957
2: 0.0279330, 0.0295076, 0.0279657, 0.0295349, -0.0010422, 0.0009900
3: 0.0044644, 0.0074046, 0.0044134, 0.0073435, -0.0018485, 0.0019460
4: -0.0105288, -0.0079472, -0.0104752, -0.0079025, -0.0017086, 0.0016231
5: 0.0097502, 0.0107280, 0.0097705, 0.0107449, -0.0006472, 0.0006148
6: 0.0060540, 0.0097855, 0.0059893, 0.0097080, -0.0023460, 0.0024697
7: 0.9822956, 0.9849067, 0.9822503, 0.9848524, -0.0016416, 0.0017282
8: -0.0055462, -0.0027466, -0.0055947, -0.0028048, -0.0017601, 0.0018529
9: -0.0031853, -0.0013360, -0.0031469, -0.0013040, -0.0012239, 0.0011626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010398, upper bound: 0.0010076
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010327, upper bound: 0.0010266
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028661, -0.0018343, -0.0006572, 0.0007158
1: -0.0114483, -0.0089602, -0.0115838, -0.0089654, -0.0016677, 0.0018164
2: 0.0279274, 0.0294711, 0.0278434, 0.0294679, -0.0010347, 0.0011269
3: 0.0045325, 0.0074150, 0.0045385, 0.0075719, -0.0021042, 0.0019320
4: -0.0105379, -0.0080071, -0.0106757, -0.0080123, -0.0016964, 0.0018476
5: 0.0097467, 0.0107053, 0.0096945, 0.0107033, -0.0006425, 0.0006998
6: 0.0061405, 0.0097987, 0.0061481, 0.0099978, -0.0026705, 0.0024520
7: 0.9823561, 0.9849159, 0.9823614, 0.9850553, -0.0018687, 0.0017158
8: -0.0054813, -0.0027368, -0.0054756, -0.0025874, -0.0020036, 0.0018396
9: -0.0031918, -0.0013789, -0.0032905, -0.0013827, -0.0012151, 0.0013235

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0009916
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010089
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028128, -0.0018323, -0.0028455, -0.0018186, -0.0006751, 0.0007000
1: -0.0114483, -0.0089602, -0.0115315, -0.0089255, -0.0017131, 0.0017765
2: 0.0279274, 0.0294711, 0.0278758, 0.0294926, -0.0010628, 0.0011021
3: 0.0045325, 0.0074150, 0.0044924, 0.0075113, -0.0020579, 0.0019846
4: -0.0105379, -0.0080071, -0.0106225, -0.0079718, -0.0017426, 0.0018070
5: 0.0097467, 0.0107053, 0.0097147, 0.0107187, -0.0006600, 0.0006844
6: 0.0061405, 0.0097987, 0.0060895, 0.0099209, -0.0026118, 0.0025187
7: 0.9823561, 0.9849159, 0.9823204, 0.9850016, -0.0018276, 0.0017625
8: -0.0054813, -0.0027368, -0.0055195, -0.0026450, -0.0019595, 0.0018897
9: -0.0031918, -0.0013789, -0.0032524, -0.0013536, -0.0012482, 0.0012943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0009929
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010091
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028661, -0.0018343, -0.0006360, 0.0007215
1: -0.0114394, -0.0089014, -0.0115838, -0.0089654, -0.0016140, 0.0018309
2: 0.0279330, 0.0295076, 0.0278434, 0.0294679, -0.0010013, 0.0011359
3: 0.0044644, 0.0074046, 0.0045385, 0.0075719, -0.0021210, 0.0018697
4: -0.0105288, -0.0079472, -0.0106757, -0.0080123, -0.0016417, 0.0018624
5: 0.0097502, 0.0107280, 0.0096945, 0.0107033, -0.0006218, 0.0007054
6: 0.0060540, 0.0097855, 0.0061481, 0.0099978, -0.0026919, 0.0023729
7: 0.9822956, 0.9849067, 0.9823614, 0.9850553, -0.0018836, 0.0016605
8: -0.0055462, -0.0027466, -0.0054756, -0.0025874, -0.0020196, 0.0017803
9: -0.0031853, -0.0013360, -0.0032905, -0.0013827, -0.0011760, 0.0013340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010125, upper bound: 0.0009887
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010138, upper bound: 0.0010129
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028092, -0.0018091, -0.0028455, -0.0018186, -0.0006497, 0.0007040
1: -0.0114394, -0.0089014, -0.0115315, -0.0089255, -0.0016486, 0.0017866
2: 0.0279330, 0.0295076, 0.0278758, 0.0294926, -0.0010228, 0.0011084
3: 0.0044644, 0.0074046, 0.0044924, 0.0075113, -0.0020696, 0.0019098
4: -0.0105288, -0.0079472, -0.0106225, -0.0079718, -0.0016769, 0.0018172
5: 0.0097502, 0.0107280, 0.0097147, 0.0107187, -0.0006352, 0.0006883
6: 0.0060540, 0.0097855, 0.0060895, 0.0099209, -0.0026266, 0.0024238
7: 0.9822956, 0.9849067, 0.9823204, 0.9850016, -0.0018380, 0.0016961
8: -0.0055462, -0.0027466, -0.0055195, -0.0026450, -0.0019706, 0.0018185
9: -0.0031853, -0.0013360, -0.0032524, -0.0013536, -0.0012012, 0.0013017

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010125, upper bound: 0.0009893
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010138, upper bound: 0.0010129
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028092, -0.0018091, -0.0007472, 0.0006330
1: -0.0115960, -0.0090144, -0.0114394, -0.0089014, -0.0018962, 0.0016062
2: 0.0278358, 0.0294374, 0.0279330, 0.0295076, -0.0011764, 0.0009965
3: 0.0045954, 0.0075860, 0.0044644, 0.0074046, -0.0018607, 0.0021967
4: -0.0106881, -0.0080622, -0.0105288, -0.0079472, -0.0019288, 0.0016338
5: 0.0096898, 0.0106844, 0.0097502, 0.0107280, -0.0007306, 0.0006188
6: 0.0062202, 0.0100157, 0.0060540, 0.0097855, -0.0023615, 0.0027879
7: 0.9824119, 0.9850678, 0.9822956, 0.9849067, -0.0016525, 0.0019508
8: -0.0054215, -0.0025739, -0.0055462, -0.0027466, -0.0017717, 0.0020916
9: -0.0032994, -0.0014184, -0.0031853, -0.0013360, -0.0013816, 0.0011703

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009841
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010074
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028661, -0.0018343, -0.0006790, 0.0006510
1: -0.0115960, -0.0090144, -0.0115838, -0.0089654, -0.0017230, 0.0016519
2: 0.0278358, 0.0294374, 0.0278434, 0.0294679, -0.0010690, 0.0010249
3: 0.0045954, 0.0075860, 0.0045385, 0.0075719, -0.0019137, 0.0019961
4: -0.0106881, -0.0080622, -0.0106757, -0.0080123, -0.0017526, 0.0016803
5: 0.0096898, 0.0106844, 0.0096945, 0.0107033, -0.0006638, 0.0006365
6: 0.0062202, 0.0100157, 0.0061481, 0.0099978, -0.0024287, 0.0025332
7: 0.9824119, 0.9850678, 0.9823614, 0.9850553, -0.0016995, 0.0017726
8: -0.0054215, -0.0025739, -0.0054756, -0.0025874, -0.0018221, 0.0019006
9: -0.0032994, -0.0014184, -0.0032905, -0.0013827, -0.0012554, 0.0012036

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009842
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010074
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028665, -0.0018343, -0.0028092, -0.0018091, -0.0007207, 0.0006360
1: -0.0115847, -0.0089654, -0.0114394, -0.0089014, -0.0018290, 0.0016140
2: 0.0278428, 0.0294679, 0.0279330, 0.0295076, -0.0011347, 0.0010013
3: 0.0045385, 0.0075729, 0.0044644, 0.0074046, -0.0018697, 0.0021188
4: -0.0106766, -0.0080123, -0.0105288, -0.0079472, -0.0018604, 0.0016417
5: 0.0096942, 0.0107033, 0.0097502, 0.0107280, -0.0007047, 0.0006218
6: 0.0061481, 0.0099991, 0.0060540, 0.0097855, -0.0023729, 0.0026890
7: 0.9823614, 0.9850562, 0.9822956, 0.9849067, -0.0016605, 0.0018816
8: -0.0054756, -0.0025863, -0.0055462, -0.0027466, -0.0017803, 0.0020174
9: -0.0032912, -0.0013827, -0.0031853, -0.0013360, -0.0013326, 0.0011760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009799
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010093
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028665, -0.0018343, -0.0028661, -0.0018343, -0.0006505, 0.0006507
1: -0.0115847, -0.0089654, -0.0115838, -0.0089654, -0.0016508, 0.0016513
2: 0.0278428, 0.0294679, 0.0278434, 0.0294679, -0.0010242, 0.0010245
3: 0.0045385, 0.0075729, 0.0045385, 0.0075719, -0.0019130, 0.0019124
4: -0.0106766, -0.0080123, -0.0106757, -0.0080123, -0.0016792, 0.0016797
5: 0.0096942, 0.0107033, 0.0096945, 0.0107033, -0.0006360, 0.0006362
6: 0.0061481, 0.0099991, 0.0061481, 0.0099978, -0.0024278, 0.0024271
7: 0.9823614, 0.9850562, 0.9823614, 0.9850553, -0.0016989, 0.0016983
8: -0.0054756, -0.0025863, -0.0054756, -0.0025874, -0.0018215, 0.0018209
9: -0.0032912, -0.0013827, -0.0032905, -0.0013827, -0.0012028, 0.0012032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009799
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010093
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0027885, -0.0017917, -0.0007714, 0.0006207
1: -0.0115960, -0.0090144, -0.0113867, -0.0088574, -0.0019575, 0.0015752
2: 0.0278358, 0.0294374, 0.0279657, 0.0295349, -0.0012145, 0.0009772
3: 0.0045954, 0.0075860, 0.0044134, 0.0073435, -0.0018248, 0.0022677
4: -0.0106881, -0.0080622, -0.0104752, -0.0079025, -0.0019911, 0.0016022
5: 0.0096898, 0.0106844, 0.0097705, 0.0107449, -0.0007542, 0.0006069
6: 0.0062202, 0.0100157, 0.0059893, 0.0097080, -0.0023159, 0.0028780
7: 0.9824119, 0.9850678, 0.9822503, 0.9848524, -0.0016205, 0.0020139
8: -0.0054215, -0.0025739, -0.0055947, -0.0028048, -0.0017375, 0.0021592
9: -0.0032994, -0.0014184, -0.0031469, -0.0013040, -0.0014263, 0.0011477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009827
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010033
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028709, -0.0018536, -0.0028455, -0.0018186, -0.0007042, 0.0006385
1: -0.0115960, -0.0090144, -0.0115315, -0.0089255, -0.0017870, 0.0016203
2: 0.0278358, 0.0294374, 0.0278758, 0.0294926, -0.0011087, 0.0010053
3: 0.0045954, 0.0075860, 0.0044924, 0.0075113, -0.0018771, 0.0020702
4: -0.0106881, -0.0080622, -0.0106225, -0.0079718, -0.0018177, 0.0016482
5: 0.0096898, 0.0106844, 0.0097147, 0.0107187, -0.0006885, 0.0006243
6: 0.0062202, 0.0100157, 0.0060895, 0.0099209, -0.0023823, 0.0026273
7: 0.9824119, 0.9850678, 0.9823204, 0.9850016, -0.0016670, 0.0018385
8: -0.0054215, -0.0025739, -0.0055195, -0.0026450, -0.0017873, 0.0019711
9: -0.0032994, -0.0014184, -0.0032524, -0.0013536, -0.0013020, 0.0011806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009827
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010033
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028665, -0.0018343, -0.0027885, -0.0017917, -0.0007407, 0.0006214
1: -0.0115847, -0.0089654, -0.0113867, -0.0088574, -0.0018795, 0.0015768
2: 0.0278428, 0.0294679, 0.0279657, 0.0295349, -0.0011661, 0.0009783
3: 0.0045385, 0.0075729, 0.0044134, 0.0073435, -0.0018267, 0.0021774
4: -0.0106766, -0.0080123, -0.0104752, -0.0079025, -0.0019118, 0.0016039
5: 0.0096942, 0.0107033, 0.0097705, 0.0107449, -0.0007241, 0.0006075
6: 0.0061481, 0.0099991, 0.0059893, 0.0097080, -0.0023183, 0.0027634
7: 0.9823614, 0.9850562, 0.9822503, 0.9848524, -0.0016222, 0.0019337
8: -0.0054756, -0.0025863, -0.0055947, -0.0028048, -0.0017393, 0.0020732
9: -0.0032912, -0.0013827, -0.0031469, -0.0013040, -0.0013695, 0.0011489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009793
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010066
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028665, -0.0018343, -0.0028455, -0.0018186, -0.0006708, 0.0006363
1: -0.0115847, -0.0089654, -0.0115315, -0.0089255, -0.0017023, 0.0016148
2: 0.0278428, 0.0294679, 0.0278758, 0.0294926, -0.0010561, 0.0010018
3: 0.0045385, 0.0075729, 0.0044924, 0.0075113, -0.0018707, 0.0019720
4: -0.0106766, -0.0080123, -0.0106225, -0.0079718, -0.0017315, 0.0016425
5: 0.0096942, 0.0107033, 0.0097147, 0.0107187, -0.0006558, 0.0006221
6: 0.0061481, 0.0099991, 0.0060895, 0.0099209, -0.0023741, 0.0025027
7: 0.9823614, 0.9850562, 0.9823204, 0.9850016, -0.0016613, 0.0017513
8: -0.0054756, -0.0025863, -0.0055195, -0.0026450, -0.0017812, 0.0018777
9: -0.0032912, -0.0013827, -0.0032524, -0.0013536, -0.0012403, 0.0011766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009793
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010066
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028128, -0.0018323, -0.0006201, 0.0006530
1: -0.0114011, -0.0089103, -0.0114483, -0.0089602, -0.0015737, 0.0016571
2: 0.0279567, 0.0295020, 0.0279274, 0.0294711, -0.0009763, 0.0010281
3: 0.0044748, 0.0073603, 0.0045325, 0.0074150, -0.0019197, 0.0018230
4: -0.0104899, -0.0079563, -0.0105379, -0.0080071, -0.0016007, 0.0016856
5: 0.0097649, 0.0107245, 0.0097467, 0.0107053, -0.0006063, 0.0006385
6: 0.0060672, 0.0097292, 0.0061405, 0.0097987, -0.0024364, 0.0023137
7: 0.9823048, 0.9848673, 0.9823561, 0.9849159, -0.0017049, 0.0016190
8: -0.0055363, -0.0027888, -0.0054813, -0.0027368, -0.0018279, 0.0017358
9: -0.0031574, -0.0013426, -0.0031918, -0.0013789, -0.0011466, 0.0012074

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010004
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010188
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0027942, -0.0018126, -0.0006515, 0.0006515
1: -0.0114011, -0.0089103, -0.0114011, -0.0089103, -0.0016532, 0.0016532
2: 0.0279567, 0.0295020, 0.0279567, 0.0295020, -0.0010257, 0.0010257
3: 0.0044748, 0.0073603, 0.0044748, 0.0073603, -0.0019152, 0.0019152
4: -0.0104899, -0.0079563, -0.0104899, -0.0079563, -0.0016816, 0.0016816
5: 0.0097649, 0.0107245, 0.0097649, 0.0107245, -0.0006370, 0.0006370
6: 0.0060672, 0.0097292, 0.0060672, 0.0097292, -0.0024306, 0.0024306
7: 0.9823048, 0.9848673, 0.9823048, 0.9848673, -0.0017009, 0.0017009
8: -0.0055363, -0.0027888, -0.0055363, -0.0027888, -0.0018236, 0.0018236
9: -0.0031574, -0.0013426, -0.0031574, -0.0013426, -0.0012046, 0.0012046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010226
time: 0.74 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010408
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028128, -0.0018323, -0.0006315, 0.0006939
1: -0.0113867, -0.0088574, -0.0114483, -0.0089602, -0.0016024, 0.0017609
2: 0.0279657, 0.0295349, 0.0279274, 0.0294711, -0.0009941, 0.0010925
3: 0.0044134, 0.0073435, 0.0045325, 0.0074150, -0.0020399, 0.0018563
4: -0.0104752, -0.0079025, -0.0105379, -0.0080071, -0.0016299, 0.0017911
5: 0.0097705, 0.0107449, 0.0097467, 0.0107053, -0.0006174, 0.0006784
6: 0.0059893, 0.0097080, 0.0061405, 0.0097987, -0.0025889, 0.0023559
7: 0.9822503, 0.9848524, 0.9823561, 0.9849159, -0.0018116, 0.0016485
8: -0.0055947, -0.0028048, -0.0054813, -0.0027368, -0.0019423, 0.0017675
9: -0.0031469, -0.0013040, -0.0031918, -0.0013789, -0.0011675, 0.0012830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010324, upper bound: 0.0010002
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010231, upper bound: 0.0010188
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0027942, -0.0018126, -0.0006596, 0.0006868
1: -0.0113867, -0.0088574, -0.0114011, -0.0089103, -0.0016739, 0.0017427
2: 0.0279657, 0.0295349, 0.0279567, 0.0295020, -0.0010385, 0.0010812
3: 0.0044134, 0.0073435, 0.0044748, 0.0073603, -0.0020189, 0.0019391
4: -0.0104752, -0.0079025, -0.0104899, -0.0079563, -0.0017026, 0.0017727
5: 0.0097705, 0.0107449, 0.0097649, 0.0107245, -0.0006449, 0.0006714
6: 0.0059893, 0.0097080, 0.0060672, 0.0097292, -0.0025622, 0.0024610
7: 0.9822503, 0.9848524, 0.9823048, 0.9848673, -0.0017929, 0.0017221
8: -0.0055947, -0.0028048, -0.0055363, -0.0027888, -0.0019223, 0.0018463
9: -0.0031469, -0.0013040, -0.0031574, -0.0013426, -0.0012196, 0.0012698

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010324, upper bound: 0.0010223
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010231, upper bound: 0.0010408
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028709, -0.0018536, -0.0006100, 0.0007299
1: -0.0114011, -0.0089103, -0.0115958, -0.0090144, -0.0015480, 0.0018522
2: 0.0279567, 0.0295020, 0.0278359, 0.0294374, -0.0009604, 0.0011491
3: 0.0044748, 0.0073603, 0.0045954, 0.0075858, -0.0021456, 0.0017933
4: -0.0104899, -0.0079563, -0.0106879, -0.0080622, -0.0015746, 0.0018840
5: 0.0097649, 0.0107245, 0.0096899, 0.0106844, -0.0005964, 0.0007136
6: 0.0060672, 0.0097292, 0.0062202, 0.0100155, -0.0027231, 0.0022760
7: 0.9823048, 0.9848673, 0.9824119, 0.9850676, -0.0019055, 0.0015926
8: -0.0055363, -0.0027888, -0.0054215, -0.0025741, -0.0020430, 0.0017075
9: -0.0031574, -0.0013426, -0.0032993, -0.0014184, -0.0011279, 0.0013495

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0009848
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010081
time: 0.77 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028514, -0.0018361, -0.0006422, 0.0007304
1: -0.0114011, -0.0089103, -0.0115463, -0.0089699, -0.0016296, 0.0018536
2: 0.0279567, 0.0295020, 0.0278666, 0.0294651, -0.0010110, 0.0011500
3: 0.0044748, 0.0073603, 0.0045438, 0.0075285, -0.0021473, 0.0018878
4: -0.0104899, -0.0079563, -0.0106376, -0.0080169, -0.0016576, 0.0018854
5: 0.0097649, 0.0107245, 0.0097090, 0.0107016, -0.0006278, 0.0007141
6: 0.0060672, 0.0097292, 0.0061547, 0.0099427, -0.0027252, 0.0023958
7: 0.9823048, 0.9848673, 0.9823660, 0.9850166, -0.0019070, 0.0016765
8: -0.0055363, -0.0027888, -0.0054706, -0.0026287, -0.0020446, 0.0017975
9: -0.0031574, -0.0013426, -0.0032632, -0.0013860, -0.0011873, 0.0013506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0009953
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010167
time: 0.75 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028709, -0.0018536, -0.0006211, 0.0007708
1: -0.0113867, -0.0088574, -0.0115958, -0.0090144, -0.0015762, 0.0019559
2: 0.0279657, 0.0295349, 0.0278359, 0.0294374, -0.0009779, 0.0012134
3: 0.0044134, 0.0073435, 0.0045954, 0.0075858, -0.0022658, 0.0018259
4: -0.0104752, -0.0079025, -0.0106879, -0.0080622, -0.0016032, 0.0019895
5: 0.0097705, 0.0107449, 0.0096899, 0.0106844, -0.0006073, 0.0007536
6: 0.0059893, 0.0097080, 0.0062202, 0.0100155, -0.0028756, 0.0023173
7: 0.9822503, 0.9848524, 0.9824119, 0.9850676, -0.0020122, 0.0016215
8: -0.0055947, -0.0028048, -0.0054215, -0.0025741, -0.0021574, 0.0017386
9: -0.0031469, -0.0013040, -0.0032993, -0.0014184, -0.0011484, 0.0014251

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010069, upper bound: 0.0009847
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0010081
time: 0.68 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028514, -0.0018361, -0.0006501, 0.0007657
1: -0.0113867, -0.0088574, -0.0115463, -0.0089699, -0.0016497, 0.0019431
2: 0.0279657, 0.0295349, 0.0278666, 0.0294651, -0.0010235, 0.0012055
3: 0.0044134, 0.0073435, 0.0045438, 0.0075285, -0.0022510, 0.0019111
4: -0.0104752, -0.0079025, -0.0106376, -0.0080169, -0.0016780, 0.0019764
5: 0.0097705, 0.0107449, 0.0097090, 0.0107016, -0.0006356, 0.0007486
6: 0.0059893, 0.0097080, 0.0061547, 0.0099427, -0.0028568, 0.0024254
7: 0.9822503, 0.9848524, 0.9823660, 0.9850166, -0.0019990, 0.0016972
8: -0.0055947, -0.0028048, -0.0054706, -0.0026287, -0.0021433, 0.0018197
9: -0.0031469, -0.0013040, -0.0032632, -0.0013860, -0.0012020, 0.0014158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010069, upper bound: 0.0009953
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0010167
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028128, -0.0018323, -0.0006942, 0.0006384
1: -0.0115469, -0.0089699, -0.0114483, -0.0089602, -0.0017618, 0.0016199
2: 0.0278663, 0.0294651, 0.0279274, 0.0294711, -0.0010930, 0.0010050
3: 0.0045438, 0.0075291, 0.0045325, 0.0074150, -0.0018766, 0.0020409
4: -0.0106381, -0.0080169, -0.0105379, -0.0080071, -0.0017920, 0.0016477
5: 0.0097087, 0.0107016, 0.0097467, 0.0107053, -0.0006788, 0.0006241
6: 0.0061547, 0.0099435, 0.0061405, 0.0097987, -0.0023817, 0.0025902
7: 0.9823660, 0.9850173, 0.9823561, 0.9849159, -0.0016666, 0.0018125
8: -0.0054706, -0.0026281, -0.0054813, -0.0027368, -0.0017868, 0.0019433
9: -0.0032636, -0.0013860, -0.0031918, -0.0013789, -0.0012836, 0.0011803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009740
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010027
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028709, -0.0018536, -0.0006288, 0.0006619
1: -0.0115469, -0.0089699, -0.0115958, -0.0090144, -0.0015956, 0.0016796
2: 0.0278663, 0.0294651, 0.0278359, 0.0294374, -0.0009899, 0.0010420
3: 0.0045438, 0.0075291, 0.0045954, 0.0075858, -0.0019458, 0.0018484
4: -0.0106381, -0.0080169, -0.0106879, -0.0080622, -0.0016230, 0.0017085
5: 0.0097087, 0.0107016, 0.0096899, 0.0106844, -0.0006147, 0.0006471
6: 0.0061547, 0.0099435, 0.0062202, 0.0100155, -0.0024694, 0.0023459
7: 0.9823660, 0.9850173, 0.9824119, 0.9850676, -0.0017280, 0.0016415
8: -0.0054706, -0.0026281, -0.0054215, -0.0025741, -0.0018527, 0.0017600
9: -0.0032636, -0.0013860, -0.0032993, -0.0014184, -0.0011626, 0.0012238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009740
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010027
time: 0.69 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028128, -0.0018323, -0.0007003, 0.0006751
1: -0.0115328, -0.0089255, -0.0114483, -0.0089602, -0.0017770, 0.0017131
2: 0.0278750, 0.0294926, 0.0279274, 0.0294711, -0.0011025, 0.0010628
3: 0.0044924, 0.0075128, 0.0045325, 0.0074150, -0.0019846, 0.0020586
4: -0.0106238, -0.0079718, -0.0105379, -0.0080071, -0.0018075, 0.0017426
5: 0.0097142, 0.0107187, 0.0097467, 0.0107053, -0.0006846, 0.0006600
6: 0.0060895, 0.0099228, 0.0061405, 0.0097987, -0.0025187, 0.0026126
7: 0.9823204, 0.9850028, 0.9823561, 0.9849159, -0.0017625, 0.0018282
8: -0.0055195, -0.0026436, -0.0054813, -0.0027368, -0.0018896, 0.0019601
9: -0.0032533, -0.0013536, -0.0031918, -0.0013789, -0.0012948, 0.0012482

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009734
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010091, upper bound: 0.0010027
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028709, -0.0018536, -0.0006393, 0.0007031
1: -0.0115328, -0.0089255, -0.0115958, -0.0090144, -0.0016223, 0.0017843
2: 0.0278750, 0.0294926, 0.0278359, 0.0294374, -0.0010065, 0.0011070
3: 0.0044924, 0.0075128, 0.0045954, 0.0075858, -0.0020670, 0.0018793
4: -0.0106238, -0.0079718, -0.0106879, -0.0080622, -0.0016501, 0.0018149
5: 0.0097142, 0.0107187, 0.0096899, 0.0106844, -0.0006250, 0.0006874
6: 0.0060895, 0.0099228, 0.0062202, 0.0100155, -0.0026233, 0.0023851
7: 0.9823204, 0.9850028, 0.9824119, 0.9850676, -0.0018357, 0.0016690
8: -0.0055195, -0.0026436, -0.0054215, -0.0025741, -0.0019681, 0.0017894
9: -0.0032533, -0.0013536, -0.0032993, -0.0014184, -0.0011820, 0.0013000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009734
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010091, upper bound: 0.0010027
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0027942, -0.0018126, -0.0007294, 0.0006422
1: -0.0115469, -0.0089699, -0.0114011, -0.0089103, -0.0018509, 0.0016296
2: 0.0278663, 0.0294651, 0.0279567, 0.0295020, -0.0011483, 0.0010110
3: 0.0045438, 0.0075291, 0.0044748, 0.0073603, -0.0018878, 0.0021441
4: -0.0106381, -0.0080169, -0.0104899, -0.0079563, -0.0018826, 0.0016576
5: 0.0097087, 0.0107016, 0.0097649, 0.0107245, -0.0007131, 0.0006278
6: 0.0061547, 0.0099435, 0.0060672, 0.0097292, -0.0023958, 0.0027212
7: 0.9823660, 0.9850173, 0.9823048, 0.9848673, -0.0016765, 0.0019042
8: -0.0054706, -0.0026281, -0.0055363, -0.0027888, -0.0017975, 0.0020416
9: -0.0032636, -0.0013860, -0.0031574, -0.0013426, -0.0013486, 0.0011873

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009834
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010097
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028514, -0.0018361, -0.0006572, 0.0006577
1: -0.0115469, -0.0089699, -0.0115463, -0.0089699, -0.0016677, 0.0016690
2: 0.0278663, 0.0294651, 0.0278666, 0.0294651, -0.0010347, 0.0010355
3: 0.0045438, 0.0075291, 0.0045438, 0.0075285, -0.0019335, 0.0019320
4: -0.0106381, -0.0080169, -0.0106376, -0.0080169, -0.0016963, 0.0016977
5: 0.0097087, 0.0107016, 0.0097090, 0.0107016, -0.0006425, 0.0006430
6: 0.0061547, 0.0099435, 0.0061547, 0.0099427, -0.0024539, 0.0024519
7: 0.9823660, 0.9850173, 0.9823660, 0.9850166, -0.0017171, 0.0017157
8: -0.0054706, -0.0026281, -0.0054706, -0.0026287, -0.0018410, 0.0018395
9: -0.0032636, -0.0013860, -0.0032632, -0.0013860, -0.0012151, 0.0012161

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009834
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010097
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0027942, -0.0018126, -0.0007342, 0.0006744
1: -0.0115328, -0.0089255, -0.0114011, -0.0089103, -0.0018632, 0.0017114
2: 0.0278750, 0.0294926, 0.0279567, 0.0295020, -0.0011559, 0.0010617
3: 0.0044924, 0.0075128, 0.0044748, 0.0073603, -0.0019825, 0.0021584
4: -0.0106238, -0.0079718, -0.0104899, -0.0079563, -0.0018952, 0.0017407
5: 0.0097142, 0.0107187, 0.0097649, 0.0107245, -0.0007178, 0.0006593
6: 0.0060895, 0.0099228, 0.0060672, 0.0097292, -0.0025161, 0.0027393
7: 0.9823204, 0.9850028, 0.9823048, 0.9848673, -0.0017606, 0.0019169
8: -0.0055195, -0.0026436, -0.0055363, -0.0027888, -0.0018877, 0.0020552
9: -0.0032533, -0.0013536, -0.0031574, -0.0013426, -0.0013576, 0.0012469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010169, upper bound: 0.0009829
time: 0.70 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010176, upper bound: 0.0010097
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028514, -0.0018361, -0.0006663, 0.0006928
1: -0.0115328, -0.0089255, -0.0115463, -0.0089699, -0.0016909, 0.0017580
2: 0.0278750, 0.0294926, 0.0278666, 0.0294651, -0.0010490, 0.0010907
3: 0.0044924, 0.0075128, 0.0045438, 0.0075285, -0.0020366, 0.0019588
4: -0.0106238, -0.0079718, -0.0106376, -0.0080169, -0.0017199, 0.0017882
5: 0.0097142, 0.0107187, 0.0097090, 0.0107016, -0.0006515, 0.0006773
6: 0.0060895, 0.0099228, 0.0061547, 0.0099427, -0.0025846, 0.0024860
7: 0.9823204, 0.9850028, 0.9823660, 0.9850166, -0.0018086, 0.0017396
8: -0.0055195, -0.0026436, -0.0054706, -0.0026287, -0.0019391, 0.0018651
9: -0.0032533, -0.0013536, -0.0032632, -0.0013860, -0.0012320, 0.0012809

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010169, upper bound: 0.0009830
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010176, upper bound: 0.0010097
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028092, -0.0018091, -0.0006561, 0.0006616
1: -0.0114011, -0.0089103, -0.0114394, -0.0089014, -0.0016650, 0.0016788
2: 0.0279567, 0.0295020, 0.0279330, 0.0295076, -0.0010330, 0.0010416
3: 0.0044748, 0.0073603, 0.0044644, 0.0074046, -0.0019448, 0.0019288
4: -0.0104899, -0.0079563, -0.0105288, -0.0079472, -0.0016936, 0.0017077
5: 0.0097649, 0.0107245, 0.0097502, 0.0107280, -0.0006415, 0.0006468
6: 0.0060672, 0.0097292, 0.0060540, 0.0097855, -0.0024683, 0.0024479
7: 0.9823048, 0.9848673, 0.9822956, 0.9849067, -0.0017272, 0.0017129
8: -0.0055363, -0.0027888, -0.0055462, -0.0027466, -0.0018518, 0.0018365
9: -0.0031574, -0.0013426, -0.0031853, -0.0013360, -0.0012131, 0.0012232

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010091
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010215
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0027885, -0.0017917, -0.0006868, 0.0006596
1: -0.0114011, -0.0089103, -0.0113867, -0.0088574, -0.0017427, 0.0016739
2: 0.0279567, 0.0295020, 0.0279657, 0.0295349, -0.0010812, 0.0010385
3: 0.0044748, 0.0073603, 0.0044134, 0.0073435, -0.0019391, 0.0020189
4: -0.0104899, -0.0079563, -0.0104752, -0.0079025, -0.0017727, 0.0017026
5: 0.0097649, 0.0107245, 0.0097705, 0.0107449, -0.0006714, 0.0006449
6: 0.0060672, 0.0097292, 0.0059893, 0.0097080, -0.0024610, 0.0025622
7: 0.9823048, 0.9848673, 0.9822503, 0.9848524, -0.0017221, 0.0017929
8: -0.0055363, -0.0027888, -0.0055947, -0.0028048, -0.0018463, 0.0019223
9: -0.0031574, -0.0013426, -0.0031469, -0.0013040, -0.0012698, 0.0012196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010322
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010439
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028092, -0.0018091, -0.0006287, 0.0006620
1: -0.0113867, -0.0088574, -0.0114394, -0.0089014, -0.0015954, 0.0016798
2: 0.0279657, 0.0295349, 0.0279330, 0.0295076, -0.0009898, 0.0010422
3: 0.0044134, 0.0073435, 0.0044644, 0.0074046, -0.0019460, 0.0018482
4: -0.0104752, -0.0079025, -0.0105288, -0.0079472, -0.0016228, 0.0017086
5: 0.0097705, 0.0107449, 0.0097502, 0.0107280, -0.0006147, 0.0006472
6: 0.0059893, 0.0097080, 0.0060540, 0.0097855, -0.0024697, 0.0023456
7: 0.9822503, 0.9848524, 0.9822956, 0.9849067, -0.0017282, 0.0016414
8: -0.0055947, -0.0028048, -0.0055462, -0.0027466, -0.0018529, 0.0017598
9: -0.0031469, -0.0013040, -0.0031853, -0.0013360, -0.0011625, 0.0012239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010384, upper bound: 0.0010067
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010287, upper bound: 0.0010250
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0027885, -0.0017917, -0.0006598, 0.0006598
1: -0.0113867, -0.0088574, -0.0113867, -0.0088574, -0.0016744, 0.0016744
2: 0.0279657, 0.0295349, 0.0279657, 0.0295349, -0.0010388, 0.0010388
3: 0.0044134, 0.0073435, 0.0044134, 0.0073435, -0.0019397, 0.0019397
4: -0.0104752, -0.0079025, -0.0104752, -0.0079025, -0.0017031, 0.0017031
5: 0.0097705, 0.0107449, 0.0097705, 0.0107449, -0.0006451, 0.0006451
6: 0.0059893, 0.0097080, 0.0059893, 0.0097080, -0.0024617, 0.0024617
7: 0.9822503, 0.9848524, 0.9822503, 0.9848524, -0.0017226, 0.0017226
8: -0.0055947, -0.0028048, -0.0055947, -0.0028048, -0.0018469, 0.0018469
9: -0.0031469, -0.0013040, -0.0031469, -0.0013040, -0.0012200, 0.0012200

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010384, upper bound: 0.0010271
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010287, upper bound: 0.0010449
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028665, -0.0018343, -0.0006428, 0.0007354
1: -0.0114011, -0.0089103, -0.0115847, -0.0089654, -0.0016312, 0.0018662
2: 0.0279567, 0.0295020, 0.0278428, 0.0294679, -0.0010120, 0.0011578
3: 0.0044748, 0.0073603, 0.0045385, 0.0075729, -0.0021619, 0.0018897
4: -0.0104899, -0.0079563, -0.0106766, -0.0080123, -0.0016592, 0.0018982
5: 0.0097649, 0.0107245, 0.0096942, 0.0107033, -0.0006285, 0.0007190
6: 0.0060672, 0.0097292, 0.0061481, 0.0099991, -0.0027437, 0.0023983
7: 0.9823048, 0.9848673, 0.9823614, 0.9850562, -0.0019199, 0.0016782
8: -0.0055363, -0.0027888, -0.0054756, -0.0025863, -0.0020585, 0.0017993
9: -0.0031574, -0.0013426, -0.0032912, -0.0013827, -0.0011885, 0.0013597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0009916
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010092
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0027942, -0.0018126, -0.0028460, -0.0018186, -0.0006744, 0.0007342
1: -0.0114011, -0.0089103, -0.0115328, -0.0089255, -0.0017114, 0.0018632
2: 0.0279567, 0.0295020, 0.0278750, 0.0294926, -0.0010617, 0.0011559
3: 0.0044748, 0.0073603, 0.0044924, 0.0075128, -0.0021584, 0.0019825
4: -0.0104899, -0.0079563, -0.0106238, -0.0079718, -0.0017407, 0.0018952
5: 0.0097649, 0.0107245, 0.0097142, 0.0107187, -0.0006593, 0.0007178
6: 0.0060672, 0.0097292, 0.0060895, 0.0099228, -0.0027393, 0.0025161
7: 0.9823048, 0.9848673, 0.9823204, 0.9850028, -0.0019169, 0.0017606
8: -0.0055363, -0.0027888, -0.0055195, -0.0026436, -0.0020552, 0.0018877
9: -0.0031574, -0.0013426, -0.0032533, -0.0013536, -0.0012469, 0.0013576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0010022
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010177
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028665, -0.0018343, -0.0006211, 0.0007407
1: -0.0113867, -0.0088574, -0.0115847, -0.0089654, -0.0015760, 0.0018795
2: 0.0279657, 0.0295349, 0.0278428, 0.0294679, -0.0009778, 0.0011661
3: 0.0044134, 0.0073435, 0.0045385, 0.0075729, -0.0021774, 0.0018258
4: -0.0104752, -0.0079025, -0.0106766, -0.0080123, -0.0016031, 0.0019118
5: 0.0097705, 0.0107449, 0.0096942, 0.0107033, -0.0006072, 0.0007241
6: 0.0059893, 0.0097080, 0.0061481, 0.0099991, -0.0027634, 0.0023171
7: 0.9822503, 0.9848524, 0.9823614, 0.9850562, -0.0019337, 0.0016214
8: -0.0055947, -0.0028048, -0.0054756, -0.0025863, -0.0020732, 0.0017384
9: -0.0031469, -0.0013040, -0.0032912, -0.0013827, -0.0011483, 0.0013695

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010130, upper bound: 0.0009891
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010124, upper bound: 0.0010134
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0027885, -0.0017917, -0.0028460, -0.0018186, -0.0006529, 0.0007399
1: -0.0113867, -0.0088574, -0.0115328, -0.0089255, -0.0016569, 0.0018775
2: 0.0279657, 0.0295349, 0.0278750, 0.0294926, -0.0010280, 0.0011648
3: 0.0044134, 0.0073435, 0.0044924, 0.0075128, -0.0021750, 0.0019195
4: -0.0104752, -0.0079025, -0.0106238, -0.0079718, -0.0016854, 0.0019097
5: 0.0097705, 0.0107449, 0.0097142, 0.0107187, -0.0006384, 0.0007234
6: 0.0059893, 0.0097080, 0.0060895, 0.0099228, -0.0027604, 0.0024360
7: 0.9822503, 0.9848524, 0.9823204, 0.9850028, -0.0019316, 0.0017046
8: -0.0055947, -0.0028048, -0.0055195, -0.0026436, -0.0020709, 0.0018276
9: -0.0031469, -0.0013040, -0.0032533, -0.0013536, -0.0012073, 0.0013680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010130, upper bound: 0.0009998
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010124, upper bound: 0.0010210
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028092, -0.0018091, -0.0007304, 0.0006469
1: -0.0115469, -0.0089699, -0.0114394, -0.0089014, -0.0018536, 0.0016416
2: 0.0278663, 0.0294651, 0.0279330, 0.0295076, -0.0011500, 0.0010185
3: 0.0045438, 0.0075291, 0.0044644, 0.0074046, -0.0019017, 0.0021473
4: -0.0106381, -0.0080169, -0.0105288, -0.0079472, -0.0018854, 0.0016698
5: 0.0097087, 0.0107016, 0.0097502, 0.0107280, -0.0007142, 0.0006325
6: 0.0061547, 0.0099435, 0.0060540, 0.0097855, -0.0024135, 0.0027252
7: 0.9823660, 0.9850173, 0.9822956, 0.9849067, -0.0016889, 0.0019070
8: -0.0054706, -0.0026281, -0.0055462, -0.0027466, -0.0018108, 0.0020446
9: -0.0032636, -0.0013860, -0.0031853, -0.0013360, -0.0013506, 0.0011961

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009842
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010074
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028661, -0.0018343, -0.0006650, 0.0006702
1: -0.0115469, -0.0089699, -0.0115838, -0.0089654, -0.0016874, 0.0017007
2: 0.0278663, 0.0294651, 0.0278434, 0.0294679, -0.0010469, 0.0010551
3: 0.0045438, 0.0075291, 0.0045385, 0.0075719, -0.0019702, 0.0019548
4: -0.0106381, -0.0080169, -0.0106757, -0.0080123, -0.0017164, 0.0017299
5: 0.0097087, 0.0107016, 0.0096945, 0.0107033, -0.0006501, 0.0006552
6: 0.0061547, 0.0099435, 0.0061481, 0.0099978, -0.0025004, 0.0024809
7: 0.9823660, 0.9850173, 0.9823614, 0.9850553, -0.0017497, 0.0017360
8: -0.0054706, -0.0026281, -0.0054756, -0.0025874, -0.0018759, 0.0018613
9: -0.0032636, -0.0013860, -0.0032905, -0.0013827, -0.0012295, 0.0012392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009841
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010074
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028092, -0.0018091, -0.0007038, 0.0006497
1: -0.0115328, -0.0089255, -0.0114394, -0.0089014, -0.0017859, 0.0016486
2: 0.0278750, 0.0294926, 0.0279330, 0.0295076, -0.0011080, 0.0010228
3: 0.0044924, 0.0075128, 0.0044644, 0.0074046, -0.0019098, 0.0020689
4: -0.0106238, -0.0079718, -0.0105288, -0.0079472, -0.0018166, 0.0016769
5: 0.0097142, 0.0107187, 0.0097502, 0.0107280, -0.0006881, 0.0006352
6: 0.0060895, 0.0099228, 0.0060540, 0.0097855, -0.0024238, 0.0026257
7: 0.9823204, 0.9850028, 0.9822956, 0.9849067, -0.0016961, 0.0018374
8: -0.0055195, -0.0026436, -0.0055462, -0.0027466, -0.0018185, 0.0019699
9: -0.0032533, -0.0013536, -0.0031853, -0.0013360, -0.0013013, 0.0012012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010100, upper bound: 0.0009799
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010142, upper bound: 0.0010093
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028661, -0.0018343, -0.0006365, 0.0006698
1: -0.0115328, -0.0089255, -0.0115838, -0.0089654, -0.0016153, 0.0016997
2: 0.0278750, 0.0294926, 0.0278434, 0.0294679, -0.0010022, 0.0010545
3: 0.0044924, 0.0075128, 0.0045385, 0.0075719, -0.0019690, 0.0018713
4: -0.0106238, -0.0079718, -0.0106757, -0.0080123, -0.0016431, 0.0017289
5: 0.0097142, 0.0107187, 0.0096945, 0.0107033, -0.0006223, 0.0006549
6: 0.0060895, 0.0099228, 0.0061481, 0.0099978, -0.0024989, 0.0023749
7: 0.9823204, 0.9850028, 0.9823614, 0.9850553, -0.0017486, 0.0016618
8: -0.0055195, -0.0026436, -0.0054756, -0.0025874, -0.0018748, 0.0017817
9: -0.0032533, -0.0013536, -0.0032905, -0.0013827, -0.0011769, 0.0012384

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010100, upper bound: 0.0009799
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010142, upper bound: 0.0010093
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0027885, -0.0017917, -0.0007650, 0.0006501
1: -0.0115469, -0.0089699, -0.0113867, -0.0088574, -0.0019413, 0.0016497
2: 0.0278663, 0.0294651, 0.0279657, 0.0295349, -0.0012044, 0.0010235
3: 0.0045438, 0.0075291, 0.0044134, 0.0073435, -0.0019111, 0.0022490
4: -0.0106381, -0.0080169, -0.0104752, -0.0079025, -0.0019747, 0.0016780
5: 0.0097087, 0.0107016, 0.0097705, 0.0107449, -0.0007480, 0.0006356
6: 0.0061547, 0.0099435, 0.0059893, 0.0097080, -0.0024254, 0.0028542
7: 0.9823660, 0.9850173, 0.9822503, 0.9848524, -0.0016972, 0.0019972
8: -0.0054706, -0.0026281, -0.0055947, -0.0028048, -0.0018197, 0.0021414
9: -0.0032636, -0.0013860, -0.0031469, -0.0013040, -0.0014145, 0.0012020

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009933
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010121
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0028516, -0.0018361, -0.0028460, -0.0018186, -0.0006927, 0.0006663
1: -0.0115469, -0.0089699, -0.0115328, -0.0089255, -0.0017578, 0.0016909
2: 0.0278663, 0.0294651, 0.0278750, 0.0294926, -0.0010905, 0.0010490
3: 0.0045438, 0.0075291, 0.0044924, 0.0075128, -0.0019588, 0.0020363
4: -0.0106381, -0.0080169, -0.0106238, -0.0079718, -0.0017880, 0.0017199
5: 0.0097087, 0.0107016, 0.0097142, 0.0107187, -0.0006772, 0.0006515
6: 0.0061547, 0.0099435, 0.0060895, 0.0099228, -0.0024860, 0.0025843
7: 0.9823660, 0.9850173, 0.9823204, 0.9850028, -0.0017396, 0.0018084
8: -0.0054706, -0.0026281, -0.0055195, -0.0026436, -0.0018651, 0.0019389
9: -0.0032636, -0.0013860, -0.0032533, -0.0013536, -0.0012807, 0.0012320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009933
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010121
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0027885, -0.0017917, -0.0007399, 0.0006529
1: -0.0115328, -0.0089255, -0.0113867, -0.0088574, -0.0018775, 0.0016569
2: 0.0278750, 0.0294926, 0.0279657, 0.0295349, -0.0011648, 0.0010280
3: 0.0044924, 0.0075128, 0.0044134, 0.0073435, -0.0019195, 0.0021750
4: -0.0106238, -0.0079718, -0.0104752, -0.0079025, -0.0019097, 0.0016854
5: 0.0097142, 0.0107187, 0.0097705, 0.0107449, -0.0007234, 0.0006384
6: 0.0060895, 0.0099228, 0.0059893, 0.0097080, -0.0024360, 0.0027604
7: 0.9823204, 0.9850028, 0.9822503, 0.9848524, -0.0017046, 0.0019316
8: -0.0055195, -0.0026436, -0.0055947, -0.0028048, -0.0018276, 0.0020709
9: -0.0032533, -0.0013536, -0.0031469, -0.0013040, -0.0013680, 0.0012073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010206, upper bound: 0.0009884
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010218, upper bound: 0.0010140
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0028460, -0.0018186, -0.0028460, -0.0018186, -0.0006646, 0.0006646
1: -0.0115328, -0.0089255, -0.0115328, -0.0089255, -0.0016866, 0.0016866
2: 0.0278750, 0.0294926, 0.0278750, 0.0294926, -0.0010463, 0.0010463
3: 0.0044924, 0.0075128, 0.0044924, 0.0075128, -0.0019538, 0.0019538
4: -0.0106238, -0.0079718, -0.0106238, -0.0079718, -0.0017155, 0.0017155
5: 0.0097142, 0.0107187, 0.0097142, 0.0107187, -0.0006498, 0.0006498
6: 0.0060895, 0.0099228, 0.0060895, 0.0099228, -0.0024796, 0.0024796
7: 0.9823204, 0.9850028, 0.9823204, 0.9850028, -0.0017351, 0.0017351
8: -0.0055195, -0.0026436, -0.0055195, -0.0026436, -0.0018603, 0.0018603
9: -0.0032533, -0.0013536, -0.0032533, -0.0013536, -0.0012288, 0.0012288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 81
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 123
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010206, upper bound: 0.0009884
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010218, upper bound: 0.0010140
time: 0.73 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.35 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010003
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010188
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010013
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010205
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010340, upper bound: 0.0010001
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010265, upper bound: 0.0010188
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010340, upper bound: 0.0010010
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010265, upper bound: 0.0010205
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0009846
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010079
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0009853
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010079
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009843
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010078, upper bound: 0.0010079
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009848
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010078, upper bound: 0.0010079
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009740
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010027
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009740
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010027
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009734
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010144, upper bound: 0.0010027
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009734
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010144, upper bound: 0.0010027
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009728
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010007
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009728
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010007
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009721
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010143, upper bound: 0.0010007
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010101, upper bound: 0.0009721
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010143, upper bound: 0.0010007
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010091
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010215
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010320, upper bound: 0.0010110
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010213, upper bound: 0.0010231
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010398, upper bound: 0.0010066
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010327, upper bound: 0.0010250
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010398, upper bound: 0.0010076
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010327, upper bound: 0.0010266
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0009916
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010089
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010038, upper bound: 0.0009929
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010031, upper bound: 0.0010091
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010125, upper bound: 0.0009887
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010138, upper bound: 0.0010129
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010125, upper bound: 0.0009893
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010138, upper bound: 0.0010129
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009841
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010074
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009842
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010120, upper bound: 0.0010074
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009799
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010093
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009799
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010093
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009827
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010033
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010092, upper bound: 0.0009827
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010119, upper bound: 0.0010033
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009793
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010066
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010136, upper bound: 0.0009793
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010201, upper bound: 0.0010066
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010004
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010188
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010226
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010408
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010324, upper bound: 0.0010002
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010231, upper bound: 0.0010188
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010324, upper bound: 0.0010223
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010231, upper bound: 0.0010408
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0009848
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010081
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0009953
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010167
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010069, upper bound: 0.0009847
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0010081
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010069, upper bound: 0.0009953
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0010167
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009740
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010027
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009740
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010027
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009734
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010091, upper bound: 0.0010027
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0009734
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010091, upper bound: 0.0010027
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009834
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010097
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009834
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010097
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010169, upper bound: 0.0009829
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010176, upper bound: 0.0010097
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010169, upper bound: 0.0009830
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010176, upper bound: 0.0010097
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010091
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010215
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010334, upper bound: 0.0010322
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010205, upper bound: 0.0010439
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010384, upper bound: 0.0010067
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010287, upper bound: 0.0010250
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010384, upper bound: 0.0010271
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010287, upper bound: 0.0010449
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0009916
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010092
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010061, upper bound: 0.0010022
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010039, upper bound: 0.0010177
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010130, upper bound: 0.0009891
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010124, upper bound: 0.0010134
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010130, upper bound: 0.0009998
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010124, upper bound: 0.0010210
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009842
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010074
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010064, upper bound: 0.0009841
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010079, upper bound: 0.0010074
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010100, upper bound: 0.0009799
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010142, upper bound: 0.0010093
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010100, upper bound: 0.0009799
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010142, upper bound: 0.0010093
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009933
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010121
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010179, upper bound: 0.0009933
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010167, upper bound: 0.0010121
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010206, upper bound: 0.0009884
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010218, upper bound: 0.0010140
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010206, upper bound: 0.0009884
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.35
Output dim: 7, lower bound: -0.0010218, upper bound: 0.0010140

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0028126, -0.0018365, -0.0006258, 0.0006143
1: -0.0114628, -0.0089936, -0.0114480, -0.0089709, -0.0015880, 0.0015589
2: 0.0279184, 0.0294504, 0.0279277, 0.0294644, -0.0009852, 0.0009672
3: 0.0045712, 0.0074317, 0.0045449, 0.0074145, -0.0018059, 0.0018397
4: -0.0105527, -0.0080410, -0.0105375, -0.0080179, -0.0016153, 0.0015857
5: 0.0097411, 0.0106925, 0.0097469, 0.0107012, -0.0006118, 0.0006006
6: 0.0061896, 0.0098200, 0.0061562, 0.0097981, -0.0022920, 0.0023348
7: 0.9823905, 0.9849308, 0.9823671, 0.9849154, -0.0016038, 0.0016338
8: -0.0054445, -0.0027208, -0.0054695, -0.0027372, -0.0017195, 0.0017516
9: -0.0032024, -0.0014032, -0.0031915, -0.0013867, -0.0011571, 0.0011358

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009886, upper bound: 0.0009888
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010164, upper bound: 0.0009888
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0028127, -0.0018368, -0.0006261, 0.0006138
1: -0.0114474, -0.0090018, -0.0114481, -0.0089718, -0.0015887, 0.0015577
2: 0.0279280, 0.0294453, 0.0279276, 0.0294639, -0.0009856, 0.0009664
3: 0.0045807, 0.0074139, 0.0045460, 0.0074147, -0.0018045, 0.0018405
4: -0.0105370, -0.0080494, -0.0105377, -0.0080189, -0.0016160, 0.0015844
5: 0.0097471, 0.0106893, 0.0097468, 0.0107008, -0.0006121, 0.0006001
6: 0.0062017, 0.0097973, 0.0061576, 0.0097983, -0.0022902, 0.0023358
7: 0.9823989, 0.9849149, 0.9823681, 0.9849156, -0.0016025, 0.0016345
8: -0.0054354, -0.0027378, -0.0054685, -0.0027371, -0.0017182, 0.0017524
9: -0.0031911, -0.0014092, -0.0031916, -0.0013874, -0.0011576, 0.0011350

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009789, upper bound: 0.0010063
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0010063
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0027940, -0.0018163, -0.0006447, 0.0005999
1: -0.0114628, -0.0089936, -0.0114007, -0.0089197, -0.0016361, 0.0015224
2: 0.0279184, 0.0294504, 0.0279570, 0.0294962, -0.0010151, 0.0009445
3: 0.0045712, 0.0074317, 0.0044856, 0.0073597, -0.0017636, 0.0018954
4: -0.0105527, -0.0080410, -0.0104895, -0.0079658, -0.0016642, 0.0015485
5: 0.0097411, 0.0106925, 0.0097651, 0.0107209, -0.0006304, 0.0005865
6: 0.0061896, 0.0098200, 0.0060810, 0.0097286, -0.0022382, 0.0024055
7: 0.9823905, 0.9849308, 0.9823144, 0.9848669, -0.0015662, 0.0016832
8: -0.0054445, -0.0027208, -0.0055260, -0.0027893, -0.0016792, 0.0018047
9: -0.0032024, -0.0014032, -0.0031571, -0.0013494, -0.0011921, 0.0011092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009886, upper bound: 0.0009857
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010164, upper bound: 0.0009857
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0027940, -0.0018176, -0.0006445, 0.0006011
1: -0.0114474, -0.0090018, -0.0114009, -0.0089229, -0.0016355, 0.0015254
2: 0.0279280, 0.0294453, 0.0279569, 0.0294942, -0.0010147, 0.0009464
3: 0.0045807, 0.0074139, 0.0044893, 0.0073599, -0.0017671, 0.0018946
4: -0.0105370, -0.0080494, -0.0104896, -0.0079691, -0.0016636, 0.0015516
5: 0.0097471, 0.0106893, 0.0097650, 0.0107197, -0.0006301, 0.0005877
6: 0.0062017, 0.0097973, 0.0060857, 0.0097288, -0.0022427, 0.0024045
7: 0.9823989, 0.9849149, 0.9823177, 0.9848670, -0.0015693, 0.0016826
8: -0.0054354, -0.0027378, -0.0055224, -0.0027891, -0.0016825, 0.0018040
9: -0.0031911, -0.0014092, -0.0031572, -0.0013517, -0.0011916, 0.0011114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009789, upper bound: 0.0010060
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0010060
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0028126, -0.0018365, -0.0006329, 0.0006486
1: -0.0114514, -0.0089391, -0.0114480, -0.0089709, -0.0016060, 0.0016459
2: 0.0279255, 0.0294842, 0.0279277, 0.0294644, -0.0009964, 0.0010211
3: 0.0045081, 0.0074185, 0.0045449, 0.0074145, -0.0019067, 0.0018605
4: -0.0105410, -0.0079856, -0.0105375, -0.0080179, -0.0016336, 0.0016742
5: 0.0097455, 0.0107135, 0.0097469, 0.0107012, -0.0006188, 0.0006341
6: 0.0061095, 0.0098032, 0.0061562, 0.0097981, -0.0024199, 0.0023612
7: 0.9823344, 0.9849191, 0.9823671, 0.9849154, -0.0016933, 0.0016523
8: -0.0055045, -0.0027334, -0.0054695, -0.0027372, -0.0018155, 0.0017715
9: -0.0031940, -0.0013635, -0.0031915, -0.0013867, -0.0011702, 0.0011993

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009902, upper bound: 0.0009888
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010184, upper bound: 0.0009888
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0028127, -0.0018368, -0.0006344, 0.0006479
1: -0.0114384, -0.0089415, -0.0114481, -0.0089718, -0.0016098, 0.0016441
2: 0.0279336, 0.0294826, 0.0279276, 0.0294639, -0.0009987, 0.0010200
3: 0.0045109, 0.0074034, 0.0045460, 0.0074147, -0.0019046, 0.0018649
4: -0.0105278, -0.0079881, -0.0105377, -0.0080189, -0.0016374, 0.0016724
5: 0.0097505, 0.0107125, 0.0097468, 0.0107008, -0.0006202, 0.0006334
6: 0.0061131, 0.0097840, 0.0061576, 0.0097983, -0.0024172, 0.0023668
7: 0.9823369, 0.9849057, 0.9823681, 0.9849156, -0.0016915, 0.0016561
8: -0.0055018, -0.0027477, -0.0054685, -0.0027371, -0.0018135, 0.0017757
9: -0.0031846, -0.0013653, -0.0031916, -0.0013874, -0.0011729, 0.0011979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009845, upper bound: 0.0010063
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010112, upper bound: 0.0010063
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0027940, -0.0018163, -0.0006518, 0.0006342
1: -0.0114514, -0.0089391, -0.0114007, -0.0089197, -0.0016541, 0.0016094
2: 0.0279255, 0.0294842, 0.0279570, 0.0294962, -0.0010262, 0.0009985
3: 0.0045081, 0.0074185, 0.0044856, 0.0073597, -0.0018644, 0.0019162
4: -0.0105410, -0.0079856, -0.0104895, -0.0079658, -0.0016825, 0.0016370
5: 0.0097455, 0.0107135, 0.0097651, 0.0107209, -0.0006373, 0.0006201
6: 0.0061095, 0.0098032, 0.0060810, 0.0097286, -0.0023662, 0.0024319
7: 0.9823344, 0.9849191, 0.9823144, 0.9848669, -0.0016558, 0.0017017
8: -0.0055045, -0.0027334, -0.0055260, -0.0027893, -0.0017752, 0.0018245
9: -0.0031940, -0.0013635, -0.0031571, -0.0013494, -0.0012052, 0.0011726

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009902, upper bound: 0.0009853
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010184, upper bound: 0.0009853
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0027940, -0.0018176, -0.0006528, 0.0006352
1: -0.0114384, -0.0089415, -0.0114009, -0.0089229, -0.0016566, 0.0016118
2: 0.0279336, 0.0294826, 0.0279569, 0.0294942, -0.0010277, 0.0010000
3: 0.0045109, 0.0074034, 0.0044893, 0.0073599, -0.0018672, 0.0019190
4: -0.0105278, -0.0079881, -0.0104896, -0.0079691, -0.0016850, 0.0016395
5: 0.0097505, 0.0107125, 0.0097650, 0.0107197, -0.0006382, 0.0006210
6: 0.0061131, 0.0097840, 0.0060857, 0.0097288, -0.0023697, 0.0024355
7: 0.9823369, 0.9849057, 0.9823177, 0.9848670, -0.0016582, 0.0017043
8: -0.0055018, -0.0027477, -0.0055224, -0.0027891, -0.0017779, 0.0018272
9: -0.0031846, -0.0013653, -0.0031572, -0.0013517, -0.0012070, 0.0011744

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009846, upper bound: 0.0010060
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010112, upper bound: 0.0010060
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0028707, -0.0018579, -0.0006151, 0.0006912
1: -0.0114628, -0.0089936, -0.0115954, -0.0090252, -0.0015609, 0.0017539
2: 0.0279184, 0.0294504, 0.0278362, 0.0294307, -0.0009684, 0.0010881
3: 0.0045712, 0.0074317, 0.0046079, 0.0075853, -0.0020318, 0.0018083
4: -0.0105527, -0.0080410, -0.0106875, -0.0080732, -0.0015877, 0.0017840
5: 0.0097411, 0.0106925, 0.0096900, 0.0106803, -0.0006014, 0.0006757
6: 0.0061896, 0.0098200, 0.0062361, 0.0100149, -0.0025786, 0.0022949
7: 0.9823905, 0.9849308, 0.9824229, 0.9850672, -0.0018044, 0.0016059
8: -0.0054445, -0.0027208, -0.0054095, -0.0025746, -0.0019346, 0.0017217
9: -0.0032024, -0.0014032, -0.0032990, -0.0014263, -0.0011373, 0.0012779

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009569, upper bound: 0.0009707
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009870, upper bound: 0.0009707
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0028708, -0.0018583, -0.0006162, 0.0006900
1: -0.0114474, -0.0090018, -0.0115956, -0.0090263, -0.0015638, 0.0017509
2: 0.0279280, 0.0294453, 0.0278361, 0.0294300, -0.0009702, 0.0010863
3: 0.0045807, 0.0074139, 0.0046092, 0.0075855, -0.0020283, 0.0018116
4: -0.0105370, -0.0080494, -0.0106877, -0.0080743, -0.0015906, 0.0017809
5: 0.0097471, 0.0106893, 0.0096900, 0.0106798, -0.0006025, 0.0006746
6: 0.0062017, 0.0097973, 0.0062378, 0.0100151, -0.0025742, 0.0022991
7: 0.9823989, 0.9849149, 0.9824242, 0.9850674, -0.0018013, 0.0016088
8: -0.0054354, -0.0027378, -0.0054083, -0.0025744, -0.0019313, 0.0017249
9: -0.0031911, -0.0014092, -0.0032991, -0.0014271, -0.0011394, 0.0012757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009581, upper bound: 0.0009952
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009877, upper bound: 0.0009952
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0028512, -0.0018402, -0.0006293, 0.0006739
1: -0.0114628, -0.0089936, -0.0115459, -0.0089802, -0.0015969, 0.0017100
2: 0.0279184, 0.0294504, 0.0278669, 0.0294587, -0.0009907, 0.0010609
3: 0.0045712, 0.0074317, 0.0045557, 0.0075279, -0.0019810, 0.0018499
4: -0.0105527, -0.0080410, -0.0106371, -0.0080274, -0.0016243, 0.0017394
5: 0.0097411, 0.0106925, 0.0097091, 0.0106976, -0.0006153, 0.0006588
6: 0.0061896, 0.0098200, 0.0061699, 0.0099420, -0.0025141, 0.0023478
7: 0.9823905, 0.9849308, 0.9823767, 0.9850162, -0.0017593, 0.0016429
8: -0.0054445, -0.0027208, -0.0054592, -0.0026292, -0.0018862, 0.0017614
9: -0.0032024, -0.0014032, -0.0032629, -0.0013935, -0.0011635, 0.0012459

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009569, upper bound: 0.0009676
time: 0.74 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009870, upper bound: 0.0009676
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0028513, -0.0018408, -0.0006299, 0.0006739
1: -0.0114474, -0.0090018, -0.0115461, -0.0089818, -0.0015985, 0.0017101
2: 0.0279280, 0.0294453, 0.0278668, 0.0294577, -0.0009917, 0.0010610
3: 0.0045807, 0.0074139, 0.0045576, 0.0075282, -0.0019811, 0.0018518
4: -0.0105370, -0.0080494, -0.0106373, -0.0080290, -0.0016259, 0.0017395
5: 0.0097471, 0.0106893, 0.0097091, 0.0106970, -0.0006159, 0.0006589
6: 0.0062017, 0.0097973, 0.0061723, 0.0099423, -0.0025143, 0.0023502
7: 0.9823989, 0.9849149, 0.9823784, 0.9850164, -0.0017594, 0.0016445
8: -0.0054354, -0.0027378, -0.0054574, -0.0026290, -0.0018863, 0.0017632
9: -0.0031911, -0.0014092, -0.0032630, -0.0013947, -0.0011647, 0.0012460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009581, upper bound: 0.0009915
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009877, upper bound: 0.0009915
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0028707, -0.0018579, -0.0006222, 0.0007255
1: -0.0114514, -0.0089391, -0.0115954, -0.0090252, -0.0015789, 0.0018409
2: 0.0279255, 0.0294842, 0.0278362, 0.0294307, -0.0009796, 0.0011421
3: 0.0045081, 0.0074185, 0.0046079, 0.0075853, -0.0021326, 0.0018291
4: -0.0105410, -0.0079856, -0.0106875, -0.0080732, -0.0016060, 0.0018725
5: 0.0097455, 0.0107135, 0.0096900, 0.0106803, -0.0006083, 0.0007093
6: 0.0061095, 0.0098032, 0.0062361, 0.0100149, -0.0027066, 0.0023213
7: 0.9823344, 0.9849191, 0.9824229, 0.9850672, -0.0018939, 0.0016244
8: -0.0055045, -0.0027334, -0.0054095, -0.0025746, -0.0020306, 0.0017416
9: -0.0031940, -0.0013635, -0.0032990, -0.0014263, -0.0011504, 0.0013413

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009607, upper bound: 0.0009707
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009896, upper bound: 0.0009707
time: 0.75 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0028708, -0.0018583, -0.0006245, 0.0007240
1: -0.0114384, -0.0089415, -0.0115956, -0.0090263, -0.0015848, 0.0018373
2: 0.0279336, 0.0294826, 0.0278361, 0.0294300, -0.0009832, 0.0011399
3: 0.0045109, 0.0074034, 0.0046092, 0.0075855, -0.0021284, 0.0018360
4: -0.0105278, -0.0079881, -0.0106877, -0.0080743, -0.0016121, 0.0018689
5: 0.0097505, 0.0107125, 0.0096900, 0.0106798, -0.0006106, 0.0007079
6: 0.0061131, 0.0097840, 0.0062378, 0.0100151, -0.0027013, 0.0023301
7: 0.9823369, 0.9849057, 0.9824242, 0.9850674, -0.0018902, 0.0016305
8: -0.0055018, -0.0027477, -0.0054083, -0.0025744, -0.0020266, 0.0017481
9: -0.0031846, -0.0013653, -0.0032991, -0.0014271, -0.0011547, 0.0013387

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009641, upper bound: 0.0009952
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009922, upper bound: 0.0009952
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0028512, -0.0018402, -0.0006364, 0.0007082
1: -0.0114514, -0.0089391, -0.0115459, -0.0089802, -0.0016149, 0.0017971
2: 0.0279255, 0.0294842, 0.0278669, 0.0294587, -0.0010019, 0.0011149
3: 0.0045081, 0.0074185, 0.0045557, 0.0075279, -0.0020818, 0.0018708
4: -0.0105410, -0.0079856, -0.0106371, -0.0080274, -0.0016426, 0.0018279
5: 0.0097455, 0.0107135, 0.0097091, 0.0106976, -0.0006222, 0.0006924
6: 0.0061095, 0.0098032, 0.0061699, 0.0099420, -0.0026421, 0.0023743
7: 0.9823344, 0.9849191, 0.9823767, 0.9850162, -0.0018488, 0.0016614
8: -0.0055045, -0.0027334, -0.0054592, -0.0026292, -0.0019822, 0.0017813
9: -0.0031940, -0.0013635, -0.0032629, -0.0013935, -0.0011766, 0.0013094

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009607, upper bound: 0.0009673
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009896, upper bound: 0.0009673
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0028513, -0.0018408, -0.0006382, 0.0007080
1: -0.0114384, -0.0089415, -0.0115461, -0.0089818, -0.0016196, 0.0017966
2: 0.0279336, 0.0294826, 0.0278668, 0.0294577, -0.0010048, 0.0011146
3: 0.0045109, 0.0074034, 0.0045576, 0.0075282, -0.0020812, 0.0018762
4: -0.0105278, -0.0079881, -0.0106373, -0.0080290, -0.0016474, 0.0018274
5: 0.0097505, 0.0107125, 0.0097091, 0.0106970, -0.0006240, 0.0006922
6: 0.0061131, 0.0097840, 0.0061723, 0.0099423, -0.0026413, 0.0023812
7: 0.9823369, 0.9849057, 0.9823784, 0.9850164, -0.0018483, 0.0016662
8: -0.0055018, -0.0027477, -0.0054574, -0.0026290, -0.0019816, 0.0017864
9: -0.0031846, -0.0013653, -0.0032630, -0.0013947, -0.0011801, 0.0013090

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009641, upper bound: 0.0009915
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009922, upper bound: 0.0009915
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028775, -0.0018676, -0.0028126, -0.0018365, -0.0007021, 0.0006028
1: -0.0116127, -0.0090498, -0.0114480, -0.0089709, -0.0017817, 0.0015296
2: 0.0278255, 0.0294155, 0.0279277, 0.0294644, -0.0011054, 0.0009490
3: 0.0046364, 0.0076053, 0.0045449, 0.0074145, -0.0017719, 0.0020640
4: -0.0107051, -0.0080982, -0.0105375, -0.0080179, -0.0018123, 0.0015558
5: 0.0096834, 0.0106708, 0.0097469, 0.0107012, -0.0006864, 0.0005893
6: 0.0062723, 0.0100403, 0.0061562, 0.0097981, -0.0022488, 0.0026195
7: 0.9824483, 0.9850850, 0.9823671, 0.9849154, -0.0015736, 0.0018330
8: -0.0053824, -0.0025555, -0.0054695, -0.0027372, -0.0016872, 0.0019652
9: -0.0033115, -0.0014442, -0.0031915, -0.0013867, -0.0012981, 0.0011145

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009686, upper bound: 0.0009566
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009919, upper bound: 0.0009566
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028705, -0.0018707, -0.0028127, -0.0018368, -0.0007031, 0.0006017
1: -0.0115949, -0.0090578, -0.0114481, -0.0089718, -0.0017842, 0.0015269
2: 0.0278365, 0.0294105, 0.0279276, 0.0294639, -0.0011069, 0.0009473
3: 0.0046456, 0.0075847, 0.0045460, 0.0074147, -0.0017689, 0.0020669
4: -0.0106870, -0.0081063, -0.0105377, -0.0080189, -0.0018148, 0.0015532
5: 0.0096902, 0.0106677, 0.0097468, 0.0107008, -0.0006874, 0.0005883
6: 0.0062840, 0.0100141, 0.0061576, 0.0097983, -0.0022449, 0.0026231
7: 0.9824564, 0.9850667, 0.9823681, 0.9849156, -0.0015709, 0.0018355
8: -0.0053737, -0.0025751, -0.0054685, -0.0027371, -0.0016843, 0.0019680
9: -0.0032986, -0.0014500, -0.0031916, -0.0013874, -0.0013000, 0.0011125

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009737, upper bound: 0.0009877
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009952, upper bound: 0.0009877
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028775, -0.0018676, -0.0028707, -0.0018579, -0.0006338, 0.0006224
1: -0.0116127, -0.0090498, -0.0115954, -0.0090252, -0.0016083, 0.0015794
2: 0.0278255, 0.0294155, 0.0278362, 0.0294307, -0.0009978, 0.0009799
3: 0.0046364, 0.0076053, 0.0046079, 0.0075853, -0.0018296, 0.0018631
4: -0.0107051, -0.0080982, -0.0106875, -0.0080732, -0.0016359, 0.0016065
5: 0.0096834, 0.0106708, 0.0096900, 0.0106803, -0.0006196, 0.0006085
6: 0.0062723, 0.0100403, 0.0062361, 0.0100149, -0.0023221, 0.0023646
7: 0.9824483, 0.9850850, 0.9824229, 0.9850672, -0.0016249, 0.0016546
8: -0.0053824, -0.0025555, -0.0054095, -0.0025746, -0.0017421, 0.0017740
9: -0.0033115, -0.0014442, -0.0032990, -0.0014263, -0.0011718, 0.0011508

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009686, upper bound: 0.0009566
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009919, upper bound: 0.0009566
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028705, -0.0018707, -0.0028708, -0.0018583, -0.0006341, 0.0006222
1: -0.0115949, -0.0090578, -0.0115956, -0.0090263, -0.0016092, 0.0015789
2: 0.0278365, 0.0294105, 0.0278361, 0.0294300, -0.0009984, 0.0009796
3: 0.0046456, 0.0075847, 0.0046092, 0.0075855, -0.0018291, 0.0018642
4: -0.0106870, -0.0081063, -0.0106877, -0.0080743, -0.0016368, 0.0016060
5: 0.0096902, 0.0106677, 0.0096900, 0.0106798, -0.0006200, 0.0006083
6: 0.0062840, 0.0100141, 0.0062378, 0.0100151, -0.0023214, 0.0023659
7: 0.9824564, 0.9850667, 0.9824242, 0.9850674, -0.0016244, 0.0016555
8: -0.0053737, -0.0025751, -0.0054083, -0.0025744, -0.0017416, 0.0017750
9: -0.0032986, -0.0014500, -0.0032991, -0.0014271, -0.0011725, 0.0011504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009737, upper bound: 0.0009869
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009952, upper bound: 0.0009869
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028719, -0.0018488, -0.0028126, -0.0018365, -0.0007062, 0.0006346
1: -0.0115985, -0.0090022, -0.0114480, -0.0089709, -0.0017920, 0.0016105
2: 0.0278343, 0.0294450, 0.0279277, 0.0294644, -0.0011118, 0.0009991
3: 0.0045812, 0.0075889, 0.0045449, 0.0074145, -0.0018656, 0.0020760
4: -0.0106907, -0.0080498, -0.0105375, -0.0080179, -0.0018228, 0.0016381
5: 0.0096888, 0.0106891, 0.0097469, 0.0107012, -0.0006904, 0.0006205
6: 0.0062023, 0.0100194, 0.0061562, 0.0097981, -0.0023677, 0.0026347
7: 0.9823994, 0.9850704, 0.9823671, 0.9849154, -0.0016568, 0.0018436
8: -0.0054349, -0.0025711, -0.0054695, -0.0027372, -0.0017764, 0.0019767
9: -0.0033012, -0.0014095, -0.0031915, -0.0013867, -0.0013057, 0.0011734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009716, upper bound: 0.0009562
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009932, upper bound: 0.0009561
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028658, -0.0018499, -0.0028127, -0.0018368, -0.0007071, 0.0006334
1: -0.0115828, -0.0090050, -0.0114481, -0.0089718, -0.0017944, 0.0016074
2: 0.0278440, 0.0294433, 0.0279276, 0.0294639, -0.0011132, 0.0009972
3: 0.0045844, 0.0075708, 0.0045460, 0.0074147, -0.0018621, 0.0020787
4: -0.0106747, -0.0080526, -0.0105377, -0.0080189, -0.0018252, 0.0016350
5: 0.0096949, 0.0106881, 0.0097468, 0.0107008, -0.0006913, 0.0006193
6: 0.0062063, 0.0099964, 0.0061576, 0.0097983, -0.0023632, 0.0026382
7: 0.9824021, 0.9850543, 0.9823681, 0.9849156, -0.0016537, 0.0018461
8: -0.0054319, -0.0025884, -0.0054685, -0.0027371, -0.0017730, 0.0019793
9: -0.0032898, -0.0014115, -0.0031916, -0.0013874, -0.0013074, 0.0011711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009763, upper bound: 0.0009877
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009979, upper bound: 0.0009877
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028719, -0.0018488, -0.0028707, -0.0018579, -0.0006409, 0.0006565
1: -0.0115985, -0.0090022, -0.0115954, -0.0090252, -0.0016263, 0.0016660
2: 0.0278343, 0.0294450, 0.0278362, 0.0294307, -0.0010090, 0.0010336
3: 0.0045812, 0.0075889, 0.0046079, 0.0075853, -0.0019300, 0.0018840
4: -0.0106907, -0.0080498, -0.0106875, -0.0080732, -0.0016543, 0.0016946
5: 0.0096888, 0.0106891, 0.0096900, 0.0106803, -0.0006266, 0.0006419
6: 0.0062023, 0.0100194, 0.0062361, 0.0100149, -0.0024494, 0.0023911
7: 0.9823994, 0.9850704, 0.9824229, 0.9850672, -0.0017140, 0.0016732
8: -0.0054349, -0.0025711, -0.0054095, -0.0025746, -0.0018377, 0.0017939
9: -0.0033012, -0.0014095, -0.0032990, -0.0014263, -0.0011850, 0.0012139

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009716, upper bound: 0.0009562
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009932, upper bound: 0.0009562
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028658, -0.0018499, -0.0028708, -0.0018583, -0.0006422, 0.0006559
1: -0.0115828, -0.0090050, -0.0115956, -0.0090263, -0.0016296, 0.0016644
2: 0.0278440, 0.0294433, 0.0278361, 0.0294300, -0.0010110, 0.0010326
3: 0.0045844, 0.0075708, 0.0046092, 0.0075855, -0.0019281, 0.0018879
4: -0.0106747, -0.0080526, -0.0106877, -0.0080743, -0.0016576, 0.0016930
5: 0.0096949, 0.0106881, 0.0096900, 0.0106798, -0.0006279, 0.0006412
6: 0.0062063, 0.0099964, 0.0062378, 0.0100151, -0.0024470, 0.0023959
7: 0.9824021, 0.9850543, 0.9824242, 0.9850674, -0.0017123, 0.0016766
8: -0.0054319, -0.0025884, -0.0054083, -0.0025744, -0.0018359, 0.0017975
9: -0.0032898, -0.0014115, -0.0032991, -0.0014271, -0.0011874, 0.0012127

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009763, upper bound: 0.0009869
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009979, upper bound: 0.0009869
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028775, -0.0018676, -0.0027940, -0.0018163, -0.0007210, 0.0005884
1: -0.0116127, -0.0090498, -0.0114007, -0.0089197, -0.0018298, 0.0014930
2: 0.0278255, 0.0294155, 0.0279570, 0.0294962, -0.0011352, 0.0009263
3: 0.0046364, 0.0076053, 0.0044856, 0.0073597, -0.0017296, 0.0021197
4: -0.0107051, -0.0080982, -0.0104895, -0.0079658, -0.0018612, 0.0015187
5: 0.0096834, 0.0106708, 0.0097651, 0.0107209, -0.0007050, 0.0005752
6: 0.0062723, 0.0100403, 0.0060810, 0.0097286, -0.0021951, 0.0026902
7: 0.9824483, 0.9850850, 0.9823144, 0.9848669, -0.0015360, 0.0018824
8: -0.0053824, -0.0025555, -0.0055260, -0.0027893, -0.0016469, 0.0020183
9: -0.0033115, -0.0014442, -0.0031571, -0.0013494, -0.0013332, 0.0010878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009686, upper bound: 0.0009551
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009919, upper bound: 0.0009551
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028705, -0.0018707, -0.0027940, -0.0018176, -0.0007215, 0.0005890
1: -0.0115949, -0.0090578, -0.0114009, -0.0089229, -0.0018309, 0.0014946
2: 0.0278365, 0.0294105, 0.0279569, 0.0294942, -0.0011359, 0.0009273
3: 0.0046456, 0.0075847, 0.0044893, 0.0073599, -0.0017315, 0.0021210
4: -0.0106870, -0.0081063, -0.0104896, -0.0079691, -0.0018624, 0.0015203
5: 0.0096902, 0.0106677, 0.0097650, 0.0107197, -0.0007054, 0.0005758
6: 0.0062840, 0.0100141, 0.0060857, 0.0097288, -0.0021974, 0.0026919
7: 0.9824564, 0.9850667, 0.9823177, 0.9848670, -0.0015377, 0.0018836
8: -0.0053737, -0.0025751, -0.0055224, -0.0027891, -0.0016486, 0.0020196
9: -0.0032986, -0.0014500, -0.0031572, -0.0013517, -0.0013340, 0.0010890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009737, upper bound: 0.0009888
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009952, upper bound: 0.0009888
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028775, -0.0018676, -0.0028512, -0.0018402, -0.0006530, 0.0006079
1: -0.0116127, -0.0090498, -0.0115459, -0.0089802, -0.0016572, 0.0015427
2: 0.0278255, 0.0294155, 0.0278669, 0.0294587, -0.0010281, 0.0009571
3: 0.0046364, 0.0076053, 0.0045557, 0.0075279, -0.0017871, 0.0019197
4: -0.0107051, -0.0080982, -0.0106371, -0.0080274, -0.0016856, 0.0015692
5: 0.0096834, 0.0106708, 0.0097091, 0.0106976, -0.0006385, 0.0005944
6: 0.0062723, 0.0100403, 0.0061699, 0.0099420, -0.0022681, 0.0024364
7: 0.9824483, 0.9850850, 0.9823767, 0.9850162, -0.0015871, 0.0017049
8: -0.0053824, -0.0025555, -0.0054592, -0.0026292, -0.0017016, 0.0018279
9: -0.0033115, -0.0014442, -0.0032629, -0.0013935, -0.0012074, 0.0011240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009686, upper bound: 0.0009551
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009919, upper bound: 0.0009551
time: 0.73 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028705, -0.0018707, -0.0028513, -0.0018408, -0.0006535, 0.0006092
1: -0.0115949, -0.0090578, -0.0115461, -0.0089818, -0.0016584, 0.0015460
2: 0.0278365, 0.0294105, 0.0278668, 0.0294577, -0.0010289, 0.0009591
3: 0.0046456, 0.0075847, 0.0045576, 0.0075282, -0.0017909, 0.0019211
4: -0.0106870, -0.0081063, -0.0106373, -0.0080290, -0.0016868, 0.0015725
5: 0.0096902, 0.0106677, 0.0097091, 0.0106970, -0.0006389, 0.0005956
6: 0.0062840, 0.0100141, 0.0061723, 0.0099423, -0.0022729, 0.0024382
7: 0.9824564, 0.9850667, 0.9823784, 0.9850164, -0.0015905, 0.0017061
8: -0.0053737, -0.0025751, -0.0054574, -0.0026290, -0.0017052, 0.0018292
9: -0.0032986, -0.0014500, -0.0032630, -0.0013947, -0.0012083, 0.0011264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009737, upper bound: 0.0009854
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009952, upper bound: 0.0009854
time: 0.71 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028719, -0.0018488, -0.0027940, -0.0018163, -0.0007251, 0.0006202
1: -0.0115985, -0.0090022, -0.0114007, -0.0089197, -0.0018401, 0.0015739
2: 0.0278343, 0.0294450, 0.0279570, 0.0294962, -0.0011416, 0.0009765
3: 0.0045812, 0.0075889, 0.0044856, 0.0073597, -0.0018233, 0.0021317
4: -0.0106907, -0.0080498, -0.0104895, -0.0079658, -0.0018717, 0.0016009
5: 0.0096888, 0.0106891, 0.0097651, 0.0107209, -0.0007090, 0.0006064
6: 0.0062023, 0.0100194, 0.0060810, 0.0097286, -0.0023140, 0.0027054
7: 0.9823994, 0.9850704, 0.9823144, 0.9848669, -0.0016192, 0.0018931
8: -0.0054349, -0.0025711, -0.0055260, -0.0027893, -0.0017361, 0.0020297
9: -0.0033012, -0.0014095, -0.0031571, -0.0013494, -0.0013407, 0.0011468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009716, upper bound: 0.0009544
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009932, upper bound: 0.0009544
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028658, -0.0018499, -0.0027940, -0.0018176, -0.0007255, 0.0006207
1: -0.0115828, -0.0090050, -0.0114009, -0.0089229, -0.0018412, 0.0015751
2: 0.0278440, 0.0294433, 0.0279569, 0.0294942, -0.0011423, 0.0009772
3: 0.0045844, 0.0075708, 0.0044893, 0.0073599, -0.0018246, 0.0021329
4: -0.0106747, -0.0080526, -0.0104896, -0.0079691, -0.0018728, 0.0016021
5: 0.0096949, 0.0106881, 0.0097650, 0.0107197, -0.0007094, 0.0006068
6: 0.0062063, 0.0099964, 0.0060857, 0.0097288, -0.0023157, 0.0027069
7: 0.9824021, 0.9850543, 0.9823177, 0.9848670, -0.0016204, 0.0018942
8: -0.0054319, -0.0025884, -0.0055224, -0.0027891, -0.0017373, 0.0020308
9: -0.0032898, -0.0014115, -0.0031572, -0.0013517, -0.0013415, 0.0011476

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009763, upper bound: 0.0009888
time: 0.75 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009979, upper bound: 0.0009888
time: 0.72 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028719, -0.0018488, -0.0028512, -0.0018402, -0.0006601, 0.0006420
1: -0.0115985, -0.0090022, -0.0115459, -0.0089802, -0.0016752, 0.0016293
2: 0.0278343, 0.0294450, 0.0278669, 0.0294587, -0.0010393, 0.0010108
3: 0.0045812, 0.0075889, 0.0045557, 0.0075279, -0.0018875, 0.0019406
4: -0.0106907, -0.0080498, -0.0106371, -0.0080274, -0.0017040, 0.0016573
5: 0.0096888, 0.0106891, 0.0097091, 0.0106976, -0.0006454, 0.0006277
6: 0.0062023, 0.0100194, 0.0061699, 0.0099420, -0.0023954, 0.0024629
7: 0.9823994, 0.9850704, 0.9823767, 0.9850162, -0.0016762, 0.0017234
8: -0.0054349, -0.0025711, -0.0054592, -0.0026292, -0.0017972, 0.0018478
9: -0.0033012, -0.0014095, -0.0032629, -0.0013935, -0.0012206, 0.0011871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009716, upper bound: 0.0009544
time: 0.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009932, upper bound: 0.0009544
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028658, -0.0018499, -0.0028513, -0.0018408, -0.0006616, 0.0006429
1: -0.0115828, -0.0090050, -0.0115461, -0.0089818, -0.0016788, 0.0016314
2: 0.0278440, 0.0294433, 0.0278668, 0.0294577, -0.0010415, 0.0010121
3: 0.0045844, 0.0075708, 0.0045576, 0.0075282, -0.0018899, 0.0019448
4: -0.0106747, -0.0080526, -0.0106373, -0.0080290, -0.0017076, 0.0016594
5: 0.0096949, 0.0106881, 0.0097091, 0.0106970, -0.0006468, 0.0006285
6: 0.0062063, 0.0099964, 0.0061723, 0.0099423, -0.0023985, 0.0024682
7: 0.9824021, 0.9850543, 0.9823784, 0.9850164, -0.0016784, 0.0017271
8: -0.0054319, -0.0025884, -0.0054574, -0.0026290, -0.0017995, 0.0018518
9: -0.0032898, -0.0014115, -0.0032630, -0.0013947, -0.0012232, 0.0011887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009763, upper bound: 0.0009854
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009979, upper bound: 0.0009854
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0028091, -0.0018137, -0.0006609, 0.0006230
1: -0.0114628, -0.0089936, -0.0114390, -0.0089130, -0.0016772, 0.0015808
2: 0.0279184, 0.0294504, 0.0279332, 0.0295004, -0.0010406, 0.0009808
3: 0.0045712, 0.0074317, 0.0044779, 0.0074041, -0.0018313, 0.0019430
4: -0.0105527, -0.0080410, -0.0105284, -0.0079590, -0.0017060, 0.0016080
5: 0.0097411, 0.0106925, 0.0097503, 0.0107235, -0.0006462, 0.0006091
6: 0.0061896, 0.0098200, 0.0060711, 0.0097849, -0.0023242, 0.0024659
7: 0.9823905, 0.9849308, 0.9823076, 0.9849062, -0.0016264, 0.0017255
8: -0.0054445, -0.0027208, -0.0055333, -0.0027471, -0.0017437, 0.0018500
9: -0.0032024, -0.0014032, -0.0031850, -0.0013445, -0.0012221, 0.0011518

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009886, upper bound: 0.0009969
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010164, upper bound: 0.0009969
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0028091, -0.0018135, -0.0006625, 0.0006206
1: -0.0114474, -0.0090018, -0.0114391, -0.0089125, -0.0016811, 0.0015748
2: 0.0279280, 0.0294453, 0.0279331, 0.0295007, -0.0010430, 0.0009770
3: 0.0045807, 0.0074139, 0.0044772, 0.0074043, -0.0018243, 0.0019475
4: -0.0105370, -0.0080494, -0.0105285, -0.0079585, -0.0017100, 0.0016018
5: 0.0097471, 0.0106893, 0.0097503, 0.0107237, -0.0006477, 0.0006067
6: 0.0062017, 0.0097973, 0.0060703, 0.0097851, -0.0023153, 0.0024716
7: 0.9823989, 0.9849149, 0.9823070, 0.9849063, -0.0016201, 0.0017295
8: -0.0054354, -0.0027378, -0.0055339, -0.0027469, -0.0017370, 0.0018543
9: -0.0031911, -0.0014092, -0.0031851, -0.0013441, -0.0012249, 0.0011474

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009789, upper bound: 0.0010112
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010063, upper bound: 0.0010112
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0027883, -0.0017961, -0.0006852, 0.0006106
1: -0.0114628, -0.0089936, -0.0113862, -0.0088685, -0.0017388, 0.0015496
2: 0.0279184, 0.0294504, 0.0279660, 0.0295279, -0.0010788, 0.0009614
3: 0.0045712, 0.0074317, 0.0044264, 0.0073430, -0.0017951, 0.0020143
4: -0.0105527, -0.0080410, -0.0104747, -0.0079138, -0.0017687, 0.0015762
5: 0.0097411, 0.0106925, 0.0097706, 0.0107406, -0.0006699, 0.0005970
6: 0.0061896, 0.0098200, 0.0060057, 0.0097073, -0.0022782, 0.0025564
7: 0.9823905, 0.9849308, 0.9822618, 0.9848520, -0.0015942, 0.0017889
8: -0.0054445, -0.0027208, -0.0055824, -0.0028053, -0.0017092, 0.0019180
9: -0.0032024, -0.0014032, -0.0031465, -0.0013121, -0.0012669, 0.0011290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009886, upper bound: 0.0009952
time: 0.73 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010164, upper bound: 0.0009952
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0027884, -0.0017963, -0.0006855, 0.0006101
1: -0.0114474, -0.0090018, -0.0113864, -0.0088689, -0.0017396, 0.0015483
2: 0.0279280, 0.0294453, 0.0279658, 0.0295277, -0.0010792, 0.0009606
3: 0.0045807, 0.0074139, 0.0044268, 0.0073432, -0.0017937, 0.0020152
4: -0.0105370, -0.0080494, -0.0104749, -0.0079142, -0.0017694, 0.0015749
5: 0.0097471, 0.0106893, 0.0097706, 0.0107405, -0.0006702, 0.0005965
6: 0.0062017, 0.0097973, 0.0060063, 0.0097076, -0.0022764, 0.0025576
7: 0.9823989, 0.9849149, 0.9822623, 0.9848522, -0.0015929, 0.0017897
8: -0.0054354, -0.0027378, -0.0055819, -0.0028051, -0.0017078, 0.0019188
9: -0.0031911, -0.0014092, -0.0031467, -0.0013124, -0.0012675, 0.0011281

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010043, upper bound: 0.0010231
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010043, upper bound: 0.0010231
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0028091, -0.0018137, -0.0006377, 0.0006236
1: -0.0114514, -0.0089391, -0.0114390, -0.0089130, -0.0016183, 0.0015826
2: 0.0279255, 0.0294842, 0.0279332, 0.0295004, -0.0010040, 0.0009818
3: 0.0045081, 0.0074185, 0.0044779, 0.0074041, -0.0018333, 0.0018748
4: -0.0105410, -0.0079856, -0.0105284, -0.0079590, -0.0016461, 0.0016097
5: 0.0097455, 0.0107135, 0.0097503, 0.0107235, -0.0006235, 0.0006097
6: 0.0061095, 0.0098032, 0.0060711, 0.0097849, -0.0023267, 0.0023793
7: 0.9823344, 0.9849191, 0.9823076, 0.9849062, -0.0016281, 0.0016649
8: -0.0055045, -0.0027334, -0.0055333, -0.0027471, -0.0017456, 0.0017851
9: -0.0031940, -0.0013635, -0.0031850, -0.0013445, -0.0011791, 0.0011531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009962, upper bound: 0.0009954
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010242, upper bound: 0.0009954
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0028091, -0.0018135, -0.0006352, 0.0006270
1: -0.0114384, -0.0089415, -0.0114391, -0.0089125, -0.0016118, 0.0015910
2: 0.0279336, 0.0294826, 0.0279331, 0.0295007, -0.0010000, 0.0009871
3: 0.0045109, 0.0074034, 0.0044772, 0.0074043, -0.0018432, 0.0018672
4: -0.0105278, -0.0079881, -0.0105285, -0.0079585, -0.0016395, 0.0016184
5: 0.0097505, 0.0107125, 0.0097503, 0.0107237, -0.0006210, 0.0006130
6: 0.0061131, 0.0097840, 0.0060703, 0.0097851, -0.0023392, 0.0023698
7: 0.9823369, 0.9849057, 0.9823070, 0.9849063, -0.0016369, 0.0016582
8: -0.0055018, -0.0027477, -0.0055339, -0.0027469, -0.0017550, 0.0017779
9: -0.0031846, -0.0013653, -0.0031851, -0.0013441, -0.0011744, 0.0011593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009906, upper bound: 0.0010134
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010176, upper bound: 0.0010134
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0027883, -0.0017961, -0.0006567, 0.0006090
1: -0.0114514, -0.0089391, -0.0113862, -0.0088685, -0.0016664, 0.0015455
2: 0.0279255, 0.0294842, 0.0279660, 0.0295279, -0.0010338, 0.0009588
3: 0.0045081, 0.0074185, 0.0044264, 0.0073430, -0.0017904, 0.0019304
4: -0.0105410, -0.0079856, -0.0104747, -0.0079138, -0.0016950, 0.0015720
5: 0.0097455, 0.0107135, 0.0097706, 0.0107406, -0.0006420, 0.0005954
6: 0.0061095, 0.0098032, 0.0060057, 0.0097073, -0.0022723, 0.0024499
7: 0.9823344, 0.9849191, 0.9822618, 0.9848520, -0.0015900, 0.0017143
8: -0.0055045, -0.0027334, -0.0055824, -0.0028053, -0.0017047, 0.0018381
9: -0.0031940, -0.0013635, -0.0031465, -0.0013121, -0.0012141, 0.0011261

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009962, upper bound: 0.0009918
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010242, upper bound: 0.0009918
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0027884, -0.0017963, -0.0006537, 0.0006126
1: -0.0114384, -0.0089415, -0.0113864, -0.0088689, -0.0016587, 0.0015545
2: 0.0279336, 0.0294826, 0.0279658, 0.0295277, -0.0010291, 0.0009644
3: 0.0045109, 0.0074034, 0.0044268, 0.0073432, -0.0018008, 0.0019216
4: -0.0105278, -0.0079881, -0.0104749, -0.0079142, -0.0016872, 0.0015812
5: 0.0097505, 0.0107125, 0.0097706, 0.0107405, -0.0006391, 0.0005989
6: 0.0061131, 0.0097840, 0.0060063, 0.0097076, -0.0022855, 0.0024387
7: 0.9823369, 0.9849057, 0.9822623, 0.9848522, -0.0015993, 0.0017065
8: -0.0055018, -0.0027477, -0.0055819, -0.0028051, -0.0017147, 0.0018296
9: -0.0031846, -0.0013653, -0.0031467, -0.0013124, -0.0012086, 0.0011326

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010184, upper bound: 0.0010266
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010184, upper bound: 0.0010266
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0028660, -0.0018387, -0.0006475, 0.0006957
1: -0.0114628, -0.0089936, -0.0115834, -0.0089766, -0.0016432, 0.0017655
2: 0.0279184, 0.0294504, 0.0278436, 0.0294609, -0.0010194, 0.0010953
3: 0.0045712, 0.0074317, 0.0045515, 0.0075714, -0.0020452, 0.0019035
4: -0.0105527, -0.0080410, -0.0106753, -0.0080237, -0.0016714, 0.0017958
5: 0.0097411, 0.0106925, 0.0096947, 0.0106990, -0.0006331, 0.0006802
6: 0.0061896, 0.0098200, 0.0061646, 0.0099972, -0.0025957, 0.0024158
7: 0.9823905, 0.9849308, 0.9823729, 0.9850549, -0.0018163, 0.0016905
8: -0.0054445, -0.0027208, -0.0054632, -0.0025878, -0.0019474, 0.0018125
9: -0.0032024, -0.0014032, -0.0032902, -0.0013908, -0.0011972, 0.0012864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009569, upper bound: 0.0009774
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009870, upper bound: 0.0009774
time: 0.74 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0028660, -0.0018387, -0.0006493, 0.0006938
1: -0.0114474, -0.0090018, -0.0115835, -0.0089765, -0.0016476, 0.0017606
2: 0.0279280, 0.0294453, 0.0278435, 0.0294610, -0.0010222, 0.0010923
3: 0.0045807, 0.0074139, 0.0045514, 0.0075716, -0.0020396, 0.0019087
4: -0.0105370, -0.0080494, -0.0106754, -0.0080236, -0.0016759, 0.0017908
5: 0.0097471, 0.0106893, 0.0096946, 0.0106990, -0.0006348, 0.0006783
6: 0.0062017, 0.0097973, 0.0061645, 0.0099974, -0.0025885, 0.0024224
7: 0.9823989, 0.9849149, 0.9823729, 0.9850550, -0.0018113, 0.0016950
8: -0.0054354, -0.0027378, -0.0054633, -0.0025877, -0.0019420, 0.0018174
9: -0.0031911, -0.0014092, -0.0032903, -0.0013908, -0.0012005, 0.0012828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009581, upper bound: 0.0009980
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009877, upper bound: 0.0009980
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028185, -0.0018454, -0.0028454, -0.0018232, -0.0006654, 0.0006799
1: -0.0114628, -0.0089936, -0.0115311, -0.0089371, -0.0016887, 0.0017252
2: 0.0279184, 0.0294504, 0.0278761, 0.0294854, -0.0010477, 0.0010703
3: 0.0045712, 0.0074317, 0.0045058, 0.0075108, -0.0019986, 0.0019562
4: -0.0105527, -0.0080410, -0.0106221, -0.0079836, -0.0017177, 0.0017548
5: 0.0097411, 0.0106925, 0.0097148, 0.0107142, -0.0006506, 0.0006647
6: 0.0061896, 0.0098200, 0.0061066, 0.0099203, -0.0025365, 0.0024827
7: 0.9823905, 0.9849308, 0.9823324, 0.9850010, -0.0017749, 0.0017373
8: -0.0054445, -0.0027208, -0.0055067, -0.0026455, -0.0019030, 0.0018626
9: -0.0032024, -0.0014032, -0.0032521, -0.0013621, -0.0012304, 0.0012570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009569, upper bound: 0.0009757
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009870, upper bound: 0.0009757
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028124, -0.0018487, -0.0028454, -0.0018231, -0.0006671, 0.0006780
1: -0.0114474, -0.0090018, -0.0115313, -0.0089368, -0.0016928, 0.0017205
2: 0.0279280, 0.0294453, 0.0278760, 0.0294856, -0.0010502, 0.0010674
3: 0.0045807, 0.0074139, 0.0045055, 0.0075110, -0.0019932, 0.0019611
4: -0.0105370, -0.0080494, -0.0106223, -0.0079833, -0.0017219, 0.0017501
5: 0.0097471, 0.0106893, 0.0097148, 0.0107143, -0.0006522, 0.0006629
6: 0.0062017, 0.0097973, 0.0061062, 0.0099206, -0.0025296, 0.0024888
7: 0.9823989, 0.9849149, 0.9823321, 0.9850013, -0.0017701, 0.0017416
8: -0.0054354, -0.0027378, -0.0055070, -0.0026453, -0.0018978, 0.0018672
9: -0.0031911, -0.0014092, -0.0032522, -0.0013619, -0.0012334, 0.0012536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009581, upper bound: 0.0009922
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009877, upper bound: 0.0009922
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0028660, -0.0018387, -0.0006299, 0.0007018
1: -0.0114514, -0.0089391, -0.0115834, -0.0089766, -0.0015986, 0.0017810
2: 0.0279255, 0.0294842, 0.0278436, 0.0294609, -0.0009918, 0.0011049
3: 0.0045081, 0.0074185, 0.0045515, 0.0075714, -0.0020632, 0.0018519
4: -0.0105410, -0.0079856, -0.0106753, -0.0080237, -0.0016260, 0.0018115
5: 0.0097455, 0.0107135, 0.0096947, 0.0106990, -0.0006159, 0.0006862
6: 0.0061095, 0.0098032, 0.0061646, 0.0099972, -0.0026184, 0.0023503
7: 0.9823344, 0.9849191, 0.9823729, 0.9850549, -0.0018322, 0.0016446
8: -0.0055045, -0.0027334, -0.0054632, -0.0025878, -0.0019645, 0.0017633
9: -0.0031940, -0.0013635, -0.0032902, -0.0013908, -0.0011647, 0.0012976

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009635, upper bound: 0.0009739
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009930, upper bound: 0.0009739
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0028660, -0.0018387, -0.0006282, 0.0007022
1: -0.0114384, -0.0089415, -0.0115835, -0.0089765, -0.0015941, 0.0017820
2: 0.0279336, 0.0294826, 0.0278435, 0.0294610, -0.0009890, 0.0011056
3: 0.0045109, 0.0074034, 0.0045514, 0.0075716, -0.0020644, 0.0018466
4: -0.0105278, -0.0079881, -0.0106754, -0.0080236, -0.0016214, 0.0018126
5: 0.0097505, 0.0107125, 0.0096946, 0.0106990, -0.0006141, 0.0006866
6: 0.0061131, 0.0097840, 0.0061645, 0.0099974, -0.0026199, 0.0023436
7: 0.9823369, 0.9849057, 0.9823729, 0.9850550, -0.0018333, 0.0016399
8: -0.0055018, -0.0027477, -0.0054633, -0.0025877, -0.0019656, 0.0017583
9: -0.0031846, -0.0013653, -0.0032903, -0.0013908, -0.0011614, 0.0012984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009705, upper bound: 0.0010009
time: 0.72 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009982, upper bound: 0.0010009
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028140, -0.0018240, -0.0028454, -0.0018232, -0.0006435, 0.0006843
1: -0.0114514, -0.0089391, -0.0115311, -0.0089371, -0.0016330, 0.0017365
2: 0.0279255, 0.0294842, 0.0278761, 0.0294854, -0.0010131, 0.0010773
3: 0.0045081, 0.0074185, 0.0045058, 0.0075108, -0.0020116, 0.0018917
4: -0.0105410, -0.0079856, -0.0106221, -0.0079836, -0.0016610, 0.0017663
5: 0.0097455, 0.0107135, 0.0097148, 0.0107142, -0.0006291, 0.0006690
6: 0.0061095, 0.0098032, 0.0061066, 0.0099203, -0.0025530, 0.0024008
7: 0.9823344, 0.9849191, 0.9823324, 0.9850010, -0.0017865, 0.0016800
8: -0.0055045, -0.0027334, -0.0055067, -0.0026455, -0.0019154, 0.0018012
9: -0.0031940, -0.0013635, -0.0032521, -0.0013621, -0.0011898, 0.0012652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009635, upper bound: 0.0009700
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009930, upper bound: 0.0009700
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028088, -0.0018249, -0.0028454, -0.0018231, -0.0006417, 0.0006856
1: -0.0114384, -0.0089415, -0.0115313, -0.0089368, -0.0016284, 0.0017397
2: 0.0279336, 0.0294826, 0.0278760, 0.0294856, -0.0010102, 0.0010793
3: 0.0045109, 0.0074034, 0.0045055, 0.0075110, -0.0020153, 0.0018864
4: -0.0105278, -0.0079881, -0.0106223, -0.0079833, -0.0016563, 0.0017696
5: 0.0097505, 0.0107125, 0.0097148, 0.0107143, -0.0006274, 0.0006703
6: 0.0061131, 0.0097840, 0.0061062, 0.0099206, -0.0025577, 0.0023941
7: 0.9823369, 0.9849057, 0.9823321, 0.9850013, -0.0017898, 0.0016753
8: -0.0055018, -0.0027477, -0.0055070, -0.0026453, -0.0019189, 0.0017961
9: -0.0031846, -0.0013653, -0.0032522, -0.0013619, -0.0011864, 0.0012676

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009705, upper bound: 0.0009968
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009981, upper bound: 0.0009968
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028776, -0.0018676, -0.0028091, -0.0018137, -0.0007372, 0.0006114
1: -0.0116128, -0.0090498, -0.0114390, -0.0089130, -0.0018708, 0.0015515
2: 0.0278254, 0.0294155, 0.0279332, 0.0295004, -0.0011606, 0.0009626
3: 0.0046364, 0.0076055, 0.0044779, 0.0074041, -0.0017973, 0.0021672
4: -0.0107052, -0.0080982, -0.0105284, -0.0079590, -0.0019029, 0.0015781
5: 0.0096833, 0.0106708, 0.0097503, 0.0107235, -0.0007208, 0.0005978
6: 0.0062723, 0.0100405, 0.0060711, 0.0097849, -0.0022811, 0.0027504
7: 0.9824483, 0.9850851, 0.9823076, 0.9849062, -0.0015962, 0.0019246
8: -0.0053824, -0.0025553, -0.0055333, -0.0027471, -0.0017114, 0.0020635
9: -0.0033117, -0.0014442, -0.0031850, -0.0013445, -0.0013631, 0.0011304

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009686, upper bound: 0.0009667
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009919, upper bound: 0.0009667
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028706, -0.0018707, -0.0028091, -0.0018135, -0.0007393, 0.0006085
1: -0.0115951, -0.0090578, -0.0114391, -0.0089125, -0.0018760, 0.0015440
2: 0.0278364, 0.0294105, 0.0279331, 0.0295007, -0.0011639, 0.0009579
3: 0.0046456, 0.0075849, 0.0044772, 0.0074043, -0.0017887, 0.0021732
4: -0.0106872, -0.0081063, -0.0105285, -0.0079585, -0.0019082, 0.0015705
5: 0.0096902, 0.0106677, 0.0097503, 0.0107237, -0.0007228, 0.0005949
6: 0.0062840, 0.0100144, 0.0060703, 0.0097851, -0.0022701, 0.0027581
7: 0.9824564, 0.9850669, 0.9823070, 0.9849063, -0.0015885, 0.0019300
8: -0.0053737, -0.0025749, -0.0055339, -0.0027469, -0.0017031, 0.0020693
9: -0.0032987, -0.0014500, -0.0031851, -0.0013441, -0.0013669, 0.0011250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009737, upper bound: 0.0009922
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009952, upper bound: 0.0009922
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028776, -0.0018676, -0.0028660, -0.0018387, -0.0006688, 0.0006308
1: -0.0116128, -0.0090498, -0.0115834, -0.0089766, -0.0016971, 0.0016007
2: 0.0278254, 0.0294155, 0.0278436, 0.0294609, -0.0010529, 0.0009931
3: 0.0046364, 0.0076055, 0.0045515, 0.0075714, -0.0018544, 0.0019660
4: -0.0107052, -0.0080982, -0.0106753, -0.0080237, -0.0017262, 0.0016282
5: 0.0096833, 0.0106708, 0.0096947, 0.0106990, -0.0006538, 0.0006167
6: 0.0062723, 0.0100405, 0.0061646, 0.0099972, -0.0023534, 0.0024951
7: 0.9824483, 0.9850851, 0.9823729, 0.9850549, -0.0016468, 0.0017459
8: -0.0053824, -0.0025553, -0.0054632, -0.0025878, -0.0017656, 0.0018719
9: -0.0033117, -0.0014442, -0.0032902, -0.0013908, -0.0012365, 0.0011663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009686, upper bound: 0.0009667
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009919, upper bound: 0.0009667
time: 0.77 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028706, -0.0018707, -0.0028660, -0.0018387, -0.0006709, 0.0006290
1: -0.0115951, -0.0090578, -0.0115835, -0.0089765, -0.0017024, 0.0015961
2: 0.0278364, 0.0294105, 0.0278435, 0.0294610, -0.0010562, 0.0009902
3: 0.0046456, 0.0075849, 0.0045514, 0.0075716, -0.0018490, 0.0019721
4: -0.0106872, -0.0081063, -0.0106754, -0.0080236, -0.0017316, 0.0016235
5: 0.0096902, 0.0106677, 0.0096946, 0.0106990, -0.0006559, 0.0006149
6: 0.0062840, 0.0100144, 0.0061645, 0.0099974, -0.0023467, 0.0025029
7: 0.9824564, 0.9850669, 0.9823729, 0.9850550, -0.0016421, 0.0017514
8: -0.0053737, -0.0025749, -0.0054633, -0.0025877, -0.0017606, 0.0018778
9: -0.0032987, -0.0014500, -0.0032903, -0.0013908, -0.0012404, 0.0011630

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009737, upper bound: 0.0009913
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009952, upper bound: 0.0009913
time: 0.75 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028723, -0.0018488, -0.0028091, -0.0018137, -0.0007125, 0.0006151
1: -0.0115995, -0.0090022, -0.0114390, -0.0089130, -0.0018079, 0.0015610
2: 0.0278337, 0.0294450, 0.0279332, 0.0295004, -0.0011217, 0.0009684
3: 0.0045812, 0.0075900, 0.0044779, 0.0074041, -0.0018083, 0.0020944
4: -0.0106916, -0.0080498, -0.0105284, -0.0079590, -0.0018390, 0.0015878
5: 0.0096885, 0.0106891, 0.0097503, 0.0107235, -0.0006966, 0.0006014
6: 0.0062023, 0.0100208, 0.0060711, 0.0097849, -0.0022950, 0.0026581
7: 0.9823994, 0.9850714, 0.9823076, 0.9849062, -0.0016059, 0.0018600
8: -0.0054349, -0.0025701, -0.0055333, -0.0027471, -0.0017218, 0.0019942
9: -0.0033019, -0.0014095, -0.0031850, -0.0013445, -0.0013173, 0.0011374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009741, upper bound: 0.0009599
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009953, upper bound: 0.0009599
time: 0.71 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018499, -0.0028091, -0.0018135, -0.0007123, 0.0006181
1: -0.0115838, -0.0090050, -0.0114391, -0.0089125, -0.0018075, 0.0015685
2: 0.0278434, 0.0294433, 0.0279331, 0.0295007, -0.0011214, 0.0009731
3: 0.0045844, 0.0075718, 0.0044772, 0.0074043, -0.0018170, 0.0020939
4: -0.0106757, -0.0080526, -0.0105285, -0.0079585, -0.0018385, 0.0015954
5: 0.0096945, 0.0106881, 0.0097503, 0.0107237, -0.0006964, 0.0006043
6: 0.0062063, 0.0099977, 0.0060703, 0.0097851, -0.0023060, 0.0026575
7: 0.9824021, 0.9850552, 0.9823070, 0.9849063, -0.0016136, 0.0018596
8: -0.0054319, -0.0025874, -0.0055339, -0.0027469, -0.0017301, 0.0019937
9: -0.0032905, -0.0014115, -0.0031851, -0.0013441, -0.0013170, 0.0011428

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009822, upper bound: 0.0009942
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010030, upper bound: 0.0009942
time: 0.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0028723, -0.0018488, -0.0028660, -0.0018387, -0.0006442, 0.0006308
1: -0.0115995, -0.0090022, -0.0115834, -0.0089766, -0.0016346, 0.0016008
2: 0.0278337, 0.0294450, 0.0278436, 0.0294609, -0.0010141, 0.0009932
3: 0.0045812, 0.0075900, 0.0045515, 0.0075714, -0.0018545, 0.0018937
4: -0.0106916, -0.0080498, -0.0106753, -0.0080237, -0.0016627, 0.0016283
5: 0.0096885, 0.0106891, 0.0096947, 0.0106990, -0.0006298, 0.0006168
6: 0.0062023, 0.0100208, 0.0061646, 0.0099972, -0.0023536, 0.0024033
7: 0.9823994, 0.9850714, 0.9823729, 0.9850549, -0.0016469, 0.0016817
8: -0.0054349, -0.0025701, -0.0054632, -0.0025878, -0.0017658, 0.0018031
9: -0.0033019, -0.0014095, -0.0032902, -0.0013908, -0.0011910, 0.0011664

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009741, upper bound: 0.0009599
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009953, upper bound: 0.0009599
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0028661, -0.0018499, -0.0028660, -0.0018387, -0.0006422, 0.0006351
1: -0.0115838, -0.0090050, -0.0115835, -0.0089765, -0.0016297, 0.0016117
2: 0.0278434, 0.0294433, 0.0278435, 0.0294610, -0.0010110, 0.0009999
3: 0.0045844, 0.0075718, 0.0045514, 0.0075716, -0.0018671, 0.0018879
4: -0.0106757, -0.0080526, -0.0106754, -0.0080236, -0.0016576, 0.0016394
5: 0.0096945, 0.0106881, 0.0096946, 0.0106990, -0.0006279, 0.0006209
6: 0.0062063, 0.0099977, 0.0061645, 0.0099974, -0.0023696, 0.0023960
7: 0.9824021, 0.9850552, 0.9823729, 0.9850550, -0.0016581, 0.0016766
8: -0.0054319, -0.0025874, -0.0054633, -0.0025877, -0.0017778, 0.0017976
9: -0.0032905, -0.0014115, -0.0032903, -0.0013908, -0.0011874, 0.0011743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 219

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0009822, upper bound: 0.0009939
time: 0.74 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.0010030, upper bound: 0.0009939
time: 0.73 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0028776, -0.0018676, -0.0027883, -0.0017961, -0.0007621, 0.0005991
1: -0.0116128, -0.0090498, -0.0113862, -0.0088685, -0.0019339, 0.0015202
2: 0.0278254, 0.0294155, 0.0279660, 0.0295279, -0.0011998, 0.0009431
3: 0.0046364, 0.0076055, 0.0044264, 0.0073430, -0.0017611, 0.0022403
4: -0.0107052, -0.0080982, -0.0104747, -0.0079138, -0.0019671, 0.0015463
5: 0.0096833, 0.0106708, 0.0097706, 0.0107406, -0.0007451, 0.0005857
6: 0.0062723, 0.0100405, 0.0060057, 0.0097073, -0.0022351, 0.0028433
7: 0.9824483, 0.9850851, 0.9822618, 0.9848520, -0.0015640, 0.0019896
8: -0.0053824, -0.0025553, -0.0055824, -0.0028053, -0.0016768, 0.0021332
9: -0.0033117, -0.0014442, -0.0031465, -0.0013121, -0.0014091, 0.0011076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 81
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 123
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.27 + 596.87 = 600.13 seconds
