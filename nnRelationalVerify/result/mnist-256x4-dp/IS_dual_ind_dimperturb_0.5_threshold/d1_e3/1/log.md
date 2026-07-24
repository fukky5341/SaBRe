## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.01171875
Delta epsilon: 0.00390625
execution index: (1, 3, 1)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000371


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0041064, -0.0040747, -0.0041064, -0.0040747, -0.0000173, 0.0000173)
1: (-0.0064410, -0.0052534, -0.0064410, -0.0052534, -0.0006464, 0.0006464)
2: (0.9687340, 0.9701592, 0.9687340, 0.9701592, -0.0007757, 0.0007757)
3: (0.0156934, 0.0262046, 0.0156934, 0.0262046, -0.0057217, 0.0057217)
4: (-0.0026860, -0.0018866, -0.0026860, -0.0018866, -0.0004352, 0.0004352)
5: (0.0145556, 0.0153636, 0.0145556, 0.0153636, -0.0004398, 0.0004398)
6: (0.0044339, 0.0048269, 0.0044339, 0.0048269, -0.0002139, 0.0002139)
7: (-0.0145694, -0.0118453, -0.0145694, -0.0118453, -0.0014828, 0.0014828)
8: (0.0051705, 0.0073316, 0.0051705, 0.0073316, -0.0011764, 0.0011764)
9: (0.0070242, 0.0109113, 0.0070242, 0.0109113, -0.0021159, 0.0021159)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.31 + 1.65 = 2.95 seconds
status: Status.UNKNOWN
relational distance
Output dim: 2, lower bound: -0.0005306, upper bound: 0.0005306

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 75
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 75

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005130, upper bound: 0.0005117
time: 0.80 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005116, upper bound: 0.0005116
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.86 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 2, lower bound: -0.0005130, upper bound: 0.0005117
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.86
Output dim: 2, lower bound: -0.0005116, upper bound: 0.0005116

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040747, -0.0041063, -0.0040747, -0.0000165, 0.0000170
1: -0.0064178, -0.0052548, -0.0064361, -0.0052537, -0.0006183, 0.0006352
2: 0.9687619, 0.9701574, 0.9687399, 0.9701588, -0.0007420, 0.0007623
3: 0.0158986, 0.0261927, 0.0157365, 0.0262021, -0.0054732, 0.0056226
4: -0.0026851, -0.0019022, -0.0026859, -0.0018899, -0.0004276, 0.0004163
5: 0.0145565, 0.0153478, 0.0145558, 0.0153603, -0.0004322, 0.0004207
6: 0.0044416, 0.0048265, 0.0044355, 0.0048268, -0.0002046, 0.0002102
7: -0.0145663, -0.0118985, -0.0145688, -0.0118565, -0.0014571, 0.0014184
8: 0.0051729, 0.0072894, 0.0051710, 0.0073228, -0.0011560, 0.0011253
9: 0.0070286, 0.0108353, 0.0070251, 0.0108953, -0.0020792, 0.0020240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005012, upper bound: 0.0004847
time: 0.84 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005012, upper bound: 0.0005007
time: 0.83 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040744, -0.0041061, -0.0040747, -0.0000166, 0.0000174
1: -0.0064082, -0.0052412, -0.0064299, -0.0052540, -0.0006198, 0.0006525
2: 0.9687733, 0.9701738, 0.9687474, 0.9701585, -0.0007438, 0.0007831
3: 0.0159832, 0.0263129, 0.0157918, 0.0261999, -0.0054862, 0.0057756
4: -0.0026943, -0.0019086, -0.0026857, -0.0018941, -0.0004393, 0.0004173
5: 0.0145473, 0.0153413, 0.0145560, 0.0153560, -0.0004440, 0.0004217
6: 0.0044448, 0.0048310, 0.0044376, 0.0048268, -0.0002051, 0.0002159
7: -0.0145975, -0.0119204, -0.0145682, -0.0118708, -0.0014968, 0.0014218
8: 0.0051482, 0.0072720, 0.0051714, 0.0073114, -0.0011875, 0.0011280
9: 0.0069841, 0.0108041, 0.0070259, 0.0108748, -0.0021358, 0.0020288

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 83
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 83

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005007, upper bound: 0.0004847
time: 0.82 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0005007, upper bound: 0.0005007
time: 0.79 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.98 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005012, upper bound: 0.0004847
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005012, upper bound: 0.0005007
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005007, upper bound: 0.0004847
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.98
Output dim: 2, lower bound: -0.0005007, upper bound: 0.0005007

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040748, -0.0041053, -0.0040751, -0.0000151, 0.0000157
1: -0.0064064, -0.0052559, -0.0063983, -0.0052683, -0.0005653, 0.0005886
2: 0.9687755, 0.9701563, 0.9687853, 0.9701413, -0.0006784, 0.0007064
3: 0.0159995, 0.0261831, 0.0160712, 0.0260731, -0.0050037, 0.0052102
4: -0.0026844, -0.0019099, -0.0026760, -0.0019153, -0.0003963, 0.0003806
5: 0.0145573, 0.0153401, 0.0145657, 0.0153346, -0.0004005, 0.0003846
6: 0.0044454, 0.0048261, 0.0044481, 0.0048220, -0.0001871, 0.0001948
7: -0.0145638, -0.0119247, -0.0145353, -0.0119433, -0.0013503, 0.0012968
8: 0.0051749, 0.0072687, 0.0051975, 0.0072539, -0.0010712, 0.0010288
9: 0.0070321, 0.0107980, 0.0070728, 0.0107715, -0.0019267, 0.0018504

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004847
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004847
time: 1.05 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040748, -0.0041058, -0.0040748, -0.0000164, 0.0000152
1: -0.0064125, -0.0052554, -0.0064171, -0.0052561, -0.0006124, 0.0005689
2: 0.9687682, 0.9701567, 0.9687626, 0.9701559, -0.0007349, 0.0006827
3: 0.0159453, 0.0261872, 0.0159043, 0.0261808, -0.0054203, 0.0050355
4: -0.0026847, -0.0019058, -0.0026842, -0.0019026, -0.0003830, 0.0004122
5: 0.0145570, 0.0153442, 0.0145575, 0.0153474, -0.0003871, 0.0004166
6: 0.0044434, 0.0048263, 0.0044418, 0.0048260, -0.0002027, 0.0001883
7: -0.0145649, -0.0119106, -0.0145632, -0.0119000, -0.0013050, 0.0014047
8: 0.0051741, 0.0072798, 0.0051754, 0.0072883, -0.0010353, 0.0011144
9: 0.0070307, 0.0108181, 0.0070330, 0.0108332, -0.0018621, 0.0020044

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005007
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005006
time: 1.05 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040744, -0.0041051, -0.0040751, -0.0000151, 0.0000162
1: -0.0063962, -0.0052423, -0.0063918, -0.0052685, -0.0005668, 0.0006065
2: 0.9687878, 0.9701725, 0.9687930, 0.9701410, -0.0006802, 0.0007279
3: 0.0160896, 0.0263034, 0.0161284, 0.0260710, -0.0050168, 0.0053685
4: -0.0026936, -0.0019167, -0.0026759, -0.0019197, -0.0004083, 0.0003816
5: 0.0145480, 0.0153331, 0.0145659, 0.0153302, -0.0004127, 0.0003856
6: 0.0044488, 0.0048306, 0.0044502, 0.0048219, -0.0001876, 0.0002007
7: -0.0145950, -0.0119480, -0.0145348, -0.0119581, -0.0013913, 0.0013001
8: 0.0051502, 0.0072501, 0.0051979, 0.0072422, -0.0011038, 0.0010315
9: 0.0069877, 0.0107647, 0.0070736, 0.0107504, -0.0019853, 0.0018552

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004847
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004846
time: 1.09 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041054, -0.0040744, -0.0041056, -0.0040748, -0.0000164, 0.0000158
1: -0.0064033, -0.0052418, -0.0064115, -0.0052564, -0.0006139, 0.0005920
2: 0.9687792, 0.9701731, 0.9687694, 0.9701557, -0.0007367, 0.0007104
3: 0.0160268, 0.0263076, 0.0159546, 0.0261788, -0.0054340, 0.0052401
4: -0.0026939, -0.0019120, -0.0026841, -0.0019065, -0.0003985, 0.0004133
5: 0.0145477, 0.0153380, 0.0145576, 0.0153435, -0.0004028, 0.0004177
6: 0.0044464, 0.0048308, 0.0044437, 0.0048260, -0.0002032, 0.0001959
7: -0.0145961, -0.0119317, -0.0145627, -0.0119130, -0.0013580, 0.0014083
8: 0.0051493, 0.0072631, 0.0051758, 0.0072779, -0.0010774, 0.0011173
9: 0.0069861, 0.0107879, 0.0070338, 0.0108146, -0.0019378, 0.0020095

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 83
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 83

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005007
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005007
time: 1.02 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.12 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004847
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004847
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005007
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005006
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004847
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004846
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005007
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.12
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005007

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040751, -0.0041053, -0.0040751, -0.0000143, 0.0000147
1: -0.0063800, -0.0052691, -0.0063983, -0.0052683, -0.0005354, 0.0005518
2: 0.9688072, 0.9701403, 0.9687853, 0.9701413, -0.0006425, 0.0006622
3: 0.0162334, 0.0260656, 0.0160712, 0.0260731, -0.0047386, 0.0048841
4: -0.0026755, -0.0019277, -0.0026760, -0.0019153, -0.0003715, 0.0003604
5: 0.0145663, 0.0153221, 0.0145657, 0.0153346, -0.0003754, 0.0003642
6: 0.0044541, 0.0048217, 0.0044481, 0.0048220, -0.0001772, 0.0001826
7: -0.0145334, -0.0119853, -0.0145353, -0.0119433, -0.0012658, 0.0012281
8: 0.0051990, 0.0072206, 0.0051975, 0.0072539, -0.0010042, 0.0009743
9: 0.0070756, 0.0107115, 0.0070728, 0.0107715, -0.0018062, 0.0017523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004842
time: 0.77 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004847
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040748, -0.0041053, -0.0040751, -0.0000152, 0.0000157
1: -0.0063985, -0.0052571, -0.0063983, -0.0052683, -0.0005674, 0.0005875
2: 0.9687851, 0.9701548, 0.9687853, 0.9701413, -0.0006808, 0.0007051
3: 0.0160696, 0.0261724, 0.0160712, 0.0260731, -0.0050218, 0.0052005
4: -0.0026836, -0.0019152, -0.0026760, -0.0019153, -0.0003955, 0.0003819
5: 0.0145581, 0.0153347, 0.0145657, 0.0153346, -0.0003998, 0.0003860
6: 0.0044480, 0.0048257, 0.0044481, 0.0048220, -0.0001878, 0.0001944
7: -0.0145611, -0.0119428, -0.0145353, -0.0119433, -0.0013478, 0.0013014
8: 0.0051771, 0.0072543, 0.0051975, 0.0072539, -0.0010692, 0.0010325
9: 0.0070361, 0.0107721, 0.0070728, 0.0107715, -0.0019231, 0.0018571

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004842
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004847
time: 1.05 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040751, -0.0041058, -0.0040748, -0.0000152, 0.0000156
1: -0.0063800, -0.0052691, -0.0064171, -0.0052561, -0.0005711, 0.0005838
2: 0.9688072, 0.9701403, 0.9687626, 0.9701559, -0.0006853, 0.0007005
3: 0.0162334, 0.0260656, 0.0159043, 0.0261808, -0.0050548, 0.0051671
4: -0.0026755, -0.0019277, -0.0026842, -0.0019026, -0.0003930, 0.0003844
5: 0.0145663, 0.0153221, 0.0145575, 0.0153474, -0.0003972, 0.0003886
6: 0.0044541, 0.0048217, 0.0044418, 0.0048260, -0.0001890, 0.0001932
7: -0.0145334, -0.0119853, -0.0145632, -0.0119000, -0.0013391, 0.0013100
8: 0.0051990, 0.0072206, 0.0051754, 0.0072883, -0.0010624, 0.0010393
9: 0.0070756, 0.0107115, 0.0070330, 0.0108332, -0.0019108, 0.0018693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005002
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005007
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040748, -0.0041058, -0.0040748, -0.0000147, 0.0000152
1: -0.0063985, -0.0052571, -0.0064171, -0.0052561, -0.0005511, 0.0005674
2: 0.9687851, 0.9701548, 0.9687626, 0.9701559, -0.0006614, 0.0006809
3: 0.0160696, 0.0261724, 0.0159043, 0.0261808, -0.0048782, 0.0050219
4: -0.0026836, -0.0019152, -0.0026842, -0.0019026, -0.0003819, 0.0003710
5: 0.0145581, 0.0153347, 0.0145575, 0.0153474, -0.0003860, 0.0003750
6: 0.0044480, 0.0048257, 0.0044418, 0.0048260, -0.0001824, 0.0001878
7: -0.0145611, -0.0119428, -0.0145632, -0.0119000, -0.0013015, 0.0012642
8: 0.0051771, 0.0072543, 0.0051754, 0.0072883, -0.0010325, 0.0010030
9: 0.0070361, 0.0107721, 0.0070330, 0.0108332, -0.0018571, 0.0018040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004842
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004847
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040748, -0.0041051, -0.0040751, -0.0000143, 0.0000153
1: -0.0063692, -0.0052569, -0.0063918, -0.0052685, -0.0005367, 0.0005746
2: 0.9688202, 0.9701550, 0.9687930, 0.9701410, -0.0006441, 0.0006896
3: 0.0163289, 0.0261742, 0.0161284, 0.0260710, -0.0047505, 0.0050864
4: -0.0026837, -0.0019349, -0.0026759, -0.0019197, -0.0003868, 0.0003613
5: 0.0145580, 0.0153147, 0.0145659, 0.0153302, -0.0003910, 0.0003652
6: 0.0044577, 0.0048258, 0.0044502, 0.0048219, -0.0001776, 0.0001902
7: -0.0145615, -0.0120100, -0.0145348, -0.0119581, -0.0013182, 0.0012311
8: 0.0051767, 0.0072010, 0.0051979, 0.0072422, -0.0010458, 0.0009767
9: 0.0070354, 0.0106762, 0.0070736, 0.0107504, -0.0018809, 0.0017567

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040744, -0.0041051, -0.0040751, -0.0000152, 0.0000162
1: -0.0063900, -0.0052434, -0.0063918, -0.0052685, -0.0005687, 0.0006054
2: 0.9687951, 0.9701712, 0.9687930, 0.9701410, -0.0006825, 0.0007265
3: 0.0161442, 0.0262935, 0.0161284, 0.0260710, -0.0050337, 0.0053582
4: -0.0026928, -0.0019209, -0.0026759, -0.0019197, -0.0004075, 0.0003828
5: 0.0145488, 0.0153289, 0.0145659, 0.0153302, -0.0004119, 0.0003869
6: 0.0044508, 0.0048303, 0.0044502, 0.0048219, -0.0001882, 0.0002003
7: -0.0145924, -0.0119622, -0.0145348, -0.0119581, -0.0013886, 0.0013045
8: 0.0051522, 0.0072389, 0.0051979, 0.0072422, -0.0011017, 0.0010349
9: 0.0069913, 0.0107445, 0.0070736, 0.0107504, -0.0019814, 0.0018615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040748, -0.0041056, -0.0040748, -0.0000153, 0.0000162
1: -0.0063692, -0.0052569, -0.0064115, -0.0052564, -0.0005724, 0.0006057
2: 0.9688202, 0.9701550, 0.9687694, 0.9701557, -0.0006869, 0.0007269
3: 0.0163289, 0.0261742, 0.0159546, 0.0261788, -0.0050668, 0.0053616
4: -0.0026837, -0.0019349, -0.0026841, -0.0019065, -0.0004078, 0.0003854
5: 0.0145580, 0.0153147, 0.0145576, 0.0153435, -0.0004121, 0.0003895
6: 0.0044577, 0.0048258, 0.0044437, 0.0048260, -0.0001894, 0.0002005
7: -0.0145615, -0.0120100, -0.0145627, -0.0119130, -0.0013895, 0.0013131
8: 0.0051767, 0.0072010, 0.0051758, 0.0072779, -0.0011024, 0.0010417
9: 0.0070354, 0.0106762, 0.0070338, 0.0108146, -0.0019827, 0.0018737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005003
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005003
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040744, -0.0041056, -0.0040748, -0.0000148, 0.0000158
1: -0.0063900, -0.0052434, -0.0064115, -0.0052564, -0.0005531, 0.0005902
2: 0.9687951, 0.9701712, 0.9687694, 0.9701557, -0.0006637, 0.0007082
3: 0.0161442, 0.0262935, 0.0159546, 0.0261788, -0.0048953, 0.0052237
4: -0.0026928, -0.0019209, -0.0026841, -0.0019065, -0.0003973, 0.0003723
5: 0.0145488, 0.0153289, 0.0145576, 0.0153435, -0.0004015, 0.0003763
6: 0.0044508, 0.0048303, 0.0044437, 0.0048260, -0.0001830, 0.0001953
7: -0.0145924, -0.0119622, -0.0145627, -0.0119130, -0.0013538, 0.0012687
8: 0.0051522, 0.0072389, 0.0051758, 0.0072779, -0.0010740, 0.0010065
9: 0.0069913, 0.0107445, 0.0070338, 0.0108146, -0.0019317, 0.0018103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 75
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 75

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004842
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004842
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.38 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004842
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004847
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004842
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004922, upper bound: 0.0004847
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005002
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0005007
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004842
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004851, upper bound: 0.0004847
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004907, upper bound: 0.0004842
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005003
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0005003
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004842
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.38
Output dim: 2, lower bound: -0.0004847, upper bound: 0.0004842

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040751, -0.0041048, -0.0040751, -0.0000142, 0.0000142
1: -0.0063800, -0.0052691, -0.0063800, -0.0052691, -0.0005312, 0.0005312
2: 0.9688072, 0.9701403, 0.9688072, 0.9701403, -0.0006375, 0.0006375
3: 0.0162334, 0.0260656, 0.0162334, 0.0260656, -0.0047023, 0.0047023
4: -0.0026755, -0.0019277, -0.0026755, -0.0019277, -0.0003576, 0.0003576
5: 0.0145663, 0.0153221, 0.0145663, 0.0153221, -0.0003615, 0.0003615
6: 0.0044541, 0.0048217, 0.0044541, 0.0048217, -0.0001758, 0.0001758
7: -0.0145334, -0.0119853, -0.0145334, -0.0119853, -0.0012186, 0.0012186
8: 0.0051990, 0.0072206, 0.0051990, 0.0072206, -0.0009668, 0.0009668
9: 0.0070756, 0.0107115, 0.0070756, 0.0107115, -0.0017389, 0.0017389

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004736
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004784, upper bound: 0.0004764
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040751, -0.0041045, -0.0040748, -0.0000149, 0.0000142
1: -0.0063800, -0.0052691, -0.0063692, -0.0052569, -0.0005572, 0.0005317
2: 0.9688072, 0.9701403, 0.9688202, 0.9701550, -0.0006687, 0.0006380
3: 0.0162334, 0.0260656, 0.0163289, 0.0261742, -0.0049320, 0.0047061
4: -0.0026755, -0.0019277, -0.0026837, -0.0019349, -0.0003579, 0.0003751
5: 0.0145663, 0.0153221, 0.0145580, 0.0153147, -0.0003617, 0.0003791
6: 0.0044541, 0.0048217, 0.0044577, 0.0048258, -0.0001844, 0.0001760
7: -0.0145334, -0.0119853, -0.0145615, -0.0120100, -0.0012196, 0.0012782
8: 0.0051990, 0.0072206, 0.0051767, 0.0072010, -0.0009676, 0.0010140
9: 0.0070756, 0.0107115, 0.0070354, 0.0106762, -0.0017403, 0.0018239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.16 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004736
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004784, upper bound: 0.0004764
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040748, -0.0041048, -0.0040751, -0.0000150, 0.0000151
1: -0.0063985, -0.0052571, -0.0063800, -0.0052691, -0.0005632, 0.0005670
2: 0.9687851, 0.9701548, 0.9688072, 0.9701403, -0.0006759, 0.0006804
3: 0.0160696, 0.0261724, 0.0162334, 0.0260656, -0.0049855, 0.0050186
4: -0.0026836, -0.0019152, -0.0026755, -0.0019277, -0.0003817, 0.0003792
5: 0.0145581, 0.0153347, 0.0145663, 0.0153221, -0.0003858, 0.0003832
6: 0.0044480, 0.0048257, 0.0044541, 0.0048217, -0.0001864, 0.0001876
7: -0.0145611, -0.0119428, -0.0145334, -0.0119853, -0.0013006, 0.0012920
8: 0.0051771, 0.0072543, 0.0051990, 0.0072206, -0.0010319, 0.0010250
9: 0.0070361, 0.0107721, 0.0070756, 0.0107115, -0.0018559, 0.0018436

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004674
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004869, upper bound: 0.0004705
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040748, -0.0041045, -0.0040748, -0.0000157, 0.0000152
1: -0.0063985, -0.0052571, -0.0063692, -0.0052569, -0.0005892, 0.0005674
2: 0.9687851, 0.9701548, 0.9688202, 0.9701550, -0.0007071, 0.0006809
3: 0.0160696, 0.0261724, 0.0163289, 0.0261742, -0.0052152, 0.0050224
4: -0.0026836, -0.0019152, -0.0026837, -0.0019349, -0.0003820, 0.0003966
5: 0.0145581, 0.0153347, 0.0145580, 0.0153147, -0.0003861, 0.0004009
6: 0.0044480, 0.0048257, 0.0044577, 0.0048258, -0.0001950, 0.0001878
7: -0.0145611, -0.0119428, -0.0145615, -0.0120100, -0.0013016, 0.0013516
8: 0.0051771, 0.0072543, 0.0051767, 0.0072010, -0.0010326, 0.0010723
9: 0.0070361, 0.0107721, 0.0070354, 0.0106762, -0.0018573, 0.0019286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004677
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004869, upper bound: 0.0004708
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040751, -0.0041053, -0.0040748, -0.0000151, 0.0000150
1: -0.0063800, -0.0052691, -0.0063985, -0.0052571, -0.0005670, 0.0005632
2: 0.9688072, 0.9701403, 0.9687851, 0.9701548, -0.0006804, 0.0006759
3: 0.0162334, 0.0260656, 0.0160696, 0.0261724, -0.0050186, 0.0049855
4: -0.0026755, -0.0019277, -0.0026836, -0.0019152, -0.0003792, 0.0003817
5: 0.0145663, 0.0153221, 0.0145581, 0.0153347, -0.0003832, 0.0003858
6: 0.0044541, 0.0048217, 0.0044480, 0.0048257, -0.0001876, 0.0001864
7: -0.0145334, -0.0119853, -0.0145611, -0.0119428, -0.0012920, 0.0013006
8: 0.0051990, 0.0072206, 0.0051771, 0.0072543, -0.0010250, 0.0010319
9: 0.0070756, 0.0107115, 0.0070361, 0.0107721, -0.0018436, 0.0018559

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004835
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004718, upper bound: 0.0004858
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040751, -0.0041050, -0.0040744, -0.0000157, 0.0000150
1: -0.0063800, -0.0052691, -0.0063900, -0.0052434, -0.0005879, 0.0005617
2: 0.9688072, 0.9701403, 0.9687951, 0.9701712, -0.0007055, 0.0006741
3: 0.0162334, 0.0260656, 0.0161442, 0.0262935, -0.0052038, 0.0049718
4: -0.0026755, -0.0019277, -0.0026928, -0.0019209, -0.0003781, 0.0003958
5: 0.0145663, 0.0153221, 0.0145488, 0.0153289, -0.0003822, 0.0004000
6: 0.0044541, 0.0048217, 0.0044508, 0.0048303, -0.0001946, 0.0001859
7: -0.0145334, -0.0119853, -0.0145924, -0.0119622, -0.0012885, 0.0013486
8: 0.0051990, 0.0072206, 0.0051522, 0.0072389, -0.0010222, 0.0010699
9: 0.0070756, 0.0107115, 0.0069913, 0.0107445, -0.0018386, 0.0019244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004838
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004718, upper bound: 0.0004861
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040748, -0.0041053, -0.0040748, -0.0000146, 0.0000146
1: -0.0063985, -0.0052571, -0.0063985, -0.0052571, -0.0005470, 0.0005470
2: 0.9687851, 0.9701548, 0.9687851, 0.9701548, -0.0006564, 0.0006564
3: 0.0160696, 0.0261724, 0.0160696, 0.0261724, -0.0048412, 0.0048412
4: -0.0026836, -0.0019152, -0.0026836, -0.0019152, -0.0003682, 0.0003682
5: 0.0145581, 0.0153347, 0.0145581, 0.0153347, -0.0003721, 0.0003721
6: 0.0044480, 0.0048257, 0.0044480, 0.0048257, -0.0001810, 0.0001810
7: -0.0145611, -0.0119428, -0.0145611, -0.0119428, -0.0012547, 0.0012547
8: 0.0051771, 0.0072543, 0.0051771, 0.0072543, -0.0009954, 0.0009954
9: 0.0070361, 0.0107721, 0.0070361, 0.0107721, -0.0017903, 0.0017903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004675
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004870, upper bound: 0.0004705
time: 0.78 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041053, -0.0040748, -0.0041050, -0.0040744, -0.0000153, 0.0000146
1: -0.0063985, -0.0052571, -0.0063900, -0.0052434, -0.0005730, 0.0005475
2: 0.9687851, 0.9701548, 0.9687951, 0.9701712, -0.0006876, 0.0006570
3: 0.0160696, 0.0261724, 0.0161442, 0.0262935, -0.0050719, 0.0048457
4: -0.0026836, -0.0019152, -0.0026928, -0.0019209, -0.0003685, 0.0003857
5: 0.0145581, 0.0153347, 0.0145488, 0.0153289, -0.0003725, 0.0003899
6: 0.0044480, 0.0048257, 0.0044508, 0.0048303, -0.0001896, 0.0001812
7: -0.0145611, -0.0119428, -0.0145924, -0.0119622, -0.0012558, 0.0013144
8: 0.0051771, 0.0072543, 0.0051522, 0.0072389, -0.0009963, 0.0010428
9: 0.0070361, 0.0107721, 0.0069913, 0.0107445, -0.0017920, 0.0018756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004678
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004870, upper bound: 0.0004709
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040748, -0.0041048, -0.0040751, -0.0000142, 0.0000149
1: -0.0063692, -0.0052569, -0.0063800, -0.0052691, -0.0005317, 0.0005572
2: 0.9688202, 0.9701550, 0.9688072, 0.9701403, -0.0006380, 0.0006687
3: 0.0163289, 0.0261742, 0.0162334, 0.0260656, -0.0047061, 0.0049320
4: -0.0026837, -0.0019349, -0.0026755, -0.0019277, -0.0003751, 0.0003579
5: 0.0145580, 0.0153147, 0.0145663, 0.0153221, -0.0003791, 0.0003617
6: 0.0044577, 0.0048258, 0.0044541, 0.0048217, -0.0001760, 0.0001844
7: -0.0145615, -0.0120100, -0.0145334, -0.0119853, -0.0012782, 0.0012196
8: 0.0051767, 0.0072010, 0.0051990, 0.0072206, -0.0010140, 0.0009676
9: 0.0070354, 0.0106762, 0.0070756, 0.0107115, -0.0018239, 0.0017403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004736
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004764, upper bound: 0.0004764
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040748, -0.0041045, -0.0040748, -0.0000143, 0.0000143
1: -0.0063692, -0.0052569, -0.0063692, -0.0052569, -0.0005344, 0.0005344
2: 0.9688202, 0.9701550, 0.9688202, 0.9701550, -0.0006413, 0.0006413
3: 0.0163289, 0.0261742, 0.0163289, 0.0261742, -0.0047305, 0.0047305
4: -0.0026837, -0.0019349, -0.0026837, -0.0019349, -0.0003598, 0.0003598
5: 0.0145580, 0.0153147, 0.0145580, 0.0153147, -0.0003636, 0.0003636
6: 0.0044577, 0.0048258, 0.0044577, 0.0048258, -0.0001769, 0.0001769
7: -0.0145615, -0.0120100, -0.0145615, -0.0120100, -0.0012259, 0.0012259
8: 0.0051767, 0.0072010, 0.0051767, 0.0072010, -0.0009726, 0.0009726
9: 0.0070354, 0.0106762, 0.0070354, 0.0106762, -0.0017493, 0.0017493

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004736
time: 0.78 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004764, upper bound: 0.0004764
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040744, -0.0041048, -0.0040751, -0.0000150, 0.0000157
1: -0.0063900, -0.0052434, -0.0063800, -0.0052691, -0.0005617, 0.0005879
2: 0.9687951, 0.9701712, 0.9688072, 0.9701403, -0.0006741, 0.0007055
3: 0.0161442, 0.0262935, 0.0162334, 0.0260656, -0.0049718, 0.0052038
4: -0.0026928, -0.0019209, -0.0026755, -0.0019277, -0.0003958, 0.0003781
5: 0.0145488, 0.0153289, 0.0145663, 0.0153221, -0.0004000, 0.0003822
6: 0.0044508, 0.0048303, 0.0044541, 0.0048217, -0.0001859, 0.0001946
7: -0.0145924, -0.0119622, -0.0145334, -0.0119853, -0.0013486, 0.0012885
8: 0.0051522, 0.0072389, 0.0051990, 0.0072206, -0.0010699, 0.0010222
9: 0.0069913, 0.0107445, 0.0070756, 0.0107115, -0.0019244, 0.0018386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004674
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004705
time: 0.84 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040744, -0.0041045, -0.0040748, -0.0000151, 0.0000152
1: -0.0063900, -0.0052434, -0.0063692, -0.0052569, -0.0005664, 0.0005702
2: 0.9687951, 0.9701712, 0.9688202, 0.9701550, -0.0006797, 0.0006842
3: 0.0161442, 0.0262935, 0.0163289, 0.0261742, -0.0050137, 0.0050467
4: -0.0026928, -0.0019209, -0.0026837, -0.0019349, -0.0003838, 0.0003813
5: 0.0145488, 0.0153289, 0.0145580, 0.0153147, -0.0003879, 0.0003854
6: 0.0044508, 0.0048303, 0.0044577, 0.0048258, -0.0001875, 0.0001887
7: -0.0145924, -0.0119622, -0.0145615, -0.0120100, -0.0013079, 0.0012993
8: 0.0051522, 0.0072389, 0.0051767, 0.0072010, -0.0010376, 0.0010308
9: 0.0069913, 0.0107445, 0.0070354, 0.0106762, -0.0018663, 0.0018541

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004674
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004705
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040748, -0.0041053, -0.0040748, -0.0000152, 0.0000157
1: -0.0063692, -0.0052569, -0.0063985, -0.0052571, -0.0005674, 0.0005892
2: 0.9688202, 0.9701550, 0.9687851, 0.9701548, -0.0006809, 0.0007071
3: 0.0163289, 0.0261742, 0.0160696, 0.0261724, -0.0050224, 0.0052152
4: -0.0026837, -0.0019349, -0.0026836, -0.0019152, -0.0003966, 0.0003820
5: 0.0145580, 0.0153147, 0.0145581, 0.0153347, -0.0004009, 0.0003861
6: 0.0044577, 0.0048258, 0.0044480, 0.0048257, -0.0001878, 0.0001950
7: -0.0145615, -0.0120100, -0.0145611, -0.0119428, -0.0013516, 0.0013016
8: 0.0051767, 0.0072010, 0.0051771, 0.0072543, -0.0010723, 0.0010326
9: 0.0070354, 0.0106762, 0.0070361, 0.0107721, -0.0019286, 0.0018573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004835
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004708, upper bound: 0.0004858
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040748, -0.0041050, -0.0040744, -0.0000152, 0.0000151
1: -0.0063692, -0.0052569, -0.0063900, -0.0052434, -0.0005702, 0.0005664
2: 0.9688202, 0.9701550, 0.9687951, 0.9701712, -0.0006842, 0.0006797
3: 0.0163289, 0.0261742, 0.0161442, 0.0262935, -0.0050467, 0.0050137
4: -0.0026837, -0.0019349, -0.0026928, -0.0019209, -0.0003813, 0.0003838
5: 0.0145580, 0.0153147, 0.0145488, 0.0153289, -0.0003854, 0.0003879
6: 0.0044577, 0.0048258, 0.0044508, 0.0048303, -0.0001887, 0.0001875
7: -0.0145615, -0.0120100, -0.0145924, -0.0119622, -0.0012993, 0.0013079
8: 0.0051767, 0.0072010, 0.0051522, 0.0072389, -0.0010308, 0.0010376
9: 0.0070354, 0.0106762, 0.0069913, 0.0107445, -0.0018541, 0.0018663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004835
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004708, upper bound: 0.0004858
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040744, -0.0041053, -0.0040748, -0.0000146, 0.0000153
1: -0.0063900, -0.0052434, -0.0063985, -0.0052571, -0.0005475, 0.0005730
2: 0.9687951, 0.9701712, 0.9687851, 0.9701548, -0.0006570, 0.0006876
3: 0.0161442, 0.0262935, 0.0160696, 0.0261724, -0.0048457, 0.0050719
4: -0.0026928, -0.0019209, -0.0026836, -0.0019152, -0.0003857, 0.0003685
5: 0.0145488, 0.0153289, 0.0145581, 0.0153347, -0.0003899, 0.0003725
6: 0.0044508, 0.0048303, 0.0044480, 0.0048257, -0.0001812, 0.0001896
7: -0.0145924, -0.0119622, -0.0145611, -0.0119428, -0.0013144, 0.0012558
8: 0.0051522, 0.0072389, 0.0051771, 0.0072543, -0.0010428, 0.0009963
9: 0.0069913, 0.0107445, 0.0070361, 0.0107721, -0.0018756, 0.0017920

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004675
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004706
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040744, -0.0041050, -0.0040744, -0.0000147, 0.0000147
1: -0.0063900, -0.0052434, -0.0063900, -0.0052434, -0.0005508, 0.0005508
2: 0.9687951, 0.9701712, 0.9687951, 0.9701712, -0.0006609, 0.0006609
3: 0.0161442, 0.0262935, 0.0161442, 0.0262935, -0.0048749, 0.0048749
4: -0.0026928, -0.0019209, -0.0026928, -0.0019209, -0.0003708, 0.0003708
5: 0.0145488, 0.0153289, 0.0145488, 0.0153289, -0.0003747, 0.0003747
6: 0.0044508, 0.0048303, 0.0044508, 0.0048303, -0.0001823, 0.0001823
7: -0.0145924, -0.0119622, -0.0145924, -0.0119622, -0.0012634, 0.0012634
8: 0.0051522, 0.0072389, 0.0051522, 0.0072389, -0.0010023, 0.0010023
9: 0.0069913, 0.0107445, 0.0069913, 0.0107445, -0.0018027, 0.0018027

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004674
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004706
time: 0.80 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 2.93 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004736
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004784, upper bound: 0.0004764
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004736
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004784, upper bound: 0.0004764
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004674
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004869, upper bound: 0.0004705
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004677
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004869, upper bound: 0.0004708
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004835
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004718, upper bound: 0.0004858
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004838
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004718, upper bound: 0.0004861
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004675
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004870, upper bound: 0.0004705
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004678
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004870, upper bound: 0.0004709
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004736
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004764, upper bound: 0.0004764
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004736
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004764, upper bound: 0.0004764
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004674
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004705
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004674
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004705
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004835
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004708, upper bound: 0.0004858
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004835
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004708, upper bound: 0.0004858
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004675
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004706
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004674
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 2, lower bound: -0.0004861, upper bound: 0.0004706

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041048, -0.0040753, -0.0000139, 0.0000132
1: -0.0063975, -0.0052939, -0.0063789, -0.0052772, -0.0005216, 0.0004955
2: 0.9687861, 0.9701107, 0.9688085, 0.9701306, -0.0006259, 0.0005946
3: 0.0160782, 0.0258468, 0.0162427, 0.0259945, -0.0046169, 0.0043859
4: -0.0026588, -0.0019159, -0.0026701, -0.0019284, -0.0003336, 0.0003511
5: 0.0145831, 0.0153340, 0.0145718, 0.0153214, -0.0003371, 0.0003549
6: 0.0044483, 0.0048136, 0.0044545, 0.0048191, -0.0001726, 0.0001640
7: -0.0144767, -0.0119451, -0.0145149, -0.0119877, -0.0011367, 0.0011965
8: 0.0052440, 0.0072525, 0.0052137, 0.0072187, -0.0009018, 0.0009492
9: 0.0071565, 0.0107689, 0.0071019, 0.0107081, -0.0016219, 0.0017073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004746
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004754
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041048, -0.0040751, -0.0000141, 0.0000134
1: -0.0063789, -0.0052813, -0.0063799, -0.0052694, -0.0005283, 0.0005018
2: 0.9688084, 0.9701257, 0.9688073, 0.9701399, -0.0006340, 0.0006022
3: 0.0162426, 0.0259579, 0.0162336, 0.0260633, -0.0046765, 0.0044416
4: -0.0026673, -0.0019284, -0.0026753, -0.0019277, -0.0003378, 0.0003557
5: 0.0145746, 0.0153214, 0.0145665, 0.0153221, -0.0003414, 0.0003595
6: 0.0044545, 0.0048177, 0.0044541, 0.0048217, -0.0001748, 0.0001661
7: -0.0145055, -0.0119877, -0.0145328, -0.0119853, -0.0011511, 0.0012120
8: 0.0052212, 0.0072187, 0.0051995, 0.0072205, -0.0009132, 0.0009615
9: 0.0071154, 0.0107081, 0.0070764, 0.0107115, -0.0016425, 0.0017294

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004754, upper bound: 0.0004747
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004754, upper bound: 0.0004786
time: 0.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041045, -0.0040750, -0.0000147, 0.0000132
1: -0.0063975, -0.0052939, -0.0063682, -0.0052646, -0.0005503, 0.0004958
2: 0.9687861, 0.9701107, 0.9688213, 0.9701458, -0.0006604, 0.0005950
3: 0.0160782, 0.0258468, 0.0163374, 0.0261056, -0.0048708, 0.0043885
4: -0.0026588, -0.0019159, -0.0026785, -0.0019356, -0.0003338, 0.0003705
5: 0.0145831, 0.0153340, 0.0145632, 0.0153141, -0.0003373, 0.0003744
6: 0.0044483, 0.0048136, 0.0044580, 0.0048232, -0.0001821, 0.0001641
7: -0.0144767, -0.0119451, -0.0145438, -0.0120122, -0.0011373, 0.0012623
8: 0.0052440, 0.0072525, 0.0051908, 0.0071992, -0.0009023, 0.0010015
9: 0.0071565, 0.0107689, 0.0070608, 0.0106731, -0.0016229, 0.0018012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004735, upper bound: 0.0004714
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004735, upper bound: 0.0004725
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041045, -0.0040748, -0.0000148, 0.0000136
1: -0.0063789, -0.0052813, -0.0063691, -0.0052571, -0.0005542, 0.0005100
2: 0.9688084, 0.9701257, 0.9688202, 0.9701546, -0.0006651, 0.0006120
3: 0.0162426, 0.0259579, 0.0163291, 0.0261719, -0.0049056, 0.0045140
4: -0.0026673, -0.0019284, -0.0026836, -0.0019350, -0.0003433, 0.0003731
5: 0.0145746, 0.0153214, 0.0145581, 0.0153147, -0.0003470, 0.0003771
6: 0.0044545, 0.0048177, 0.0044577, 0.0048257, -0.0001834, 0.0001688
7: -0.0145055, -0.0119877, -0.0145609, -0.0120101, -0.0011698, 0.0012713
8: 0.0052212, 0.0072187, 0.0051772, 0.0072009, -0.0009281, 0.0010086
9: 0.0071154, 0.0107081, 0.0070363, 0.0106762, -0.0016693, 0.0018141

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004744, upper bound: 0.0004714
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004744, upper bound: 0.0004764
time: 0.91 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041048, -0.0040753, -0.0000148, 0.0000142
1: -0.0064171, -0.0052819, -0.0063789, -0.0052772, -0.0005558, 0.0005331
2: 0.9687627, 0.9701250, 0.9688085, 0.9701306, -0.0006670, 0.0006398
3: 0.0159048, 0.0259526, 0.0162427, 0.0259945, -0.0049199, 0.0047191
4: -0.0026669, -0.0019027, -0.0026701, -0.0019284, -0.0003589, 0.0003742
5: 0.0145750, 0.0153473, 0.0145718, 0.0153214, -0.0003627, 0.0003782
6: 0.0044418, 0.0048175, 0.0044545, 0.0048191, -0.0001839, 0.0001764
7: -0.0145041, -0.0119001, -0.0145149, -0.0119877, -0.0012230, 0.0012750
8: 0.0052223, 0.0072882, 0.0052137, 0.0072187, -0.0009703, 0.0010116
9: 0.0071174, 0.0108331, 0.0071019, 0.0107081, -0.0017451, 0.0018194

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004688
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004689
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041048, -0.0040751, -0.0000150, 0.0000144
1: -0.0063974, -0.0052693, -0.0063799, -0.0052694, -0.0005603, 0.0005399
2: 0.9687864, 0.9701401, 0.9688073, 0.9701399, -0.0006724, 0.0006479
3: 0.0160793, 0.0260644, 0.0162336, 0.0260633, -0.0049598, 0.0047785
4: -0.0026754, -0.0019159, -0.0026753, -0.0019277, -0.0003634, 0.0003772
5: 0.0145664, 0.0153339, 0.0145665, 0.0153221, -0.0003673, 0.0003813
6: 0.0044484, 0.0048217, 0.0044541, 0.0048217, -0.0001854, 0.0001787
7: -0.0145331, -0.0119453, -0.0145328, -0.0119853, -0.0012384, 0.0012854
8: 0.0051993, 0.0072523, 0.0051995, 0.0072205, -0.0009825, 0.0010198
9: 0.0070760, 0.0107685, 0.0070764, 0.0107115, -0.0017671, 0.0018341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004688
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004688
time: 0.99 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041045, -0.0040750, -0.0000156, 0.0000142
1: -0.0064171, -0.0052819, -0.0063682, -0.0052646, -0.0005845, 0.0005334
2: 0.9687627, 0.9701250, 0.9688213, 0.9701458, -0.0007015, 0.0006401
3: 0.0159048, 0.0259526, 0.0163374, 0.0261056, -0.0051739, 0.0047216
4: -0.0026669, -0.0019027, -0.0026785, -0.0019356, -0.0003591, 0.0003935
5: 0.0145750, 0.0153473, 0.0145632, 0.0153141, -0.0003629, 0.0003977
6: 0.0044418, 0.0048175, 0.0044580, 0.0048232, -0.0001934, 0.0001765
7: -0.0145041, -0.0119001, -0.0145438, -0.0120122, -0.0012236, 0.0013409
8: 0.0052223, 0.0072882, 0.0051908, 0.0071992, -0.0009708, 0.0010638
9: 0.0071174, 0.0108331, 0.0070608, 0.0106731, -0.0017461, 0.0019133

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004661
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004667
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041045, -0.0040748, -0.0000157, 0.0000146
1: -0.0063974, -0.0052693, -0.0063691, -0.0052571, -0.0005862, 0.0005480
2: 0.9687864, 0.9701401, 0.9688202, 0.9701546, -0.0007035, 0.0006577
3: 0.0160793, 0.0260644, 0.0163291, 0.0261719, -0.0051890, 0.0048509
4: -0.0026754, -0.0019159, -0.0026836, -0.0019350, -0.0003689, 0.0003947
5: 0.0145664, 0.0153339, 0.0145581, 0.0153147, -0.0003729, 0.0003989
6: 0.0044484, 0.0048217, 0.0044577, 0.0048257, -0.0001940, 0.0001814
7: -0.0145331, -0.0119453, -0.0145609, -0.0120101, -0.0012572, 0.0013448
8: 0.0051993, 0.0072523, 0.0051772, 0.0072009, -0.0009974, 0.0010669
9: 0.0070760, 0.0107685, 0.0070363, 0.0106762, -0.0017939, 0.0019189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004662
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004708
time: 1.00 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041052, -0.0040750, -0.0000149, 0.0000141
1: -0.0063975, -0.0052939, -0.0063974, -0.0052651, -0.0005587, 0.0005274
2: 0.9687861, 0.9701107, 0.9687864, 0.9701450, -0.0006705, 0.0006329
3: 0.0160782, 0.0258468, 0.0160790, 0.0261012, -0.0049455, 0.0046683
4: -0.0026588, -0.0019159, -0.0026782, -0.0019159, -0.0003551, 0.0003761
5: 0.0145831, 0.0153340, 0.0145636, 0.0153340, -0.0003588, 0.0003802
6: 0.0044483, 0.0048136, 0.0044484, 0.0048231, -0.0001849, 0.0001745
7: -0.0144767, -0.0119451, -0.0145426, -0.0119453, -0.0012098, 0.0012817
8: 0.0052440, 0.0072525, 0.0051917, 0.0072523, -0.0009598, 0.0010168
9: 0.0071565, 0.0107689, 0.0070624, 0.0107686, -0.0017263, 0.0018289

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004688, upper bound: 0.0004831
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004688, upper bound: 0.0004843
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041053, -0.0040748, -0.0000151, 0.0000143
1: -0.0063789, -0.0052813, -0.0063984, -0.0052573, -0.0005640, 0.0005359
2: 0.9688084, 0.9701257, 0.9687851, 0.9701545, -0.0006769, 0.0006431
3: 0.0162426, 0.0259579, 0.0160698, 0.0261701, -0.0049925, 0.0047431
4: -0.0026673, -0.0019284, -0.0026834, -0.0019152, -0.0003607, 0.0003797
5: 0.0145746, 0.0153214, 0.0145583, 0.0153347, -0.0003646, 0.0003838
6: 0.0044545, 0.0048177, 0.0044480, 0.0048256, -0.0001867, 0.0001773
7: -0.0145055, -0.0119877, -0.0145605, -0.0119429, -0.0012292, 0.0012939
8: 0.0052212, 0.0072187, 0.0051776, 0.0072542, -0.0009752, 0.0010265
9: 0.0071154, 0.0107081, 0.0070370, 0.0107720, -0.0017540, 0.0018462

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004689, upper bound: 0.0004831
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004689, upper bound: 0.0004873
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041050, -0.0040746, -0.0000155, 0.0000140
1: -0.0063975, -0.0052939, -0.0063891, -0.0052512, -0.0005812, 0.0005257
2: 0.9687861, 0.9701107, 0.9687963, 0.9701619, -0.0006974, 0.0006309
3: 0.0160782, 0.0258468, 0.0161527, 0.0262247, -0.0051439, 0.0046534
4: -0.0026588, -0.0019159, -0.0026876, -0.0019215, -0.0003539, 0.0003912
5: 0.0145831, 0.0153340, 0.0145541, 0.0153283, -0.0003577, 0.0003954
6: 0.0044483, 0.0048136, 0.0044511, 0.0048277, -0.0001923, 0.0001740
7: -0.0144767, -0.0119451, -0.0145746, -0.0119644, -0.0012060, 0.0013331
8: 0.0052440, 0.0072525, 0.0051663, 0.0072372, -0.0009568, 0.0010576
9: 0.0071565, 0.0107689, 0.0070168, 0.0107414, -0.0017208, 0.0019022

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004798
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004823
time: 1.08 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041050, -0.0040744, -0.0000156, 0.0000144
1: -0.0063789, -0.0052813, -0.0063900, -0.0052437, -0.0005849, 0.0005403
2: 0.9688084, 0.9701257, 0.9687952, 0.9701709, -0.0007019, 0.0006484
3: 0.0162426, 0.0259579, 0.0161444, 0.0262911, -0.0051773, 0.0047825
4: -0.0026673, -0.0019284, -0.0026926, -0.0019209, -0.0003637, 0.0003938
5: 0.0145746, 0.0153214, 0.0145490, 0.0153289, -0.0003676, 0.0003980
6: 0.0044545, 0.0048177, 0.0044508, 0.0048302, -0.0001936, 0.0001788
7: -0.0145055, -0.0119877, -0.0145918, -0.0119622, -0.0012394, 0.0013417
8: 0.0052212, 0.0072187, 0.0051527, 0.0072389, -0.0009833, 0.0010645
9: 0.0071154, 0.0107081, 0.0069922, 0.0107444, -0.0017685, 0.0019146

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004798
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004861
time: 1.01 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041052, -0.0040750, -0.0000144, 0.0000137
1: -0.0064171, -0.0052819, -0.0063974, -0.0052651, -0.0005376, 0.0005114
2: 0.9687627, 0.9701250, 0.9687864, 0.9701450, -0.0006452, 0.0006138
3: 0.0159048, 0.0259526, 0.0160790, 0.0261012, -0.0047586, 0.0045269
4: -0.0026669, -0.0019027, -0.0026782, -0.0019159, -0.0003443, 0.0003619
5: 0.0145750, 0.0153473, 0.0145636, 0.0153340, -0.0003480, 0.0003658
6: 0.0044418, 0.0048175, 0.0044484, 0.0048231, -0.0001779, 0.0001693
7: -0.0145041, -0.0119001, -0.0145426, -0.0119453, -0.0011732, 0.0012332
8: 0.0052223, 0.0072882, 0.0051917, 0.0072523, -0.0009308, 0.0009784
9: 0.0071174, 0.0108331, 0.0070624, 0.0107686, -0.0016741, 0.0017597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004690
time: 0.79 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004690
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041053, -0.0040748, -0.0000145, 0.0000138
1: -0.0063974, -0.0052693, -0.0063984, -0.0052573, -0.0005440, 0.0005180
2: 0.9687864, 0.9701401, 0.9687851, 0.9701545, -0.0006528, 0.0006216
3: 0.0160793, 0.0260644, 0.0160698, 0.0261701, -0.0048151, 0.0045851
4: -0.0026754, -0.0019159, -0.0026834, -0.0019152, -0.0003487, 0.0003662
5: 0.0145664, 0.0153339, 0.0145583, 0.0153347, -0.0003524, 0.0003701
6: 0.0044484, 0.0048217, 0.0044480, 0.0048256, -0.0001800, 0.0001714
7: -0.0145331, -0.0119453, -0.0145605, -0.0119429, -0.0011883, 0.0012479
8: 0.0051993, 0.0072523, 0.0051776, 0.0072542, -0.0009427, 0.0009900
9: 0.0070760, 0.0107685, 0.0070370, 0.0107720, -0.0016956, 0.0017806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.17 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004690
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004721
time: 1.10 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041050, -0.0040746, -0.0000151, 0.0000137
1: -0.0064171, -0.0052819, -0.0063891, -0.0052512, -0.0005663, 0.0005118
2: 0.9687627, 0.9701250, 0.9687963, 0.9701619, -0.0006796, 0.0006142
3: 0.0159048, 0.0259526, 0.0161527, 0.0262247, -0.0050125, 0.0045303
4: -0.0026669, -0.0019027, -0.0026876, -0.0019215, -0.0003446, 0.0003812
5: 0.0145750, 0.0153473, 0.0145541, 0.0153283, -0.0003482, 0.0003853
6: 0.0044418, 0.0048175, 0.0044511, 0.0048277, -0.0001874, 0.0001694
7: -0.0145041, -0.0119001, -0.0145746, -0.0119644, -0.0011741, 0.0012990
8: 0.0052223, 0.0072882, 0.0051663, 0.0072372, -0.0009314, 0.0010306
9: 0.0071174, 0.0108331, 0.0070168, 0.0107414, -0.0016753, 0.0018536

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004662
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004668
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041050, -0.0040744, -0.0000152, 0.0000141
1: -0.0063974, -0.0052693, -0.0063900, -0.0052437, -0.0005700, 0.0005264
2: 0.9687864, 0.9701401, 0.9687952, 0.9701709, -0.0006840, 0.0006317
3: 0.0160793, 0.0260644, 0.0161444, 0.0262911, -0.0050450, 0.0046594
4: -0.0026754, -0.0019159, -0.0026926, -0.0019209, -0.0003544, 0.0003837
5: 0.0145664, 0.0153339, 0.0145490, 0.0153289, -0.0003582, 0.0003878
6: 0.0044484, 0.0048217, 0.0044508, 0.0048302, -0.0001886, 0.0001742
7: -0.0145331, -0.0119453, -0.0145918, -0.0119622, -0.0012075, 0.0013075
8: 0.0051993, 0.0072523, 0.0051527, 0.0072389, -0.0009580, 0.0010373
9: 0.0070760, 0.0107685, 0.0069922, 0.0107444, -0.0017230, 0.0018656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004662
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004708
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041048, -0.0040753, -0.0000142, 0.0000141
1: -0.0063885, -0.0052811, -0.0063789, -0.0052772, -0.0005303, 0.0005263
2: 0.9687970, 0.9701260, 0.9688085, 0.9701306, -0.0006364, 0.0006316
3: 0.0161576, 0.0259601, 0.0162427, 0.0259945, -0.0046937, 0.0046586
4: -0.0026674, -0.0019219, -0.0026701, -0.0019284, -0.0003543, 0.0003570
5: 0.0145744, 0.0153279, 0.0145718, 0.0153214, -0.0003581, 0.0003608
6: 0.0044513, 0.0048178, 0.0044545, 0.0048191, -0.0001755, 0.0001742
7: -0.0145060, -0.0119657, -0.0145149, -0.0119877, -0.0012073, 0.0012164
8: 0.0052207, 0.0072362, 0.0052137, 0.0072187, -0.0009578, 0.0009651
9: 0.0071146, 0.0107396, 0.0071019, 0.0107081, -0.0017227, 0.0017357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004736
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004744
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041048, -0.0040751, -0.0000141, 0.0000143
1: -0.0063681, -0.0052694, -0.0063799, -0.0052694, -0.0005290, 0.0005338
2: 0.9688215, 0.9701400, 0.9688073, 0.9701399, -0.0006348, 0.0006406
3: 0.0163382, 0.0260635, 0.0162336, 0.0260633, -0.0046823, 0.0047253
4: -0.0026753, -0.0019356, -0.0026753, -0.0019277, -0.0003594, 0.0003561
5: 0.0145665, 0.0153140, 0.0145665, 0.0153221, -0.0003632, 0.0003599
6: 0.0044580, 0.0048217, 0.0044541, 0.0048217, -0.0001751, 0.0001767
7: -0.0145328, -0.0120124, -0.0145328, -0.0119853, -0.0012246, 0.0012135
8: 0.0051995, 0.0071990, 0.0051995, 0.0072205, -0.0009715, 0.0009627
9: 0.0070764, 0.0106728, 0.0070764, 0.0107115, -0.0017474, 0.0017315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004736
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004784
time: 1.03 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041045, -0.0040750, -0.0000140, 0.0000133
1: -0.0063885, -0.0052811, -0.0063682, -0.0052646, -0.0005245, 0.0004975
2: 0.9687970, 0.9701260, 0.9688213, 0.9701458, -0.0006294, 0.0005970
3: 0.0161576, 0.0259601, 0.0163374, 0.0261056, -0.0046423, 0.0044036
4: -0.0026674, -0.0019219, -0.0026785, -0.0019356, -0.0003349, 0.0003531
5: 0.0145744, 0.0153279, 0.0145632, 0.0153141, -0.0003385, 0.0003568
6: 0.0044513, 0.0048178, 0.0044580, 0.0048232, -0.0001736, 0.0001646
7: -0.0145060, -0.0119657, -0.0145438, -0.0120122, -0.0011412, 0.0012031
8: 0.0052207, 0.0072362, 0.0051908, 0.0071992, -0.0009054, 0.0009545
9: 0.0071146, 0.0107396, 0.0070608, 0.0106731, -0.0016285, 0.0017167

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004714
time: 0.82 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004725
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041045, -0.0040748, -0.0000142, 0.0000134
1: -0.0063681, -0.0052694, -0.0063691, -0.0052571, -0.0005314, 0.0005025
2: 0.9688215, 0.9701400, 0.9688202, 0.9701546, -0.0006377, 0.0006030
3: 0.0163382, 0.0260635, 0.0163291, 0.0261719, -0.0047032, 0.0044475
4: -0.0026753, -0.0019356, -0.0026836, -0.0019350, -0.0003383, 0.0003577
5: 0.0145665, 0.0153140, 0.0145581, 0.0153147, -0.0003419, 0.0003615
6: 0.0044580, 0.0048217, 0.0044577, 0.0048257, -0.0001758, 0.0001663
7: -0.0145328, -0.0120124, -0.0145609, -0.0120101, -0.0011526, 0.0012189
8: 0.0051995, 0.0071990, 0.0051772, 0.0072009, -0.0009144, 0.0009670
9: 0.0070764, 0.0106728, 0.0070363, 0.0106762, -0.0016447, 0.0017393

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004714
time: 1.05 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004764
time: 1.02 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041048, -0.0040753, -0.0000150, 0.0000149
1: -0.0064096, -0.0052675, -0.0063789, -0.0052772, -0.0005603, 0.0005585
2: 0.9687717, 0.9701422, 0.9688085, 0.9701306, -0.0006724, 0.0006702
3: 0.0159711, 0.0260805, 0.0162427, 0.0259945, -0.0049598, 0.0049435
4: -0.0026766, -0.0019077, -0.0026701, -0.0019284, -0.0003760, 0.0003772
5: 0.0145652, 0.0153423, 0.0145718, 0.0153214, -0.0003800, 0.0003812
6: 0.0044443, 0.0048223, 0.0044545, 0.0048191, -0.0001854, 0.0001848
7: -0.0145372, -0.0119173, -0.0145149, -0.0119877, -0.0012811, 0.0012854
8: 0.0051960, 0.0072745, 0.0052137, 0.0072187, -0.0010164, 0.0010197
9: 0.0070701, 0.0108085, 0.0071019, 0.0107081, -0.0018281, 0.0018341

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004678
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004678
time: 0.80 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041048, -0.0040751, -0.0000149, 0.0000151
1: -0.0063890, -0.0052558, -0.0063799, -0.0052694, -0.0005590, 0.0005662
2: 0.9687964, 0.9701563, 0.9688073, 0.9701399, -0.0006708, 0.0006794
3: 0.0161538, 0.0261836, 0.0162336, 0.0260633, -0.0049478, 0.0050114
4: -0.0026844, -0.0019216, -0.0026753, -0.0019277, -0.0003811, 0.0003763
5: 0.0145572, 0.0153282, 0.0145665, 0.0153221, -0.0003852, 0.0003803
6: 0.0044511, 0.0048261, 0.0044541, 0.0048217, -0.0001850, 0.0001874
7: -0.0145640, -0.0119646, -0.0145328, -0.0119853, -0.0012988, 0.0012823
8: 0.0051748, 0.0072370, 0.0051995, 0.0072205, -0.0010304, 0.0010173
9: 0.0070320, 0.0107410, 0.0070764, 0.0107115, -0.0018532, 0.0018297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004678
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004718
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041045, -0.0040750, -0.0000149, 0.0000143
1: -0.0064096, -0.0052675, -0.0063682, -0.0052646, -0.0005588, 0.0005360
2: 0.9687717, 0.9701422, 0.9688213, 0.9701458, -0.0006706, 0.0006432
3: 0.0159711, 0.0260805, 0.0163374, 0.0261056, -0.0049462, 0.0047442
4: -0.0026766, -0.0019077, -0.0026785, -0.0019356, -0.0003608, 0.0003762
5: 0.0145652, 0.0153423, 0.0145632, 0.0153141, -0.0003647, 0.0003802
6: 0.0044443, 0.0048223, 0.0044580, 0.0048232, -0.0001849, 0.0001774
7: -0.0145372, -0.0119173, -0.0145438, -0.0120122, -0.0012295, 0.0012818
8: 0.0051960, 0.0072745, 0.0051908, 0.0071992, -0.0009754, 0.0010170
9: 0.0070701, 0.0108085, 0.0070608, 0.0106731, -0.0017544, 0.0018291

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004661
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004665
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041045, -0.0040748, -0.0000150, 0.0000144
1: -0.0063890, -0.0052558, -0.0063691, -0.0052571, -0.0005634, 0.0005409
2: 0.9687964, 0.9701563, 0.9688202, 0.9701546, -0.0006761, 0.0006491
3: 0.0161538, 0.0261836, 0.0163291, 0.0261719, -0.0049868, 0.0047873
4: -0.0026844, -0.0019216, -0.0026836, -0.0019350, -0.0003641, 0.0003793
5: 0.0145572, 0.0153282, 0.0145581, 0.0153147, -0.0003680, 0.0003833
6: 0.0044511, 0.0048261, 0.0044577, 0.0048257, -0.0001864, 0.0001790
7: -0.0145640, -0.0119646, -0.0145609, -0.0120101, -0.0012407, 0.0012924
8: 0.0051748, 0.0072370, 0.0051772, 0.0072009, -0.0009843, 0.0010253
9: 0.0070320, 0.0107410, 0.0070363, 0.0106762, -0.0017704, 0.0018441

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004661
time: 0.79 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004705
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041052, -0.0040750, -0.0000152, 0.0000149
1: -0.0063885, -0.0052811, -0.0063974, -0.0052651, -0.0005674, 0.0005582
2: 0.9687970, 0.9701260, 0.9687864, 0.9701450, -0.0006809, 0.0006699
3: 0.0161576, 0.0259601, 0.0160790, 0.0261012, -0.0050224, 0.0049410
4: -0.0026674, -0.0019219, -0.0026782, -0.0019159, -0.0003758, 0.0003820
5: 0.0145744, 0.0153279, 0.0145636, 0.0153340, -0.0003798, 0.0003861
6: 0.0044513, 0.0048178, 0.0044484, 0.0048231, -0.0001878, 0.0001847
7: -0.0145060, -0.0119657, -0.0145426, -0.0119453, -0.0012805, 0.0013016
8: 0.0052207, 0.0072362, 0.0051917, 0.0072523, -0.0010159, 0.0010326
9: 0.0071146, 0.0107396, 0.0070624, 0.0107686, -0.0018272, 0.0018573

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.18 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004819
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004831
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041053, -0.0040748, -0.0000151, 0.0000152
1: -0.0063681, -0.0052694, -0.0063984, -0.0052573, -0.0005647, 0.0005679
2: 0.9688215, 0.9701400, 0.9687851, 0.9701545, -0.0006777, 0.0006815
3: 0.0163382, 0.0260635, 0.0160698, 0.0261701, -0.0049983, 0.0050268
4: -0.0026753, -0.0019356, -0.0026834, -0.0019152, -0.0003823, 0.0003802
5: 0.0145665, 0.0153140, 0.0145583, 0.0153347, -0.0003864, 0.0003842
6: 0.0044580, 0.0048217, 0.0044480, 0.0048256, -0.0001869, 0.0001879
7: -0.0145328, -0.0120124, -0.0145605, -0.0119429, -0.0013027, 0.0012954
8: 0.0051995, 0.0071990, 0.0051776, 0.0072542, -0.0010335, 0.0010277
9: 0.0070764, 0.0106728, 0.0070370, 0.0107720, -0.0018589, 0.0018484

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004820
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004870
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041050, -0.0040746, -0.0000150, 0.0000141
1: -0.0063885, -0.0052811, -0.0063891, -0.0052512, -0.0005615, 0.0005294
2: 0.9687970, 0.9701260, 0.9687963, 0.9701619, -0.0006738, 0.0006353
3: 0.0161576, 0.0259601, 0.0161527, 0.0262247, -0.0049699, 0.0046858
4: -0.0026674, -0.0019219, -0.0026876, -0.0019215, -0.0003564, 0.0003780
5: 0.0145744, 0.0153279, 0.0145541, 0.0153283, -0.0003602, 0.0003820
6: 0.0044513, 0.0048178, 0.0044511, 0.0048277, -0.0001858, 0.0001752
7: -0.0145060, -0.0119657, -0.0145746, -0.0119644, -0.0012144, 0.0012880
8: 0.0052207, 0.0072362, 0.0051663, 0.0072372, -0.0009634, 0.0010218
9: 0.0071146, 0.0107396, 0.0070168, 0.0107414, -0.0017328, 0.0018379

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004798
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004819
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041050, -0.0040744, -0.0000151, 0.0000143
1: -0.0063681, -0.0052694, -0.0063900, -0.0052437, -0.0005671, 0.0005367
2: 0.9688215, 0.9701400, 0.9687952, 0.9701709, -0.0006805, 0.0006441
3: 0.0163382, 0.0260635, 0.0161444, 0.0262911, -0.0050192, 0.0047505
4: -0.0026753, -0.0019356, -0.0026926, -0.0019209, -0.0003613, 0.0003817
5: 0.0145665, 0.0153140, 0.0145490, 0.0153289, -0.0003652, 0.0003858
6: 0.0044580, 0.0048217, 0.0044508, 0.0048302, -0.0001877, 0.0001776
7: -0.0145328, -0.0120124, -0.0145918, -0.0119622, -0.0012311, 0.0013008
8: 0.0051995, 0.0071990, 0.0051527, 0.0072389, -0.0009767, 0.0010320
9: 0.0070764, 0.0106728, 0.0069922, 0.0107444, -0.0017567, 0.0018561

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.19 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004798
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004858
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041052, -0.0040750, -0.0000146, 0.0000145
1: -0.0064096, -0.0052675, -0.0063974, -0.0052651, -0.0005464, 0.0005425
2: 0.9687717, 0.9701422, 0.9687864, 0.9701450, -0.0006557, 0.0006510
3: 0.0159711, 0.0260805, 0.0160790, 0.0261012, -0.0048366, 0.0048018
4: -0.0026766, -0.0019077, -0.0026782, -0.0019159, -0.0003652, 0.0003679
5: 0.0145652, 0.0153423, 0.0145636, 0.0153340, -0.0003691, 0.0003718
6: 0.0044443, 0.0048223, 0.0044484, 0.0048231, -0.0001808, 0.0001795
7: -0.0145372, -0.0119173, -0.0145426, -0.0119453, -0.0012444, 0.0012534
8: 0.0051960, 0.0072745, 0.0051917, 0.0072523, -0.0009873, 0.0009944
9: 0.0070701, 0.0108085, 0.0070624, 0.0107686, -0.0017757, 0.0017886

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004678
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004679
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041053, -0.0040748, -0.0000145, 0.0000147
1: -0.0063890, -0.0052558, -0.0063984, -0.0052573, -0.0005447, 0.0005500
2: 0.9687964, 0.9701563, 0.9687851, 0.9701545, -0.0006537, 0.0006600
3: 0.0161538, 0.0261836, 0.0160698, 0.0261701, -0.0048217, 0.0048680
4: -0.0026844, -0.0019216, -0.0026834, -0.0019152, -0.0003702, 0.0003667
5: 0.0145572, 0.0153282, 0.0145583, 0.0153347, -0.0003742, 0.0003706
6: 0.0044511, 0.0048261, 0.0044480, 0.0048256, -0.0001803, 0.0001820
7: -0.0145640, -0.0119646, -0.0145605, -0.0119429, -0.0012616, 0.0012496
8: 0.0051748, 0.0072370, 0.0051776, 0.0072542, -0.0010009, 0.0009914
9: 0.0070320, 0.0107410, 0.0070370, 0.0107720, -0.0018002, 0.0017830

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004679
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004718
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041050, -0.0040746, -0.0000144, 0.0000137
1: -0.0064096, -0.0052675, -0.0063891, -0.0052512, -0.0005407, 0.0005143
2: 0.9687717, 0.9701422, 0.9687963, 0.9701619, -0.0006489, 0.0006172
3: 0.0159711, 0.0260805, 0.0161527, 0.0262247, -0.0047862, 0.0045523
4: -0.0026766, -0.0019077, -0.0026876, -0.0019215, -0.0003462, 0.0003640
5: 0.0145652, 0.0153423, 0.0145541, 0.0153283, -0.0003499, 0.0003679
6: 0.0044443, 0.0048223, 0.0044511, 0.0048277, -0.0001789, 0.0001702
7: -0.0145372, -0.0119173, -0.0145746, -0.0119644, -0.0011798, 0.0012404
8: 0.0051960, 0.0072745, 0.0051663, 0.0072372, -0.0009360, 0.0009841
9: 0.0070701, 0.0108085, 0.0070168, 0.0107414, -0.0016835, 0.0017699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004662
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004666
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041050, -0.0040744, -0.0000146, 0.0000139
1: -0.0063890, -0.0052558, -0.0063900, -0.0052437, -0.0005476, 0.0005189
2: 0.9687964, 0.9701563, 0.9687952, 0.9701709, -0.0006572, 0.0006227
3: 0.0161538, 0.0261836, 0.0161444, 0.0262911, -0.0048473, 0.0045928
4: -0.0026844, -0.0019216, -0.0026926, -0.0019209, -0.0003493, 0.0003687
5: 0.0145572, 0.0153282, 0.0145490, 0.0153289, -0.0003530, 0.0003726
6: 0.0044511, 0.0048261, 0.0044508, 0.0048302, -0.0001812, 0.0001717
7: -0.0145640, -0.0119646, -0.0145918, -0.0119622, -0.0011903, 0.0012562
8: 0.0051748, 0.0072370, 0.0051527, 0.0072389, -0.0009443, 0.0009966
9: 0.0070320, 0.0107410, 0.0069922, 0.0107444, -0.0016984, 0.0017925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004661
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004705
time: 0.92 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.09 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004746
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004754
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004754, upper bound: 0.0004747
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004754, upper bound: 0.0004786
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004735, upper bound: 0.0004714
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004735, upper bound: 0.0004725
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004744, upper bound: 0.0004714
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004744, upper bound: 0.0004764
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004688
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004689
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004688
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004688
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004661
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004820, upper bound: 0.0004667
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004662
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004708
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004688, upper bound: 0.0004831
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004688, upper bound: 0.0004843
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004689, upper bound: 0.0004831
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004689, upper bound: 0.0004873
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004798
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004823
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004798
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004678, upper bound: 0.0004861
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004690
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004690
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004690
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004843, upper bound: 0.0004721
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004662
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004819, upper bound: 0.0004668
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004662
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004831, upper bound: 0.0004708
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004736
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004744
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004736
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004784
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004714
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004714, upper bound: 0.0004725
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004714
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004725, upper bound: 0.0004764
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004678
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004678
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004678
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004718
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004661
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004665
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004661
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004705
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004819
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004831
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004820
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004870
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004798
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004819
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004798
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004858
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004678
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004679
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004679
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004718
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004662
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004798, upper bound: 0.0004666
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004661
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.09
Output dim: 2, lower bound: -0.0004823, upper bound: 0.0004705

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041052, -0.0040758, -0.0000134, 0.0000134
1: -0.0063975, -0.0052939, -0.0063975, -0.0052939, -0.0005007, 0.0005007
2: 0.9687861, 0.9701107, 0.9687861, 0.9701107, -0.0006008, 0.0006008
3: 0.0160782, 0.0258468, 0.0160782, 0.0258468, -0.0044316, 0.0044316
4: -0.0026588, -0.0019159, -0.0026588, -0.0019159, -0.0003370, 0.0003370
5: 0.0145831, 0.0153340, 0.0145831, 0.0153340, -0.0003406, 0.0003406
6: 0.0044483, 0.0048136, 0.0044483, 0.0048136, -0.0001657, 0.0001657
7: -0.0144767, -0.0119451, -0.0144767, -0.0119451, -0.0011485, 0.0011485
8: 0.0052440, 0.0072525, 0.0052440, 0.0072525, -0.0009112, 0.0009112
9: 0.0071565, 0.0107689, 0.0071565, 0.0107689, -0.0016388, 0.0016388

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004639
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041048, -0.0040754, -0.0000141, 0.0000132
1: -0.0063975, -0.0052939, -0.0063789, -0.0052813, -0.0005276, 0.0004943
2: 0.9687861, 0.9701107, 0.9688084, 0.9701257, -0.0006331, 0.0005932
3: 0.0160782, 0.0258468, 0.0162426, 0.0259579, -0.0046699, 0.0043755
4: -0.0026588, -0.0019159, -0.0026673, -0.0019284, -0.0003328, 0.0003552
5: 0.0145831, 0.0153340, 0.0145746, 0.0153214, -0.0003363, 0.0003590
6: 0.0044483, 0.0048136, 0.0044545, 0.0048177, -0.0001746, 0.0001636
7: -0.0144767, -0.0119451, -0.0145055, -0.0119877, -0.0011339, 0.0012102
8: 0.0052440, 0.0072525, 0.0052212, 0.0072187, -0.0008996, 0.0009601
9: 0.0071565, 0.0107689, 0.0071154, 0.0107081, -0.0016180, 0.0017269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004642
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004566
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041052, -0.0040758, -0.0000132, 0.0000141
1: -0.0063789, -0.0052813, -0.0063975, -0.0052939, -0.0004943, 0.0005276
2: 0.9688084, 0.9701257, 0.9687861, 0.9701107, -0.0005932, 0.0006331
3: 0.0162426, 0.0259579, 0.0160782, 0.0258468, -0.0043755, 0.0046699
4: -0.0026673, -0.0019284, -0.0026588, -0.0019159, -0.0003552, 0.0003328
5: 0.0145746, 0.0153214, 0.0145831, 0.0153340, -0.0003590, 0.0003363
6: 0.0044545, 0.0048177, 0.0044483, 0.0048136, -0.0001636, 0.0001746
7: -0.0145055, -0.0119877, -0.0144767, -0.0119451, -0.0012102, 0.0011339
8: 0.0052212, 0.0072187, 0.0052440, 0.0072525, -0.0009601, 0.0008996
9: 0.0071154, 0.0107081, 0.0071565, 0.0107689, -0.0017269, 0.0016180

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004624
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041048, -0.0040754, -0.0000133, 0.0000133
1: -0.0063789, -0.0052813, -0.0063789, -0.0052813, -0.0004996, 0.0004996
2: 0.9688084, 0.9701257, 0.9688084, 0.9701257, -0.0005995, 0.0005995
3: 0.0162426, 0.0259579, 0.0162426, 0.0259579, -0.0044222, 0.0044222
4: -0.0026673, -0.0019284, -0.0026673, -0.0019284, -0.0003363, 0.0003363
5: 0.0145746, 0.0153214, 0.0145746, 0.0153214, -0.0003399, 0.0003399
6: 0.0044545, 0.0048177, 0.0044545, 0.0048177, -0.0001653, 0.0001653
7: -0.0145055, -0.0119877, -0.0145055, -0.0119877, -0.0011460, 0.0011460
8: 0.0052212, 0.0072187, 0.0052212, 0.0072187, -0.0009092, 0.0009092
9: 0.0071154, 0.0107081, 0.0071154, 0.0107081, -0.0016353, 0.0016353

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004685
time: 0.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004653
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041050, -0.0040754, -0.0000142, 0.0000136
1: -0.0063975, -0.0052939, -0.0063885, -0.0052811, -0.0005315, 0.0005094
2: 0.9687861, 0.9701107, 0.9687970, 0.9701260, -0.0006378, 0.0006112
3: 0.0160782, 0.0258468, 0.0161576, 0.0259601, -0.0047043, 0.0045085
4: -0.0026588, -0.0019159, -0.0026674, -0.0019219, -0.0003429, 0.0003578
5: 0.0145831, 0.0153340, 0.0145744, 0.0153279, -0.0003466, 0.0003616
6: 0.0044483, 0.0048136, 0.0044513, 0.0048178, -0.0001759, 0.0001686
7: -0.0144767, -0.0119451, -0.0145060, -0.0119657, -0.0011684, 0.0012192
8: 0.0052440, 0.0072525, 0.0052207, 0.0072362, -0.0009270, 0.0009672
9: 0.0071565, 0.0107689, 0.0071146, 0.0107396, -0.0016672, 0.0017396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004592
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041045, -0.0040751, -0.0000147, 0.0000132
1: -0.0063975, -0.0052939, -0.0063681, -0.0052694, -0.0005502, 0.0004950
2: 0.9687861, 0.9701107, 0.9688215, 0.9701400, -0.0006602, 0.0005940
3: 0.0160782, 0.0258468, 0.0163382, 0.0260635, -0.0048696, 0.0043813
4: -0.0026588, -0.0019159, -0.0026753, -0.0019356, -0.0003332, 0.0003704
5: 0.0145831, 0.0153340, 0.0145665, 0.0153140, -0.0003368, 0.0003743
6: 0.0044483, 0.0048136, 0.0044580, 0.0048217, -0.0001821, 0.0001638
7: -0.0144767, -0.0119451, -0.0145328, -0.0120124, -0.0011354, 0.0012620
8: 0.0052440, 0.0072525, 0.0051995, 0.0071990, -0.0009008, 0.0010012
9: 0.0071565, 0.0107689, 0.0070764, 0.0106728, -0.0016202, 0.0018008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004609
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004504
time: 1.01 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041050, -0.0040754, -0.0000140, 0.0000143
1: -0.0063789, -0.0052813, -0.0063885, -0.0052811, -0.0005251, 0.0005363
2: 0.9688084, 0.9701257, 0.9687970, 0.9701260, -0.0006302, 0.0006436
3: 0.0162426, 0.0259579, 0.0161576, 0.0259601, -0.0046481, 0.0047468
4: -0.0026673, -0.0019284, -0.0026674, -0.0019219, -0.0003610, 0.0003535
5: 0.0145746, 0.0153214, 0.0145744, 0.0153279, -0.0003649, 0.0003573
6: 0.0044545, 0.0048177, 0.0044513, 0.0048178, -0.0001738, 0.0001775
7: -0.0145055, -0.0119877, -0.0145060, -0.0119657, -0.0012302, 0.0012046
8: 0.0052212, 0.0072187, 0.0052207, 0.0072362, -0.0009760, 0.0009557
9: 0.0071154, 0.0107081, 0.0071146, 0.0107396, -0.0017553, 0.0017189

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004580
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041045, -0.0040751, -0.0000142, 0.0000136
1: -0.0063789, -0.0052813, -0.0063681, -0.0052694, -0.0005317, 0.0005080
2: 0.9688084, 0.9701257, 0.9688215, 0.9701400, -0.0006380, 0.0006096
3: 0.0162426, 0.0259579, 0.0163382, 0.0260635, -0.0047059, 0.0044961
4: -0.0026673, -0.0019284, -0.0026753, -0.0019356, -0.0003420, 0.0003579
5: 0.0145746, 0.0153214, 0.0145665, 0.0153140, -0.0003456, 0.0003617
6: 0.0044545, 0.0048177, 0.0044580, 0.0048217, -0.0001759, 0.0001681
7: -0.0145055, -0.0119877, -0.0145328, -0.0120124, -0.0011652, 0.0012196
8: 0.0052212, 0.0072187, 0.0051995, 0.0071990, -0.0009244, 0.0009675
9: 0.0071154, 0.0107081, 0.0070764, 0.0106728, -0.0016626, 0.0017402

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004666
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004620
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041052, -0.0040758, -0.0000143, 0.0000144
1: -0.0064171, -0.0052819, -0.0063975, -0.0052939, -0.0005349, 0.0005383
2: 0.9687627, 0.9701250, 0.9687861, 0.9701107, -0.0006419, 0.0006460
3: 0.0159048, 0.0259526, 0.0160782, 0.0258468, -0.0047346, 0.0047647
4: -0.0026669, -0.0019027, -0.0026588, -0.0019159, -0.0003624, 0.0003601
5: 0.0145750, 0.0153473, 0.0145831, 0.0153340, -0.0003663, 0.0003639
6: 0.0044418, 0.0048175, 0.0044483, 0.0048136, -0.0001770, 0.0001781
7: -0.0145041, -0.0119001, -0.0144767, -0.0119451, -0.0012348, 0.0012270
8: 0.0052223, 0.0072882, 0.0052440, 0.0072525, -0.0009796, 0.0009735
9: 0.0071174, 0.0108331, 0.0071565, 0.0107689, -0.0017620, 0.0017509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004585
time: 0.99 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004517
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041048, -0.0040754, -0.0000150, 0.0000142
1: -0.0064171, -0.0052819, -0.0063789, -0.0052813, -0.0005618, 0.0005320
2: 0.9687627, 0.9701250, 0.9688084, 0.9701257, -0.0006742, 0.0006384
3: 0.0159048, 0.0259526, 0.0162426, 0.0259579, -0.0049729, 0.0047086
4: -0.0026669, -0.0019027, -0.0026673, -0.0019284, -0.0003581, 0.0003782
5: 0.0145750, 0.0153473, 0.0145746, 0.0153214, -0.0003619, 0.0003823
6: 0.0044418, 0.0048175, 0.0044545, 0.0048177, -0.0001859, 0.0001760
7: -0.0145041, -0.0119001, -0.0145055, -0.0119877, -0.0012203, 0.0012888
8: 0.0052223, 0.0072882, 0.0052212, 0.0072187, -0.0009681, 0.0010225
9: 0.0071174, 0.0108331, 0.0071154, 0.0107081, -0.0017412, 0.0018390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004585
time: 1.03 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004521
time: 0.78 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041052, -0.0040758, -0.0000141, 0.0000150
1: -0.0063974, -0.0052693, -0.0063975, -0.0052939, -0.0005263, 0.0005616
2: 0.9687864, 0.9701401, 0.9687861, 0.9701107, -0.0006316, 0.0006740
3: 0.0160793, 0.0260644, 0.0160782, 0.0258468, -0.0046588, 0.0049710
4: -0.0026754, -0.0019159, -0.0026588, -0.0019159, -0.0003781, 0.0003543
5: 0.0145664, 0.0153339, 0.0145831, 0.0153340, -0.0003821, 0.0003581
6: 0.0044484, 0.0048217, 0.0044483, 0.0048136, -0.0001742, 0.0001859
7: -0.0145331, -0.0119453, -0.0144767, -0.0119451, -0.0012883, 0.0012074
8: 0.0051993, 0.0072523, 0.0052440, 0.0072525, -0.0010220, 0.0009579
9: 0.0070760, 0.0107685, 0.0071565, 0.0107689, -0.0018383, 0.0017228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004578
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004521
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041048, -0.0040754, -0.0000143, 0.0000144
1: -0.0063974, -0.0052693, -0.0063789, -0.0052813, -0.0005336, 0.0005377
2: 0.9687864, 0.9701401, 0.9688084, 0.9701257, -0.0006404, 0.0006452
3: 0.0160793, 0.0260644, 0.0162426, 0.0259579, -0.0047235, 0.0047591
4: -0.0026754, -0.0019159, -0.0026673, -0.0019284, -0.0003620, 0.0003592
5: 0.0145664, 0.0153339, 0.0145746, 0.0153214, -0.0003658, 0.0003631
6: 0.0044484, 0.0048217, 0.0044545, 0.0048177, -0.0001766, 0.0001779
7: -0.0145331, -0.0119453, -0.0145055, -0.0119877, -0.0012334, 0.0012241
8: 0.0051993, 0.0072523, 0.0052212, 0.0072187, -0.0009785, 0.0009712
9: 0.0070760, 0.0107685, 0.0071154, 0.0107081, -0.0017599, 0.0017467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004627
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004594
time: 1.14 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041050, -0.0040754, -0.0000151, 0.0000146
1: -0.0064171, -0.0052819, -0.0063885, -0.0052811, -0.0005657, 0.0005470
2: 0.9687627, 0.9701250, 0.9687970, 0.9701260, -0.0006789, 0.0006564
3: 0.0159048, 0.0259526, 0.0161576, 0.0259601, -0.0050073, 0.0048416
4: -0.0026669, -0.0019027, -0.0026674, -0.0019219, -0.0003682, 0.0003808
5: 0.0145750, 0.0153473, 0.0145744, 0.0153279, -0.0003722, 0.0003849
6: 0.0044418, 0.0048175, 0.0044513, 0.0048178, -0.0001872, 0.0001810
7: -0.0145041, -0.0119001, -0.0145060, -0.0119657, -0.0012547, 0.0012977
8: 0.0052223, 0.0072882, 0.0052207, 0.0072362, -0.0009955, 0.0010295
9: 0.0071174, 0.0108331, 0.0071146, 0.0107396, -0.0017904, 0.0018517

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004550
time: 0.89 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041045, -0.0040751, -0.0000156, 0.0000142
1: -0.0064171, -0.0052819, -0.0063681, -0.0052694, -0.0005844, 0.0005326
2: 0.9687627, 0.9701250, 0.9688215, 0.9701400, -0.0007013, 0.0006392
3: 0.0159048, 0.0259526, 0.0163382, 0.0260635, -0.0051727, 0.0047144
4: -0.0026669, -0.0019027, -0.0026753, -0.0019356, -0.0003586, 0.0003934
5: 0.0145750, 0.0153473, 0.0145665, 0.0153140, -0.0003624, 0.0003976
6: 0.0044418, 0.0048175, 0.0044580, 0.0048217, -0.0001934, 0.0001763
7: -0.0145041, -0.0119001, -0.0145328, -0.0120124, -0.0012218, 0.0013405
8: 0.0052223, 0.0072882, 0.0051995, 0.0071990, -0.0009693, 0.0010635
9: 0.0071174, 0.0108331, 0.0070764, 0.0106728, -0.0017434, 0.0019128

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004555
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004469
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041050, -0.0040754, -0.0000149, 0.0000152
1: -0.0063974, -0.0052693, -0.0063885, -0.0052811, -0.0005571, 0.0005703
2: 0.9687864, 0.9701401, 0.9687970, 0.9701260, -0.0006686, 0.0006844
3: 0.0160793, 0.0260644, 0.0161576, 0.0259601, -0.0049315, 0.0050478
4: -0.0026754, -0.0019159, -0.0026674, -0.0019219, -0.0003839, 0.0003751
5: 0.0145664, 0.0153339, 0.0145744, 0.0153279, -0.0003880, 0.0003791
6: 0.0044484, 0.0048217, 0.0044513, 0.0048178, -0.0001844, 0.0001887
7: -0.0145331, -0.0119453, -0.0145060, -0.0119657, -0.0013082, 0.0012780
8: 0.0051993, 0.0072523, 0.0052207, 0.0072362, -0.0010379, 0.0010139
9: 0.0070760, 0.0107685, 0.0071146, 0.0107396, -0.0018667, 0.0018236

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004539
time: 1.02 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041045, -0.0040751, -0.0000151, 0.0000146
1: -0.0063974, -0.0052693, -0.0063681, -0.0052694, -0.0005657, 0.0005460
2: 0.9687864, 0.9701401, 0.9688215, 0.9701400, -0.0006789, 0.0006552
3: 0.0160793, 0.0260644, 0.0163382, 0.0260635, -0.0050072, 0.0048330
4: -0.0026754, -0.0019159, -0.0026753, -0.0019356, -0.0003676, 0.0003808
5: 0.0145664, 0.0153339, 0.0145665, 0.0153140, -0.0003715, 0.0003849
6: 0.0044484, 0.0048217, 0.0044580, 0.0048217, -0.0001872, 0.0001807
7: -0.0145331, -0.0119453, -0.0145328, -0.0120124, -0.0012525, 0.0012977
8: 0.0051993, 0.0072523, 0.0051995, 0.0071990, -0.0009937, 0.0010295
9: 0.0070760, 0.0107685, 0.0070764, 0.0106728, -0.0017872, 0.0018516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004612
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004566
time: 1.03 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041058, -0.0040755, -0.0000144, 0.0000143
1: -0.0063975, -0.0052939, -0.0064171, -0.0052819, -0.0005383, 0.0005349
2: 0.9687861, 0.9701107, 0.9687627, 0.9701250, -0.0006460, 0.0006419
3: 0.0160782, 0.0258468, 0.0159048, 0.0259526, -0.0047647, 0.0047346
4: -0.0026588, -0.0019159, -0.0026669, -0.0019027, -0.0003601, 0.0003624
5: 0.0145831, 0.0153340, 0.0145750, 0.0153473, -0.0003639, 0.0003663
6: 0.0044483, 0.0048136, 0.0044418, 0.0048175, -0.0001781, 0.0001770
7: -0.0144767, -0.0119451, -0.0145041, -0.0119001, -0.0012270, 0.0012348
8: 0.0052440, 0.0072525, 0.0052223, 0.0072882, -0.0009735, 0.0009796
9: 0.0071565, 0.0107689, 0.0071174, 0.0108331, -0.0017509, 0.0017620

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004753
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004663
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041052, -0.0040751, -0.0000150, 0.0000141
1: -0.0063975, -0.0052939, -0.0063974, -0.0052693, -0.0005616, 0.0005263
2: 0.9687861, 0.9701107, 0.9687864, 0.9701401, -0.0006740, 0.0006316
3: 0.0160782, 0.0258468, 0.0160793, 0.0260644, -0.0049710, 0.0046588
4: -0.0026588, -0.0019159, -0.0026754, -0.0019159, -0.0003543, 0.0003781
5: 0.0145831, 0.0153340, 0.0145664, 0.0153339, -0.0003581, 0.0003821
6: 0.0044483, 0.0048136, 0.0044484, 0.0048217, -0.0001859, 0.0001742
7: -0.0144767, -0.0119451, -0.0145331, -0.0119453, -0.0012074, 0.0012883
8: 0.0052440, 0.0072525, 0.0051993, 0.0072523, -0.0009579, 0.0010220
9: 0.0071565, 0.0107689, 0.0070760, 0.0107685, -0.0017228, 0.0018383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004753
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004674
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041058, -0.0040755, -0.0000142, 0.0000150
1: -0.0063789, -0.0052813, -0.0064171, -0.0052819, -0.0005320, 0.0005618
2: 0.9688084, 0.9701257, 0.9687627, 0.9701250, -0.0006384, 0.0006742
3: 0.0162426, 0.0259579, 0.0159048, 0.0259526, -0.0047086, 0.0049729
4: -0.0026673, -0.0019284, -0.0026669, -0.0019027, -0.0003782, 0.0003581
5: 0.0145746, 0.0153214, 0.0145750, 0.0153473, -0.0003823, 0.0003619
6: 0.0044545, 0.0048177, 0.0044418, 0.0048175, -0.0001760, 0.0001859
7: -0.0145055, -0.0119877, -0.0145041, -0.0119001, -0.0012888, 0.0012203
8: 0.0052212, 0.0072187, 0.0052223, 0.0072882, -0.0010225, 0.0009681
9: 0.0071154, 0.0107081, 0.0071174, 0.0108331, -0.0018390, 0.0017412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004726
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004661
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041052, -0.0040751, -0.0000144, 0.0000143
1: -0.0063789, -0.0052813, -0.0063974, -0.0052693, -0.0005377, 0.0005336
2: 0.9688084, 0.9701257, 0.9687864, 0.9701401, -0.0006452, 0.0006404
3: 0.0162426, 0.0259579, 0.0160793, 0.0260644, -0.0047591, 0.0047235
4: -0.0026673, -0.0019284, -0.0026754, -0.0019159, -0.0003592, 0.0003620
5: 0.0145746, 0.0153214, 0.0145664, 0.0153339, -0.0003631, 0.0003658
6: 0.0044545, 0.0048177, 0.0044484, 0.0048217, -0.0001779, 0.0001766
7: -0.0145055, -0.0119877, -0.0145331, -0.0119453, -0.0012241, 0.0012334
8: 0.0052212, 0.0072187, 0.0051993, 0.0072523, -0.0009712, 0.0009785
9: 0.0071154, 0.0107081, 0.0070760, 0.0107685, -0.0017467, 0.0017599

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004774
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004738
time: 0.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041056, -0.0040751, -0.0000151, 0.0000144
1: -0.0063975, -0.0052939, -0.0064096, -0.0052675, -0.0005637, 0.0005394
2: 0.9687861, 0.9701107, 0.9687717, 0.9701422, -0.0006764, 0.0006473
3: 0.0160782, 0.0258468, 0.0159711, 0.0260805, -0.0049892, 0.0047745
4: -0.0026588, -0.0019159, -0.0026766, -0.0019077, -0.0003631, 0.0003795
5: 0.0145831, 0.0153340, 0.0145652, 0.0153423, -0.0003670, 0.0003835
6: 0.0044483, 0.0048136, 0.0044443, 0.0048223, -0.0001865, 0.0001785
7: -0.0144767, -0.0119451, -0.0145372, -0.0119173, -0.0012373, 0.0012930
8: 0.0052440, 0.0072525, 0.0051960, 0.0072745, -0.0009817, 0.0010258
9: 0.0071565, 0.0107689, 0.0070701, 0.0108085, -0.0017656, 0.0018450

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004715
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040758, -0.0041050, -0.0040748, -0.0000155, 0.0000140
1: -0.0063975, -0.0052939, -0.0063890, -0.0052558, -0.0005804, 0.0005250
2: 0.9687861, 0.9701107, 0.9687964, 0.9701563, -0.0006965, 0.0006300
3: 0.0160782, 0.0258468, 0.0161538, 0.0261836, -0.0051374, 0.0046467
4: -0.0026588, -0.0019159, -0.0026844, -0.0019216, -0.0003534, 0.0003907
5: 0.0145831, 0.0153340, 0.0145572, 0.0153282, -0.0003572, 0.0003949
6: 0.0044483, 0.0048136, 0.0044511, 0.0048261, -0.0001921, 0.0001737
7: -0.0144767, -0.0119451, -0.0145640, -0.0119646, -0.0012042, 0.0013314
8: 0.0052440, 0.0072525, 0.0051748, 0.0072370, -0.0009554, 0.0010563
9: 0.0071565, 0.0107689, 0.0070320, 0.0107410, -0.0017184, 0.0018998

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004719
time: 1.08 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004621
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041056, -0.0040751, -0.0000149, 0.0000151
1: -0.0063789, -0.0052813, -0.0064096, -0.0052675, -0.0005573, 0.0005663
2: 0.9688084, 0.9701257, 0.9687717, 0.9701422, -0.0006688, 0.0006796
3: 0.0162426, 0.0259579, 0.0159711, 0.0260805, -0.0049330, 0.0050128
4: -0.0026673, -0.0019284, -0.0026766, -0.0019077, -0.0003813, 0.0003752
5: 0.0145746, 0.0153214, 0.0145652, 0.0153423, -0.0003853, 0.0003792
6: 0.0044545, 0.0048177, 0.0044443, 0.0048223, -0.0001844, 0.0001874
7: -0.0145055, -0.0119877, -0.0145372, -0.0119173, -0.0012991, 0.0012784
8: 0.0052212, 0.0072187, 0.0051960, 0.0072745, -0.0010306, 0.0010143
9: 0.0071154, 0.0107081, 0.0070701, 0.0108085, -0.0018537, 0.0018242

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004682
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
time: 0.89 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041048, -0.0040754, -0.0041050, -0.0040748, -0.0000151, 0.0000144
1: -0.0063789, -0.0052813, -0.0063890, -0.0052558, -0.0005640, 0.0005382
2: 0.9688084, 0.9701257, 0.9687964, 0.9701563, -0.0006768, 0.0006458
3: 0.0162426, 0.0259579, 0.0161538, 0.0261836, -0.0049920, 0.0047634
4: -0.0026673, -0.0019284, -0.0026844, -0.0019216, -0.0003623, 0.0003797
5: 0.0145746, 0.0153214, 0.0145572, 0.0153282, -0.0003662, 0.0003837
6: 0.0044545, 0.0048177, 0.0044511, 0.0048261, -0.0001866, 0.0001781
7: -0.0145055, -0.0119877, -0.0145640, -0.0119646, -0.0012345, 0.0012937
8: 0.0052212, 0.0072187, 0.0051748, 0.0072370, -0.0009794, 0.0010264
9: 0.0071154, 0.0107081, 0.0070320, 0.0107410, -0.0017615, 0.0018460

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004753
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004710
time: 1.11 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041058, -0.0040755, -0.0000138, 0.0000138
1: -0.0064171, -0.0052819, -0.0064171, -0.0052819, -0.0005168, 0.0005168
2: 0.9687627, 0.9701250, 0.9687627, 0.9701250, -0.0006202, 0.0006202
3: 0.0159048, 0.0259526, 0.0159048, 0.0259526, -0.0045748, 0.0045748
4: -0.0026669, -0.0019027, -0.0026669, -0.0019027, -0.0003479, 0.0003479
5: 0.0145750, 0.0153473, 0.0145750, 0.0153473, -0.0003517, 0.0003517
6: 0.0044418, 0.0048175, 0.0044418, 0.0048175, -0.0001710, 0.0001710
7: -0.0145041, -0.0119001, -0.0145041, -0.0119001, -0.0011856, 0.0011856
8: 0.0052223, 0.0072882, 0.0052223, 0.0072882, -0.0009406, 0.0009406
9: 0.0071174, 0.0108331, 0.0071174, 0.0108331, -0.0016917, 0.0016917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004667, upper bound: 0.0004597
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004538
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041052, -0.0040751, -0.0000145, 0.0000136
1: -0.0064171, -0.0052819, -0.0063974, -0.0052693, -0.0005434, 0.0005102
2: 0.9687627, 0.9701250, 0.9687864, 0.9701401, -0.0006521, 0.0006123
3: 0.0159048, 0.0259526, 0.0160793, 0.0260644, -0.0048099, 0.0045161
4: -0.0026669, -0.0019027, -0.0026754, -0.0019159, -0.0003435, 0.0003658
5: 0.0145750, 0.0153473, 0.0145664, 0.0153339, -0.0003471, 0.0003697
6: 0.0044418, 0.0048175, 0.0044484, 0.0048217, -0.0001798, 0.0001688
7: -0.0145041, -0.0119001, -0.0145331, -0.0119453, -0.0011704, 0.0012465
8: 0.0052223, 0.0072882, 0.0051993, 0.0072523, -0.0009285, 0.0009889
9: 0.0071174, 0.0108331, 0.0070760, 0.0107685, -0.0016700, 0.0017787

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004667, upper bound: 0.0004597
time: 1.09 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004539
time: 1.09 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041058, -0.0040755, -0.0000136, 0.0000145
1: -0.0063974, -0.0052693, -0.0064171, -0.0052819, -0.0005102, 0.0005434
2: 0.9687864, 0.9701401, 0.9687627, 0.9701250, -0.0006123, 0.0006521
3: 0.0160793, 0.0260644, 0.0159048, 0.0259526, -0.0045161, 0.0048099
4: -0.0026754, -0.0019159, -0.0026669, -0.0019027, -0.0003658, 0.0003435
5: 0.0145664, 0.0153339, 0.0145750, 0.0153473, -0.0003697, 0.0003471
6: 0.0044484, 0.0048217, 0.0044418, 0.0048175, -0.0001688, 0.0001798
7: -0.0145331, -0.0119453, -0.0145041, -0.0119001, -0.0012465, 0.0011704
8: 0.0051993, 0.0072523, 0.0052223, 0.0072882, -0.0009889, 0.0009285
9: 0.0070760, 0.0107685, 0.0071174, 0.0108331, -0.0017787, 0.0016700

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004588
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004538
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041052, -0.0040751, -0.0000138, 0.0000138
1: -0.0063974, -0.0052693, -0.0063974, -0.0052693, -0.0005157, 0.0005157
2: 0.9687864, 0.9701401, 0.9687864, 0.9701401, -0.0006189, 0.0006189
3: 0.0160793, 0.0260644, 0.0160793, 0.0260644, -0.0045650, 0.0045650
4: -0.0026754, -0.0019159, -0.0026754, -0.0019159, -0.0003472, 0.0003472
5: 0.0145664, 0.0153339, 0.0145664, 0.0153339, -0.0003509, 0.0003509
6: 0.0044484, 0.0048217, 0.0044484, 0.0048217, -0.0001707, 0.0001707
7: -0.0145331, -0.0119453, -0.0145331, -0.0119453, -0.0011831, 0.0011831
8: 0.0051993, 0.0072523, 0.0051993, 0.0072523, -0.0009386, 0.0009386
9: 0.0070760, 0.0107685, 0.0070760, 0.0107685, -0.0016881, 0.0016881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004630
time: 0.99 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004598
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041056, -0.0040751, -0.0000146, 0.0000140
1: -0.0064171, -0.0052819, -0.0064096, -0.0052675, -0.0005479, 0.0005257
2: 0.9687627, 0.9701250, 0.9687717, 0.9701422, -0.0006575, 0.0006308
3: 0.0159048, 0.0259526, 0.0159711, 0.0260805, -0.0048497, 0.0046528
4: -0.0026669, -0.0019027, -0.0026766, -0.0019077, -0.0003539, 0.0003688
5: 0.0145750, 0.0153473, 0.0145652, 0.0153423, -0.0003576, 0.0003728
6: 0.0044418, 0.0048175, 0.0044443, 0.0048223, -0.0001813, 0.0001740
7: -0.0145041, -0.0119001, -0.0145372, -0.0119173, -0.0012058, 0.0012568
8: 0.0052223, 0.0072882, 0.0051960, 0.0072745, -0.0009566, 0.0009971
9: 0.0071174, 0.0108331, 0.0070701, 0.0108085, -0.0017206, 0.0017934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004559
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
time: 0.98 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041058, -0.0040755, -0.0041050, -0.0040748, -0.0000151, 0.0000136
1: -0.0064171, -0.0052819, -0.0063890, -0.0052558, -0.0005661, 0.0005110
2: 0.9687627, 0.9701250, 0.9687964, 0.9701563, -0.0006793, 0.0006132
3: 0.0159048, 0.0259526, 0.0161538, 0.0261836, -0.0050103, 0.0045227
4: -0.0026669, -0.0019027, -0.0026844, -0.0019216, -0.0003440, 0.0003811
5: 0.0145750, 0.0153473, 0.0145572, 0.0153282, -0.0003476, 0.0003851
6: 0.0044418, 0.0048175, 0.0044511, 0.0048261, -0.0001873, 0.0001691
7: -0.0145041, -0.0119001, -0.0145640, -0.0119646, -0.0011721, 0.0012985
8: 0.0052223, 0.0072882, 0.0051748, 0.0072370, -0.0009299, 0.0010301
9: 0.0071174, 0.0108331, 0.0070320, 0.0107410, -0.0016725, 0.0018528

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004560
time: 0.98 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004480
time: 1.15 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041056, -0.0040751, -0.0000145, 0.0000147
1: -0.0063974, -0.0052693, -0.0064096, -0.0052675, -0.0005413, 0.0005522
2: 0.9687864, 0.9701401, 0.9687717, 0.9701422, -0.0006496, 0.0006627
3: 0.0160793, 0.0260644, 0.0159711, 0.0260805, -0.0047910, 0.0048879
4: -0.0026754, -0.0019159, -0.0026766, -0.0019077, -0.0003718, 0.0003644
5: 0.0145664, 0.0153339, 0.0145652, 0.0153423, -0.0003757, 0.0003683
6: 0.0044484, 0.0048217, 0.0044443, 0.0048223, -0.0001791, 0.0001827
7: -0.0145331, -0.0119453, -0.0145372, -0.0119173, -0.0012667, 0.0012416
8: 0.0051993, 0.0072523, 0.0051960, 0.0072745, -0.0010050, 0.0009850
9: 0.0070760, 0.0107685, 0.0070701, 0.0108085, -0.0018075, 0.0017717

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004546
time: 0.77 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004472
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040751, -0.0041050, -0.0040748, -0.0000146, 0.0000140
1: -0.0063974, -0.0052693, -0.0063890, -0.0052558, -0.0005477, 0.0005243
2: 0.9687864, 0.9701401, 0.9687964, 0.9701563, -0.0006573, 0.0006292
3: 0.0160793, 0.0260644, 0.0161538, 0.0261836, -0.0048479, 0.0046412
4: -0.0026754, -0.0019159, -0.0026844, -0.0019216, -0.0003530, 0.0003687
5: 0.0145664, 0.0153339, 0.0145572, 0.0153282, -0.0003568, 0.0003726
6: 0.0044484, 0.0048217, 0.0044511, 0.0048261, -0.0001813, 0.0001735
7: -0.0145331, -0.0119453, -0.0145640, -0.0119646, -0.0012028, 0.0012564
8: 0.0051993, 0.0072523, 0.0051748, 0.0072370, -0.0009542, 0.0009968
9: 0.0070760, 0.0107685, 0.0070320, 0.0107410, -0.0017163, 0.0017928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004614
time: 0.97 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004568
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041052, -0.0040758, -0.0000136, 0.0000142
1: -0.0063885, -0.0052811, -0.0063975, -0.0052939, -0.0005094, 0.0005315
2: 0.9687970, 0.9701260, 0.9687861, 0.9701107, -0.0006112, 0.0006378
3: 0.0161576, 0.0259601, 0.0160782, 0.0258468, -0.0045085, 0.0047043
4: -0.0026674, -0.0019219, -0.0026588, -0.0019159, -0.0003578, 0.0003429
5: 0.0145744, 0.0153279, 0.0145831, 0.0153340, -0.0003616, 0.0003466
6: 0.0044513, 0.0048178, 0.0044483, 0.0048136, -0.0001686, 0.0001759
7: -0.0145060, -0.0119657, -0.0144767, -0.0119451, -0.0012192, 0.0011684
8: 0.0052207, 0.0072362, 0.0052440, 0.0072525, -0.0009672, 0.0009270
9: 0.0071146, 0.0107396, 0.0071565, 0.0107689, -0.0017396, 0.0016672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004610
time: 0.93 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004522
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041048, -0.0040754, -0.0000143, 0.0000140
1: -0.0063885, -0.0052811, -0.0063789, -0.0052813, -0.0005363, 0.0005251
2: 0.9687970, 0.9701260, 0.9688084, 0.9701257, -0.0006436, 0.0006302
3: 0.0161576, 0.0259601, 0.0162426, 0.0259579, -0.0047468, 0.0046481
4: -0.0026674, -0.0019219, -0.0026673, -0.0019284, -0.0003535, 0.0003610
5: 0.0145744, 0.0153279, 0.0145746, 0.0153214, -0.0003573, 0.0003649
6: 0.0044513, 0.0048178, 0.0044545, 0.0048177, -0.0001775, 0.0001738
7: -0.0145060, -0.0119657, -0.0145055, -0.0119877, -0.0012046, 0.0012302
8: 0.0052207, 0.0072362, 0.0052212, 0.0072187, -0.0009557, 0.0009760
9: 0.0071146, 0.0107396, 0.0071154, 0.0107081, -0.0017189, 0.0017553

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004621
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004535
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041052, -0.0040758, -0.0000132, 0.0000147
1: -0.0063681, -0.0052694, -0.0063975, -0.0052939, -0.0004950, 0.0005502
2: 0.9688215, 0.9701400, 0.9687861, 0.9701107, -0.0005940, 0.0006602
3: 0.0163382, 0.0260635, 0.0160782, 0.0258468, -0.0043812, 0.0048696
4: -0.0026753, -0.0019356, -0.0026588, -0.0019159, -0.0003704, 0.0003332
5: 0.0145665, 0.0153140, 0.0145831, 0.0153340, -0.0003743, 0.0003368
6: 0.0044580, 0.0048217, 0.0044483, 0.0048136, -0.0001638, 0.0001821
7: -0.0145328, -0.0120124, -0.0144767, -0.0119451, -0.0012620, 0.0011354
8: 0.0051995, 0.0071990, 0.0052440, 0.0072525, -0.0010012, 0.0009008
9: 0.0070764, 0.0106728, 0.0071565, 0.0107689, -0.0018008, 0.0016202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004601
time: 0.83 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004538
time: 0.79 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041048, -0.0040754, -0.0000136, 0.0000142
1: -0.0063681, -0.0052694, -0.0063789, -0.0052813, -0.0005080, 0.0005317
2: 0.9688215, 0.9701400, 0.9688084, 0.9701257, -0.0006096, 0.0006380
3: 0.0163382, 0.0260635, 0.0162426, 0.0259579, -0.0044961, 0.0047059
4: -0.0026753, -0.0019356, -0.0026673, -0.0019284, -0.0003579, 0.0003420
5: 0.0145665, 0.0153140, 0.0145746, 0.0153214, -0.0003617, 0.0003456
6: 0.0044580, 0.0048217, 0.0044545, 0.0048177, -0.0001681, 0.0001759
7: -0.0145328, -0.0120124, -0.0145055, -0.0119877, -0.0012196, 0.0011652
8: 0.0051995, 0.0071990, 0.0052212, 0.0072187, -0.0009675, 0.0009244
9: 0.0070764, 0.0106728, 0.0071154, 0.0107081, -0.0017402, 0.0016626

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004681
time: 1.16 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004650
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041050, -0.0040754, -0.0000134, 0.0000134
1: -0.0063885, -0.0052811, -0.0063885, -0.0052811, -0.0005024, 0.0005024
2: 0.9687970, 0.9701260, 0.9687970, 0.9701260, -0.0006029, 0.0006029
3: 0.0161576, 0.0259601, 0.0161576, 0.0259601, -0.0044470, 0.0044470
4: -0.0026674, -0.0019219, -0.0026674, -0.0019219, -0.0003382, 0.0003382
5: 0.0145744, 0.0153279, 0.0145744, 0.0153279, -0.0003418, 0.0003418
6: 0.0044513, 0.0048178, 0.0044513, 0.0048178, -0.0001663, 0.0001663
7: -0.0145060, -0.0119657, -0.0145060, -0.0119657, -0.0011525, 0.0011525
8: 0.0052207, 0.0072362, 0.0052207, 0.0072362, -0.0009143, 0.0009143
9: 0.0071146, 0.0107396, 0.0071146, 0.0107396, -0.0016445, 0.0016445

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004592
time: 0.87 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004484
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041045, -0.0040751, -0.0000142, 0.0000132
1: -0.0063885, -0.0052811, -0.0063681, -0.0052694, -0.0005309, 0.0004961
2: 0.9687970, 0.9701260, 0.9688215, 0.9701400, -0.0006371, 0.0005953
3: 0.0161576, 0.0259601, 0.0163382, 0.0260635, -0.0046990, 0.0043911
4: -0.0026674, -0.0019219, -0.0026753, -0.0019356, -0.0003340, 0.0003574
5: 0.0145744, 0.0153279, 0.0145665, 0.0153140, -0.0003375, 0.0003612
6: 0.0044513, 0.0048178, 0.0044580, 0.0048217, -0.0001757, 0.0001642
7: -0.0145060, -0.0119657, -0.0145328, -0.0120124, -0.0011380, 0.0012178
8: 0.0052207, 0.0072362, 0.0051995, 0.0071990, -0.0009028, 0.0009661
9: 0.0071146, 0.0107396, 0.0070764, 0.0106728, -0.0016238, 0.0017377

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004608
time: 0.90 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004505
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041050, -0.0040754, -0.0000132, 0.0000142
1: -0.0063681, -0.0052694, -0.0063885, -0.0052811, -0.0004961, 0.0005309
2: 0.9688215, 0.9701400, 0.9687970, 0.9701260, -0.0005953, 0.0006371
3: 0.0163382, 0.0260635, 0.0161576, 0.0259601, -0.0043911, 0.0046990
4: -0.0026753, -0.0019356, -0.0026674, -0.0019219, -0.0003574, 0.0003340
5: 0.0145665, 0.0153140, 0.0145744, 0.0153279, -0.0003612, 0.0003375
6: 0.0044580, 0.0048217, 0.0044513, 0.0048178, -0.0001642, 0.0001757
7: -0.0145328, -0.0120124, -0.0145060, -0.0119657, -0.0012178, 0.0011380
8: 0.0051995, 0.0071990, 0.0052207, 0.0072362, -0.0009661, 0.0009028
9: 0.0070764, 0.0106728, 0.0071146, 0.0107396, -0.0017377, 0.0016238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004580
time: 0.98 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004491
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041045, -0.0040751, -0.0000134, 0.0000134
1: -0.0063681, -0.0052694, -0.0063681, -0.0052694, -0.0005002, 0.0005002
2: 0.9688215, 0.9701400, 0.9688215, 0.9701400, -0.0006002, 0.0006002
3: 0.0163382, 0.0260635, 0.0163382, 0.0260635, -0.0044272, 0.0044272
4: -0.0026753, -0.0019356, -0.0026753, -0.0019356, -0.0003367, 0.0003367
5: 0.0145665, 0.0153140, 0.0145665, 0.0153140, -0.0003403, 0.0003403
6: 0.0044580, 0.0048217, 0.0044580, 0.0048217, -0.0001655, 0.0001655
7: -0.0145328, -0.0120124, -0.0145328, -0.0120124, -0.0011473, 0.0011473
8: 0.0051995, 0.0071990, 0.0051995, 0.0071990, -0.0009102, 0.0009102
9: 0.0070764, 0.0106728, 0.0070764, 0.0106728, -0.0016372, 0.0016372

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004665
time: 1.14 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004620
time: 1.09 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041052, -0.0040758, -0.0000144, 0.0000151
1: -0.0064096, -0.0052675, -0.0063975, -0.0052939, -0.0005394, 0.0005637
2: 0.9687717, 0.9701422, 0.9687861, 0.9701107, -0.0006473, 0.0006764
3: 0.0159711, 0.0260805, 0.0160782, 0.0258468, -0.0047745, 0.0049892
4: -0.0026766, -0.0019077, -0.0026588, -0.0019159, -0.0003795, 0.0003631
5: 0.0145652, 0.0153423, 0.0145831, 0.0153340, -0.0003835, 0.0003670
6: 0.0044443, 0.0048223, 0.0044483, 0.0048136, -0.0001785, 0.0001865
7: -0.0145372, -0.0119173, -0.0144767, -0.0119451, -0.0012930, 0.0012373
8: 0.0051960, 0.0072745, 0.0052440, 0.0072525, -0.0010258, 0.0009817
9: 0.0070701, 0.0108085, 0.0071565, 0.0107689, -0.0018450, 0.0017656

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004562
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004489
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041048, -0.0040754, -0.0000151, 0.0000149
1: -0.0064096, -0.0052675, -0.0063789, -0.0052813, -0.0005663, 0.0005573
2: 0.9687717, 0.9701422, 0.9688084, 0.9701257, -0.0006796, 0.0006688
3: 0.0159711, 0.0260805, 0.0162426, 0.0259579, -0.0050128, 0.0049330
4: -0.0026766, -0.0019077, -0.0026673, -0.0019284, -0.0003752, 0.0003813
5: 0.0145652, 0.0153423, 0.0145746, 0.0153214, -0.0003792, 0.0003853
6: 0.0044443, 0.0048223, 0.0044545, 0.0048177, -0.0001874, 0.0001844
7: -0.0145372, -0.0119173, -0.0145055, -0.0119877, -0.0012784, 0.0012991
8: 0.0051960, 0.0072745, 0.0052212, 0.0072187, -0.0010143, 0.0010306
9: 0.0070701, 0.0108085, 0.0071154, 0.0107081, -0.0018242, 0.0018537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004563
time: 0.94 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004494
time: 1.01 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041052, -0.0040758, -0.0000140, 0.0000155
1: -0.0063890, -0.0052558, -0.0063975, -0.0052939, -0.0005250, 0.0005804
2: 0.9687964, 0.9701563, 0.9687861, 0.9701107, -0.0006300, 0.0006965
3: 0.0161538, 0.0261836, 0.0160782, 0.0258468, -0.0046467, 0.0051374
4: -0.0026844, -0.0019216, -0.0026588, -0.0019159, -0.0003907, 0.0003534
5: 0.0145572, 0.0153282, 0.0145831, 0.0153340, -0.0003949, 0.0003572
6: 0.0044511, 0.0048261, 0.0044483, 0.0048136, -0.0001737, 0.0001921
7: -0.0145640, -0.0119646, -0.0144767, -0.0119451, -0.0013314, 0.0012042
8: 0.0051748, 0.0072370, 0.0052440, 0.0072525, -0.0010563, 0.0009554
9: 0.0070320, 0.0107410, 0.0071565, 0.0107689, -0.0018998, 0.0017184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004557
time: 0.81 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004498
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041048, -0.0040754, -0.0000144, 0.0000151
1: -0.0063890, -0.0052558, -0.0063789, -0.0052813, -0.0005382, 0.0005640
2: 0.9687964, 0.9701563, 0.9688084, 0.9701257, -0.0006458, 0.0006768
3: 0.0161538, 0.0261836, 0.0162426, 0.0259579, -0.0047634, 0.0049920
4: -0.0026844, -0.0019216, -0.0026673, -0.0019284, -0.0003797, 0.0003623
5: 0.0145572, 0.0153282, 0.0145746, 0.0153214, -0.0003837, 0.0003662
6: 0.0044511, 0.0048261, 0.0044545, 0.0048177, -0.0001781, 0.0001866
7: -0.0145640, -0.0119646, -0.0145055, -0.0119877, -0.0012937, 0.0012345
8: 0.0051748, 0.0072370, 0.0052212, 0.0072187, -0.0010264, 0.0009794
9: 0.0070320, 0.0107410, 0.0071154, 0.0107081, -0.0018460, 0.0017615

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004624
time: 1.08 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004589
time: 0.87 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041050, -0.0040754, -0.0000143, 0.0000144
1: -0.0064096, -0.0052675, -0.0063885, -0.0052811, -0.0005367, 0.0005409
2: 0.9687717, 0.9701422, 0.9687970, 0.9701260, -0.0006441, 0.0006491
3: 0.0159711, 0.0260805, 0.0161576, 0.0259601, -0.0047509, 0.0047876
4: -0.0026766, -0.0019077, -0.0026674, -0.0019219, -0.0003641, 0.0003613
5: 0.0145652, 0.0153423, 0.0145744, 0.0153279, -0.0003680, 0.0003652
6: 0.0044443, 0.0048223, 0.0044513, 0.0048178, -0.0001776, 0.0001790
7: -0.0145372, -0.0119173, -0.0145060, -0.0119657, -0.0012407, 0.0012312
8: 0.0051960, 0.0072745, 0.0052207, 0.0072362, -0.0009843, 0.0009768
9: 0.0070701, 0.0108085, 0.0071146, 0.0107396, -0.0017704, 0.0017569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004548
time: 0.84 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004451
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041045, -0.0040751, -0.0000151, 0.0000143
1: -0.0064096, -0.0052675, -0.0063681, -0.0052694, -0.0005652, 0.0005346
2: 0.9687717, 0.9701422, 0.9688215, 0.9701400, -0.0006783, 0.0006415
3: 0.0159711, 0.0260805, 0.0163382, 0.0260635, -0.0050029, 0.0047317
4: -0.0026766, -0.0019077, -0.0026753, -0.0019356, -0.0003599, 0.0003805
5: 0.0145652, 0.0153423, 0.0145665, 0.0153140, -0.0003637, 0.0003846
6: 0.0044443, 0.0048223, 0.0044580, 0.0048217, -0.0001871, 0.0001769
7: -0.0145372, -0.0119173, -0.0145328, -0.0120124, -0.0012263, 0.0012966
8: 0.0051960, 0.0072745, 0.0051995, 0.0071990, -0.0009729, 0.0010286
9: 0.0070701, 0.0108085, 0.0070764, 0.0106728, -0.0017498, 0.0018501

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004552
time: 0.80 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004469
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041050, -0.0040754, -0.0000141, 0.0000151
1: -0.0063890, -0.0052558, -0.0063885, -0.0052811, -0.0005281, 0.0005649
2: 0.9687964, 0.9701563, 0.9687970, 0.9701260, -0.0006338, 0.0006779
3: 0.0161538, 0.0261836, 0.0161576, 0.0259601, -0.0046747, 0.0049998
4: -0.0026844, -0.0019216, -0.0026674, -0.0019219, -0.0003803, 0.0003555
5: 0.0145572, 0.0153282, 0.0145744, 0.0153279, -0.0003843, 0.0003593
6: 0.0044511, 0.0048261, 0.0044513, 0.0048178, -0.0001748, 0.0001869
7: -0.0145640, -0.0119646, -0.0145060, -0.0119657, -0.0012957, 0.0012115
8: 0.0051748, 0.0072370, 0.0052207, 0.0072362, -0.0010280, 0.0009611
9: 0.0070320, 0.0107410, 0.0071146, 0.0107396, -0.0018489, 0.0017287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004538
time: 0.86 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004458
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041045, -0.0040751, -0.0000143, 0.0000144
1: -0.0063890, -0.0052558, -0.0063681, -0.0052694, -0.0005343, 0.0005386
2: 0.9687964, 0.9701563, 0.9688215, 0.9701400, -0.0006412, 0.0006463
3: 0.0161538, 0.0261836, 0.0163382, 0.0260635, -0.0047295, 0.0047670
4: -0.0026844, -0.0019216, -0.0026753, -0.0019356, -0.0003626, 0.0003597
5: 0.0145572, 0.0153282, 0.0145665, 0.0153140, -0.0003664, 0.0003635
6: 0.0044511, 0.0048261, 0.0044580, 0.0048217, -0.0001768, 0.0001782
7: -0.0145640, -0.0119646, -0.0145328, -0.0120124, -0.0012354, 0.0012257
8: 0.0051748, 0.0072370, 0.0051995, 0.0071990, -0.0009801, 0.0009724
9: 0.0070320, 0.0107410, 0.0070764, 0.0106728, -0.0017628, 0.0017490

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004611
time: 1.03 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004566
time: 1.15 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041058, -0.0040755, -0.0000146, 0.0000151
1: -0.0063885, -0.0052811, -0.0064171, -0.0052819, -0.0005470, 0.0005657
2: 0.9687970, 0.9701260, 0.9687627, 0.9701250, -0.0006564, 0.0006789
3: 0.0161576, 0.0259601, 0.0159048, 0.0259526, -0.0048416, 0.0050073
4: -0.0026674, -0.0019219, -0.0026669, -0.0019027, -0.0003808, 0.0003682
5: 0.0145744, 0.0153279, 0.0145750, 0.0153473, -0.0003849, 0.0003722
6: 0.0044513, 0.0048178, 0.0044418, 0.0048175, -0.0001810, 0.0001872
7: -0.0145060, -0.0119657, -0.0145041, -0.0119001, -0.0012977, 0.0012547
8: 0.0052207, 0.0072362, 0.0052223, 0.0072882, -0.0010295, 0.0009955
9: 0.0071146, 0.0107396, 0.0071174, 0.0108331, -0.0018517, 0.0017904

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004726
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004627
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041052, -0.0040751, -0.0000152, 0.0000149
1: -0.0063885, -0.0052811, -0.0063974, -0.0052693, -0.0005703, 0.0005571
2: 0.9687970, 0.9701260, 0.9687864, 0.9701401, -0.0006844, 0.0006686
3: 0.0161576, 0.0259601, 0.0160793, 0.0260644, -0.0050478, 0.0049315
4: -0.0026674, -0.0019219, -0.0026754, -0.0019159, -0.0003751, 0.0003839
5: 0.0145744, 0.0153279, 0.0145664, 0.0153339, -0.0003791, 0.0003880
6: 0.0044513, 0.0048178, 0.0044484, 0.0048217, -0.0001887, 0.0001844
7: -0.0145060, -0.0119657, -0.0145331, -0.0119453, -0.0012780, 0.0013082
8: 0.0052207, 0.0072362, 0.0051993, 0.0072523, -0.0010139, 0.0010379
9: 0.0071146, 0.0107396, 0.0070760, 0.0107685, -0.0018236, 0.0018667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004728
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004641
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041058, -0.0040755, -0.0000142, 0.0000156
1: -0.0063681, -0.0052694, -0.0064171, -0.0052819, -0.0005326, 0.0005844
2: 0.9688215, 0.9701400, 0.9687627, 0.9701250, -0.0006392, 0.0007013
3: 0.0163382, 0.0260635, 0.0159048, 0.0259526, -0.0047144, 0.0051727
4: -0.0026753, -0.0019356, -0.0026669, -0.0019027, -0.0003934, 0.0003586
5: 0.0145665, 0.0153140, 0.0145750, 0.0153473, -0.0003976, 0.0003624
6: 0.0044580, 0.0048217, 0.0044418, 0.0048175, -0.0001763, 0.0001934
7: -0.0145328, -0.0120124, -0.0145041, -0.0119001, -0.0013405, 0.0012218
8: 0.0051995, 0.0071990, 0.0052223, 0.0072882, -0.0010635, 0.0009693
9: 0.0070764, 0.0106728, 0.0071174, 0.0108331, -0.0019128, 0.0017434

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004703
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004633
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041052, -0.0040751, -0.0000146, 0.0000151
1: -0.0063681, -0.0052694, -0.0063974, -0.0052693, -0.0005460, 0.0005657
2: 0.9688215, 0.9701400, 0.9687864, 0.9701401, -0.0006552, 0.0006789
3: 0.0163382, 0.0260635, 0.0160793, 0.0260644, -0.0048330, 0.0050072
4: -0.0026753, -0.0019356, -0.0026754, -0.0019159, -0.0003808, 0.0003676
5: 0.0145665, 0.0153140, 0.0145664, 0.0153339, -0.0003849, 0.0003715
6: 0.0044580, 0.0048217, 0.0044484, 0.0048217, -0.0001807, 0.0001872
7: -0.0145328, -0.0120124, -0.0145331, -0.0119453, -0.0012977, 0.0012525
8: 0.0051995, 0.0071990, 0.0051993, 0.0072523, -0.0010295, 0.0009937
9: 0.0070764, 0.0106728, 0.0070760, 0.0107685, -0.0018516, 0.0017872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004771
time: 1.04 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004733
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041056, -0.0040751, -0.0000144, 0.0000143
1: -0.0063885, -0.0052811, -0.0064096, -0.0052675, -0.0005409, 0.0005367
2: 0.9687970, 0.9701260, 0.9687717, 0.9701422, -0.0006491, 0.0006441
3: 0.0161576, 0.0259601, 0.0159711, 0.0260805, -0.0047876, 0.0047509
4: -0.0026674, -0.0019219, -0.0026766, -0.0019077, -0.0003613, 0.0003641
5: 0.0145744, 0.0153279, 0.0145652, 0.0153423, -0.0003652, 0.0003680
6: 0.0044513, 0.0048178, 0.0044443, 0.0048223, -0.0001790, 0.0001776
7: -0.0145060, -0.0119657, -0.0145372, -0.0119173, -0.0012312, 0.0012407
8: 0.0052207, 0.0072362, 0.0051960, 0.0072745, -0.0009768, 0.0009843
9: 0.0071146, 0.0107396, 0.0070701, 0.0108085, -0.0017569, 0.0017704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004715
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004590
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040754, -0.0041050, -0.0040748, -0.0000151, 0.0000141
1: -0.0063885, -0.0052811, -0.0063890, -0.0052558, -0.0005649, 0.0005281
2: 0.9687970, 0.9701260, 0.9687964, 0.9701563, -0.0006779, 0.0006338
3: 0.0161576, 0.0259601, 0.0161538, 0.0261836, -0.0049998, 0.0046747
4: -0.0026674, -0.0019219, -0.0026844, -0.0019216, -0.0003555, 0.0003803
5: 0.0145744, 0.0153279, 0.0145572, 0.0153282, -0.0003593, 0.0003843
6: 0.0044513, 0.0048178, 0.0044511, 0.0048261, -0.0001869, 0.0001748
7: -0.0145060, -0.0119657, -0.0145640, -0.0119646, -0.0012115, 0.0012957
8: 0.0052207, 0.0072362, 0.0051748, 0.0072370, -0.0009611, 0.0010280
9: 0.0071146, 0.0107396, 0.0070320, 0.0107410, -0.0017287, 0.0018489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004717
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004621
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041056, -0.0040751, -0.0000143, 0.0000151
1: -0.0063681, -0.0052694, -0.0064096, -0.0052675, -0.0005346, 0.0005652
2: 0.9688215, 0.9701400, 0.9687717, 0.9701422, -0.0006415, 0.0006783
3: 0.0163382, 0.0260635, 0.0159711, 0.0260805, -0.0047317, 0.0050029
4: -0.0026753, -0.0019356, -0.0026766, -0.0019077, -0.0003805, 0.0003599
5: 0.0145665, 0.0153140, 0.0145652, 0.0153423, -0.0003846, 0.0003637
6: 0.0044580, 0.0048217, 0.0044443, 0.0048223, -0.0001769, 0.0001871
7: -0.0145328, -0.0120124, -0.0145372, -0.0119173, -0.0012966, 0.0012263
8: 0.0051995, 0.0071990, 0.0051960, 0.0072745, -0.0010286, 0.0009729
9: 0.0070764, 0.0106728, 0.0070701, 0.0108085, -0.0018501, 0.0017498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004682
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004591
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041045, -0.0040751, -0.0041050, -0.0040748, -0.0000144, 0.0000143
1: -0.0063681, -0.0052694, -0.0063890, -0.0052558, -0.0005386, 0.0005343
2: 0.9688215, 0.9701400, 0.9687964, 0.9701563, -0.0006463, 0.0006412
3: 0.0163382, 0.0260635, 0.0161538, 0.0261836, -0.0047670, 0.0047295
4: -0.0026753, -0.0019356, -0.0026844, -0.0019216, -0.0003597, 0.0003626
5: 0.0145665, 0.0153140, 0.0145572, 0.0153282, -0.0003635, 0.0003664
6: 0.0044580, 0.0048217, 0.0044511, 0.0048261, -0.0001782, 0.0001768
7: -0.0145328, -0.0120124, -0.0145640, -0.0119646, -0.0012257, 0.0012354
8: 0.0051995, 0.0071990, 0.0051748, 0.0072370, -0.0009724, 0.0009801
9: 0.0070764, 0.0106728, 0.0070320, 0.0107410, -0.0017490, 0.0017628

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004753
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004710
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041058, -0.0040755, -0.0000140, 0.0000146
1: -0.0064096, -0.0052675, -0.0064171, -0.0052819, -0.0005257, 0.0005479
2: 0.9687717, 0.9701422, 0.9687627, 0.9701250, -0.0006308, 0.0006575
3: 0.0159711, 0.0260805, 0.0159048, 0.0259526, -0.0046528, 0.0048497
4: -0.0026766, -0.0019077, -0.0026669, -0.0019027, -0.0003688, 0.0003539
5: 0.0145652, 0.0153423, 0.0145750, 0.0153473, -0.0003728, 0.0003576
6: 0.0044443, 0.0048223, 0.0044418, 0.0048175, -0.0001740, 0.0001813
7: -0.0145372, -0.0119173, -0.0145041, -0.0119001, -0.0012568, 0.0012058
8: 0.0051960, 0.0072745, 0.0052223, 0.0072882, -0.0009971, 0.0009566
9: 0.0070701, 0.0108085, 0.0071174, 0.0108331, -0.0017934, 0.0017206

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004570
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004502
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041052, -0.0040751, -0.0000147, 0.0000145
1: -0.0064096, -0.0052675, -0.0063974, -0.0052693, -0.0005522, 0.0005413
2: 0.9687717, 0.9701422, 0.9687864, 0.9701401, -0.0006627, 0.0006496
3: 0.0159711, 0.0260805, 0.0160793, 0.0260644, -0.0048879, 0.0047910
4: -0.0026766, -0.0019077, -0.0026754, -0.0019159, -0.0003644, 0.0003718
5: 0.0145652, 0.0153423, 0.0145664, 0.0153339, -0.0003683, 0.0003757
6: 0.0044443, 0.0048223, 0.0044484, 0.0048217, -0.0001827, 0.0001791
7: -0.0145372, -0.0119173, -0.0145331, -0.0119453, -0.0012416, 0.0012667
8: 0.0051960, 0.0072745, 0.0051993, 0.0072523, -0.0009850, 0.0010050
9: 0.0070701, 0.0108085, 0.0070760, 0.0107685, -0.0017717, 0.0018075

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004570
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004507
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041058, -0.0040755, -0.0000136, 0.0000151
1: -0.0063890, -0.0052558, -0.0064171, -0.0052819, -0.0005110, 0.0005661
2: 0.9687964, 0.9701563, 0.9687627, 0.9701250, -0.0006132, 0.0006793
3: 0.0161538, 0.0261836, 0.0159048, 0.0259526, -0.0045227, 0.0050103
4: -0.0026844, -0.0019216, -0.0026669, -0.0019027, -0.0003811, 0.0003440
5: 0.0145572, 0.0153282, 0.0145750, 0.0153473, -0.0003851, 0.0003476
6: 0.0044511, 0.0048261, 0.0044418, 0.0048175, -0.0001691, 0.0001873
7: -0.0145640, -0.0119646, -0.0145041, -0.0119001, -0.0012985, 0.0011721
8: 0.0051748, 0.0072370, 0.0052223, 0.0072882, -0.0010301, 0.0009299
9: 0.0070320, 0.0107410, 0.0071174, 0.0108331, -0.0018528, 0.0016725

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004563
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004508
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041052, -0.0040751, -0.0000140, 0.0000146
1: -0.0063890, -0.0052558, -0.0063974, -0.0052693, -0.0005243, 0.0005477
2: 0.9687964, 0.9701563, 0.9687864, 0.9701401, -0.0006292, 0.0006573
3: 0.0161538, 0.0261836, 0.0160793, 0.0260644, -0.0046412, 0.0048479
4: -0.0026844, -0.0019216, -0.0026754, -0.0019159, -0.0003687, 0.0003530
5: 0.0145572, 0.0153282, 0.0145664, 0.0153339, -0.0003726, 0.0003568
6: 0.0044511, 0.0048261, 0.0044484, 0.0048217, -0.0001735, 0.0001813
7: -0.0145640, -0.0119646, -0.0145331, -0.0119453, -0.0012564, 0.0012028
8: 0.0051748, 0.0072370, 0.0051993, 0.0072523, -0.0009968, 0.0009542
9: 0.0070320, 0.0107410, 0.0070760, 0.0107685, -0.0017928, 0.0017163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004626
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004592
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041056, -0.0040751, -0.0000139, 0.0000139
1: -0.0064096, -0.0052675, -0.0064096, -0.0052675, -0.0005192, 0.0005192
2: 0.9687717, 0.9701422, 0.9687717, 0.9701422, -0.0006231, 0.0006231
3: 0.0159711, 0.0260805, 0.0159711, 0.0260805, -0.0045960, 0.0045960
4: -0.0026766, -0.0019077, -0.0026766, -0.0019077, -0.0003496, 0.0003496
5: 0.0145652, 0.0153423, 0.0145652, 0.0153423, -0.0003533, 0.0003533
6: 0.0044443, 0.0048223, 0.0044443, 0.0048223, -0.0001718, 0.0001718
7: -0.0145372, -0.0119173, -0.0145372, -0.0119173, -0.0011911, 0.0011911
8: 0.0051960, 0.0072745, 0.0051960, 0.0072745, -0.0009450, 0.0009450
9: 0.0070701, 0.0108085, 0.0070701, 0.0108085, -0.0016996, 0.0016996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004556
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004468
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0041056, -0.0040751, -0.0041050, -0.0040748, -0.0000146, 0.0000137
1: -0.0064096, -0.0052675, -0.0063890, -0.0052558, -0.0005469, 0.0005129
2: 0.9687717, 0.9701422, 0.9687964, 0.9701563, -0.0006563, 0.0006155
3: 0.0159711, 0.0260805, 0.0161538, 0.0261836, -0.0048405, 0.0045400
4: -0.0026766, -0.0019077, -0.0026844, -0.0019216, -0.0003453, 0.0003681
5: 0.0145652, 0.0153423, 0.0145572, 0.0153282, -0.0003490, 0.0003721
6: 0.0044443, 0.0048223, 0.0044511, 0.0048261, -0.0001810, 0.0001697
7: -0.0145372, -0.0119173, -0.0145640, -0.0119646, -0.0011766, 0.0012545
8: 0.0051960, 0.0072745, 0.0051748, 0.0072370, -0.0009334, 0.0009952
9: 0.0070701, 0.0108085, 0.0070320, 0.0107410, -0.0016789, 0.0017900

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004557
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004480
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041056, -0.0040751, -0.0000137, 0.0000146
1: -0.0063890, -0.0052558, -0.0064096, -0.0052675, -0.0005129, 0.0005469
2: 0.9687964, 0.9701563, 0.9687717, 0.9701422, -0.0006155, 0.0006563
3: 0.0161538, 0.0261836, 0.0159711, 0.0260805, -0.0045400, 0.0048405
4: -0.0026844, -0.0019216, -0.0026766, -0.0019077, -0.0003681, 0.0003453
5: 0.0145572, 0.0153282, 0.0145652, 0.0153423, -0.0003721, 0.0003490
6: 0.0044511, 0.0048261, 0.0044443, 0.0048223, -0.0001697, 0.0001810
7: -0.0145640, -0.0119646, -0.0145372, -0.0119173, -0.0012545, 0.0011766
8: 0.0051748, 0.0072370, 0.0051960, 0.0072745, -0.0009952, 0.0009334
9: 0.0070320, 0.0107410, 0.0070701, 0.0108085, -0.0017900, 0.0016789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004545
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004472
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040748, -0.0041050, -0.0040748, -0.0000138, 0.0000138
1: -0.0063890, -0.0052558, -0.0063890, -0.0052558, -0.0005165, 0.0005165
2: 0.9687964, 0.9701563, 0.9687964, 0.9701563, -0.0006199, 0.0006199
3: 0.0161538, 0.0261836, 0.0161538, 0.0261836, -0.0045720, 0.0045720
4: -0.0026844, -0.0019216, -0.0026844, -0.0019216, -0.0003477, 0.0003477
5: 0.0145572, 0.0153282, 0.0145572, 0.0153282, -0.0003514, 0.0003514
6: 0.0044511, 0.0048261, 0.0044511, 0.0048261, -0.0001709, 0.0001709
7: -0.0145640, -0.0119646, -0.0145640, -0.0119646, -0.0011849, 0.0011849
8: 0.0051748, 0.0072370, 0.0051748, 0.0072370, -0.0009400, 0.0009400
9: 0.0070320, 0.0107410, 0.0070320, 0.0107410, -0.0016907, 0.0016907

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 249
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 189
type: A, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 249

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004613
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004568
time: 0.97 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.36 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004639
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004554, upper bound: 0.0004642
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004566
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004624
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004685
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004653
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004592
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004609
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004504
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004580
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004666
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004620
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004585
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004517
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004585
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004659, upper bound: 0.0004521
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004578
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004521
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004627
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004594
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004550
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004555
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004469
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004539
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004612
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004566
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004753
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004663
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004753
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004516, upper bound: 0.0004674
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004726
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004661
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004774
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004738
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004715
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004719
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004621
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004682
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004753
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004710
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004667, upper bound: 0.0004597
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004538
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004667, upper bound: 0.0004597
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004539
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004588
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004538
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004630
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004598
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004559
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004560
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004480
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004546
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004472
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004614
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004568
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004610
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004522
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004621
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004535
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004601
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004538
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004681
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004650
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004592
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004484
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004608
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004505
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004580
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004491
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004665
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004620
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004562
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004489
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004563
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004494
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004557
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004498
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004624
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004589
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004548
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004451
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004552
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004469
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004538
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004458
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004611
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004566
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004726
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004627
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004728
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004641
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004703
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004633
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004771
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004733
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004715
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004590
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004717
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004621
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004682
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004591
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004753
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004710
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004570
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004502
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004570
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004507
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004563
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004508
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004626
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004592
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004556
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004468
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004557
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004480
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004545
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004472
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004613
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.36
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004568

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041052, -0.0040758, -0.0000133, 0.0000133
1: -0.0063961, -0.0052976, -0.0063975, -0.0052939, -0.0004994, 0.0004973
2: 0.9687880, 0.9701061, 0.9687861, 0.9701107, -0.0005994, 0.0005967
3: 0.0160907, 0.0258135, 0.0160782, 0.0258468, -0.0044208, 0.0044013
4: -0.0026563, -0.0019168, -0.0026588, -0.0019159, -0.0003347, 0.0003362
5: 0.0145857, 0.0153331, 0.0145831, 0.0153340, -0.0003383, 0.0003398
6: 0.0044488, 0.0048123, 0.0044483, 0.0048136, -0.0001653, 0.0001646
7: -0.0144680, -0.0119483, -0.0144767, -0.0119451, -0.0011406, 0.0011457
8: 0.0052509, 0.0072499, 0.0052440, 0.0072525, -0.0009049, 0.0009089
9: 0.0071688, 0.0107643, 0.0071565, 0.0107689, -0.0016276, 0.0016348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
time: 0.96 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041052, -0.0040759, -0.0000137, 0.0000131
1: -0.0064074, -0.0053039, -0.0063964, -0.0052991, -0.0005117, 0.0004898
2: 0.9687743, 0.9700986, 0.9687874, 0.9701043, -0.0006140, 0.0005877
3: 0.0159906, 0.0257579, 0.0160875, 0.0258000, -0.0045289, 0.0043351
4: -0.0026521, -0.0019092, -0.0026553, -0.0019166, -0.0003297, 0.0003445
5: 0.0145900, 0.0153408, 0.0145867, 0.0153333, -0.0003332, 0.0003481
6: 0.0044450, 0.0048102, 0.0044487, 0.0048118, -0.0001693, 0.0001621
7: -0.0144536, -0.0119224, -0.0144645, -0.0119475, -0.0011235, 0.0011737
8: 0.0052623, 0.0072705, 0.0052537, 0.0072506, -0.0008913, 0.0009312
9: 0.0071894, 0.0108013, 0.0071738, 0.0107655, -0.0016031, 0.0016748

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
time: 0.79 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041048, -0.0040754, -0.0000141, 0.0000131
1: -0.0063961, -0.0052976, -0.0063789, -0.0052813, -0.0005264, 0.0004909
2: 0.9687880, 0.9701061, 0.9688084, 0.9701257, -0.0006317, 0.0005891
3: 0.0160907, 0.0258135, 0.0162426, 0.0259579, -0.0046591, 0.0043452
4: -0.0026563, -0.0019168, -0.0026673, -0.0019284, -0.0003305, 0.0003543
5: 0.0145857, 0.0153331, 0.0145746, 0.0153214, -0.0003340, 0.0003581
6: 0.0044488, 0.0048123, 0.0044545, 0.0048177, -0.0001742, 0.0001625
7: -0.0144680, -0.0119483, -0.0145055, -0.0119877, -0.0011261, 0.0012074
8: 0.0052509, 0.0072499, 0.0052212, 0.0072187, -0.0008934, 0.0009579
9: 0.0071688, 0.0107643, 0.0071154, 0.0107081, -0.0016069, 0.0017229

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
time: 0.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041047, -0.0040756, -0.0000144, 0.0000129
1: -0.0064074, -0.0053039, -0.0063779, -0.0052864, -0.0005391, 0.0004834
2: 0.9687743, 0.9700986, 0.9688098, 0.9701197, -0.0006469, 0.0005801
3: 0.0159906, 0.0257579, 0.0162518, 0.0259131, -0.0047716, 0.0042791
4: -0.0026521, -0.0019092, -0.0026639, -0.0019291, -0.0003254, 0.0003629
5: 0.0145900, 0.0153408, 0.0145780, 0.0153207, -0.0003289, 0.0003668
6: 0.0044450, 0.0048102, 0.0044548, 0.0048160, -0.0001784, 0.0001600
7: -0.0144536, -0.0119224, -0.0144939, -0.0119900, -0.0011090, 0.0012366
8: 0.0052623, 0.0072705, 0.0052304, 0.0072168, -0.0008798, 0.0009811
9: 0.0071894, 0.0108013, 0.0071320, 0.0107047, -0.0015824, 0.0017645

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041052, -0.0040758, -0.0000132, 0.0000140
1: -0.0063776, -0.0052852, -0.0063975, -0.0052939, -0.0004932, 0.0005239
2: 0.9688101, 0.9701211, 0.9687861, 0.9701107, -0.0005918, 0.0006286
3: 0.0162545, 0.0259238, 0.0160782, 0.0258468, -0.0043651, 0.0046368
4: -0.0026647, -0.0019293, -0.0026588, -0.0019159, -0.0003527, 0.0003320
5: 0.0145772, 0.0153205, 0.0145831, 0.0153340, -0.0003564, 0.0003355
6: 0.0044549, 0.0048164, 0.0044483, 0.0048136, -0.0001632, 0.0001734
7: -0.0144966, -0.0119907, -0.0144767, -0.0119451, -0.0012017, 0.0011312
8: 0.0052282, 0.0072163, 0.0052440, 0.0072525, -0.0009533, 0.0008975
9: 0.0071280, 0.0107037, 0.0071565, 0.0107689, -0.0017147, 0.0016142

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041052, -0.0040759, -0.0000135, 0.0000138
1: -0.0063897, -0.0052909, -0.0063964, -0.0052991, -0.0005064, 0.0005175
2: 0.9687956, 0.9701142, 0.9687874, 0.9701043, -0.0006077, 0.0006210
3: 0.0161476, 0.0258727, 0.0160875, 0.0258000, -0.0044823, 0.0045803
4: -0.0026608, -0.0019211, -0.0026553, -0.0019166, -0.0003484, 0.0003409
5: 0.0145811, 0.0153287, 0.0145867, 0.0153333, -0.0003521, 0.0003445
6: 0.0044509, 0.0048145, 0.0044487, 0.0048118, -0.0001676, 0.0001712
7: -0.0144834, -0.0119630, -0.0144645, -0.0119475, -0.0011870, 0.0011616
8: 0.0052387, 0.0072382, 0.0052537, 0.0072506, -0.0009417, 0.0009216
9: 0.0071469, 0.0107433, 0.0071738, 0.0107655, -0.0016938, 0.0016575

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
time: 0.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
time: 0.77 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041048, -0.0040754, -0.0000133, 0.0000133
1: -0.0063776, -0.0052852, -0.0063789, -0.0052813, -0.0004984, 0.0004962
2: 0.9688101, 0.9701211, 0.9688084, 0.9701257, -0.0005981, 0.0005955
3: 0.0162545, 0.0259238, 0.0162426, 0.0259579, -0.0044116, 0.0043921
4: -0.0026647, -0.0019293, -0.0026673, -0.0019284, -0.0003340, 0.0003355
5: 0.0145772, 0.0153205, 0.0145746, 0.0153214, -0.0003376, 0.0003391
6: 0.0044549, 0.0048164, 0.0044545, 0.0048177, -0.0001649, 0.0001642
7: -0.0144966, -0.0119907, -0.0145055, -0.0119877, -0.0011383, 0.0011433
8: 0.0052282, 0.0072163, 0.0052212, 0.0072187, -0.0009030, 0.0009070
9: 0.0071280, 0.0107037, 0.0071154, 0.0107081, -0.0016242, 0.0016314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004649, upper bound: 0.0004646
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004649, upper bound: 0.0004653
time: 0.89 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041047, -0.0040756, -0.0000136, 0.0000131
1: -0.0063897, -0.0052909, -0.0063779, -0.0052864, -0.0005101, 0.0004890
2: 0.9687956, 0.9701142, 0.9688098, 0.9701197, -0.0006121, 0.0005868
3: 0.0161476, 0.0258727, 0.0162518, 0.0259131, -0.0045151, 0.0043281
4: -0.0026608, -0.0019211, -0.0026639, -0.0019291, -0.0003292, 0.0003434
5: 0.0145811, 0.0153287, 0.0145780, 0.0153207, -0.0003327, 0.0003471
6: 0.0044509, 0.0048145, 0.0044548, 0.0048160, -0.0001688, 0.0001618
7: -0.0144834, -0.0119630, -0.0144939, -0.0119900, -0.0011217, 0.0011701
8: 0.0052387, 0.0072382, 0.0052304, 0.0072168, -0.0008899, 0.0009283
9: 0.0071469, 0.0107433, 0.0071320, 0.0107047, -0.0016005, 0.0016697

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004657, upper bound: 0.0004646
time: 0.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004657, upper bound: 0.0004653
time: 0.81 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041050, -0.0040754, -0.0000142, 0.0000135
1: -0.0063961, -0.0052976, -0.0063885, -0.0052811, -0.0005303, 0.0005059
2: 0.9687880, 0.9701061, 0.9687970, 0.9701260, -0.0006363, 0.0006071
3: 0.0160907, 0.0258135, 0.0161576, 0.0259601, -0.0046934, 0.0044782
4: -0.0026563, -0.0019168, -0.0026674, -0.0019219, -0.0003406, 0.0003570
5: 0.0145857, 0.0153331, 0.0145744, 0.0153279, -0.0003442, 0.0003608
6: 0.0044488, 0.0048123, 0.0044513, 0.0048178, -0.0001755, 0.0001674
7: -0.0144680, -0.0119483, -0.0145060, -0.0119657, -0.0011606, 0.0012163
8: 0.0052509, 0.0072499, 0.0052207, 0.0072362, -0.0009207, 0.0009650
9: 0.0071688, 0.0107643, 0.0071146, 0.0107396, -0.0016560, 0.0017356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
time: 0.98 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041050, -0.0040756, -0.0000145, 0.0000133
1: -0.0064074, -0.0053039, -0.0063874, -0.0052865, -0.0005424, 0.0004984
2: 0.9687743, 0.9700986, 0.9687982, 0.9701194, -0.0006509, 0.0005981
3: 0.0159906, 0.0257579, 0.0161673, 0.0259117, -0.0048009, 0.0044113
4: -0.0026521, -0.0019092, -0.0026638, -0.0019226, -0.0003355, 0.0003651
5: 0.0145900, 0.0153408, 0.0145781, 0.0153272, -0.0003391, 0.0003690
6: 0.0044450, 0.0048102, 0.0044517, 0.0048160, -0.0001795, 0.0001649
7: -0.0144536, -0.0119224, -0.0144935, -0.0119682, -0.0011432, 0.0012442
8: 0.0052623, 0.0072705, 0.0052307, 0.0072342, -0.0009070, 0.0009871
9: 0.0071894, 0.0108013, 0.0071325, 0.0107360, -0.0016313, 0.0017754

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041045, -0.0040751, -0.0000147, 0.0000131
1: -0.0063961, -0.0052976, -0.0063681, -0.0052694, -0.0005489, 0.0004916
2: 0.9687880, 0.9701061, 0.9688215, 0.9701400, -0.0006587, 0.0005899
3: 0.0160907, 0.0258135, 0.0163382, 0.0260635, -0.0048588, 0.0043510
4: -0.0026563, -0.0019168, -0.0026753, -0.0019356, -0.0003309, 0.0003695
5: 0.0145857, 0.0153331, 0.0145665, 0.0153140, -0.0003345, 0.0003735
6: 0.0044488, 0.0048123, 0.0044580, 0.0048217, -0.0001817, 0.0001627
7: -0.0144680, -0.0119483, -0.0145328, -0.0120124, -0.0011276, 0.0012592
8: 0.0052509, 0.0072499, 0.0051995, 0.0071990, -0.0008946, 0.0009990
9: 0.0071688, 0.0107643, 0.0070764, 0.0106728, -0.0016090, 0.0017968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004503
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004504
time: 0.94 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041044, -0.0040753, -0.0000150, 0.0000129
1: -0.0064074, -0.0053039, -0.0063670, -0.0052748, -0.0005614, 0.0004840
2: 0.9687743, 0.9700986, 0.9688227, 0.9701335, -0.0006737, 0.0005809
3: 0.0159906, 0.0257579, 0.0163478, 0.0260155, -0.0049693, 0.0042844
4: -0.0026521, -0.0019092, -0.0026717, -0.0019364, -0.0003259, 0.0003779
5: 0.0145900, 0.0153408, 0.0145702, 0.0153133, -0.0003293, 0.0003820
6: 0.0044450, 0.0048102, 0.0044584, 0.0048199, -0.0001858, 0.0001602
7: -0.0144536, -0.0119224, -0.0145204, -0.0120149, -0.0011103, 0.0012878
8: 0.0052623, 0.0072705, 0.0052093, 0.0071971, -0.0008809, 0.0010217
9: 0.0071894, 0.0108013, 0.0070941, 0.0106692, -0.0015844, 0.0018376

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004503
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004504
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041050, -0.0040754, -0.0000140, 0.0000142
1: -0.0063776, -0.0052852, -0.0063885, -0.0052811, -0.0005240, 0.0005325
2: 0.9688101, 0.9701211, 0.9687970, 0.9701260, -0.0006288, 0.0006391
3: 0.0162545, 0.0259238, 0.0161576, 0.0259601, -0.0046377, 0.0047137
4: -0.0026647, -0.0019293, -0.0026674, -0.0019219, -0.0003585, 0.0003527
5: 0.0145772, 0.0153205, 0.0145744, 0.0153279, -0.0003623, 0.0003565
6: 0.0044549, 0.0048164, 0.0044513, 0.0048178, -0.0001734, 0.0001762
7: -0.0144966, -0.0119907, -0.0145060, -0.0119657, -0.0012216, 0.0012019
8: 0.0052282, 0.0072163, 0.0052207, 0.0072362, -0.0009692, 0.0009535
9: 0.0071280, 0.0107037, 0.0071146, 0.0107396, -0.0017431, 0.0017150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041050, -0.0040756, -0.0000143, 0.0000140
1: -0.0063897, -0.0052909, -0.0063874, -0.0052865, -0.0005371, 0.0005261
2: 0.9687956, 0.9701142, 0.9687982, 0.9701194, -0.0006446, 0.0006313
3: 0.0161476, 0.0258727, 0.0161673, 0.0259117, -0.0047543, 0.0046564
4: -0.0026608, -0.0019211, -0.0026638, -0.0019226, -0.0003541, 0.0003616
5: 0.0145811, 0.0153287, 0.0145781, 0.0153272, -0.0003579, 0.0003655
6: 0.0044509, 0.0048145, 0.0044517, 0.0048160, -0.0001778, 0.0001741
7: -0.0144834, -0.0119630, -0.0144935, -0.0119682, -0.0012068, 0.0012321
8: 0.0052387, 0.0072382, 0.0052307, 0.0072342, -0.0009574, 0.0009775
9: 0.0071469, 0.0107433, 0.0071325, 0.0107360, -0.0017219, 0.0017581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041045, -0.0040751, -0.0000142, 0.0000135
1: -0.0063776, -0.0052852, -0.0063681, -0.0052694, -0.0005305, 0.0005046
2: 0.9688101, 0.9701211, 0.9688215, 0.9701400, -0.0006366, 0.0006055
3: 0.0162545, 0.0259238, 0.0163382, 0.0260635, -0.0046953, 0.0044661
4: -0.0026647, -0.0019293, -0.0026753, -0.0019356, -0.0003397, 0.0003571
5: 0.0145772, 0.0153205, 0.0145665, 0.0153140, -0.0003433, 0.0003609
6: 0.0044549, 0.0048164, 0.0044580, 0.0048217, -0.0001756, 0.0001670
7: -0.0144966, -0.0119907, -0.0145328, -0.0120124, -0.0011574, 0.0012168
8: 0.0052282, 0.0072163, 0.0051995, 0.0071990, -0.0009182, 0.0009654
9: 0.0071280, 0.0107037, 0.0070764, 0.0106728, -0.0016515, 0.0017363

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004606
time: 0.90 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004620
time: 0.86 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041044, -0.0040753, -0.0000145, 0.0000133
1: -0.0063897, -0.0052909, -0.0063670, -0.0052748, -0.0005421, 0.0004973
2: 0.9687956, 0.9701142, 0.9688227, 0.9701335, -0.0006505, 0.0005967
3: 0.0161476, 0.0258727, 0.0163478, 0.0260155, -0.0047980, 0.0044013
4: -0.0026608, -0.0019211, -0.0026717, -0.0019364, -0.0003347, 0.0003649
5: 0.0145811, 0.0153287, 0.0145702, 0.0153133, -0.0003383, 0.0003688
6: 0.0044509, 0.0048145, 0.0044584, 0.0048199, -0.0001794, 0.0001646
7: -0.0144834, -0.0119630, -0.0145204, -0.0120149, -0.0011406, 0.0012434
8: 0.0052387, 0.0072382, 0.0052093, 0.0071971, -0.0009049, 0.0009865
9: 0.0071469, 0.0107433, 0.0070941, 0.0106692, -0.0016276, 0.0017743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004606
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004620
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041052, -0.0040758, -0.0000142, 0.0000143
1: -0.0064156, -0.0052854, -0.0063975, -0.0052939, -0.0005335, 0.0005350
2: 0.9687644, 0.9701208, 0.9687861, 0.9701107, -0.0006402, 0.0006420
3: 0.0159176, 0.0259219, 0.0160782, 0.0258468, -0.0047218, 0.0047353
4: -0.0026645, -0.0019037, -0.0026588, -0.0019159, -0.0003601, 0.0003591
5: 0.0145774, 0.0153464, 0.0145831, 0.0153340, -0.0003640, 0.0003630
6: 0.0044423, 0.0048164, 0.0044483, 0.0048136, -0.0001765, 0.0001770
7: -0.0144961, -0.0119034, -0.0144767, -0.0119451, -0.0012272, 0.0012237
8: 0.0052286, 0.0072855, 0.0052440, 0.0072525, -0.0009736, 0.0009708
9: 0.0071288, 0.0108283, 0.0071565, 0.0107689, -0.0017511, 0.0017461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004517
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004517
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041052, -0.0040759, -0.0000145, 0.0000141
1: -0.0064261, -0.0052930, -0.0063964, -0.0052991, -0.0005424, 0.0005272
2: 0.9687519, 0.9701117, 0.9687874, 0.9701043, -0.0006509, 0.0006326
3: 0.0158247, 0.0258548, 0.0160875, 0.0258000, -0.0048012, 0.0046662
4: -0.0026594, -0.0018966, -0.0026553, -0.0019166, -0.0003549, 0.0003652
5: 0.0145825, 0.0153535, 0.0145867, 0.0153333, -0.0003587, 0.0003691
6: 0.0044388, 0.0048139, 0.0044487, 0.0048118, -0.0001795, 0.0001745
7: -0.0144788, -0.0118794, -0.0144645, -0.0119475, -0.0012093, 0.0012443
8: 0.0052424, 0.0073046, 0.0052537, 0.0072506, -0.0009594, 0.0009871
9: 0.0071535, 0.0108627, 0.0071738, 0.0107655, -0.0017255, 0.0017755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004663, upper bound: 0.0004516
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004663, upper bound: 0.0004517
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041048, -0.0040754, -0.0000150, 0.0000141
1: -0.0064156, -0.0052854, -0.0063789, -0.0052813, -0.0005604, 0.0005286
2: 0.9687644, 0.9701208, 0.9688084, 0.9701257, -0.0006725, 0.0006344
3: 0.0159176, 0.0259219, 0.0162426, 0.0259579, -0.0049601, 0.0046792
4: -0.0026645, -0.0019037, -0.0026673, -0.0019284, -0.0003559, 0.0003772
5: 0.0145774, 0.0153464, 0.0145746, 0.0153214, -0.0003597, 0.0003813
6: 0.0044423, 0.0048164, 0.0044545, 0.0048177, -0.0001854, 0.0001749
7: -0.0144961, -0.0119034, -0.0145055, -0.0119877, -0.0012127, 0.0012855
8: 0.0052286, 0.0072855, 0.0052212, 0.0072187, -0.0009621, 0.0010198
9: 0.0071288, 0.0108283, 0.0071154, 0.0107081, -0.0017304, 0.0018342

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004660, upper bound: 0.0004520
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004660, upper bound: 0.0004521
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041047, -0.0040756, -0.0000152, 0.0000139
1: -0.0064261, -0.0052930, -0.0063779, -0.0052864, -0.0005698, 0.0005208
2: 0.9687519, 0.9701117, 0.9688098, 0.9701197, -0.0006838, 0.0006250
3: 0.0158247, 0.0258548, 0.0162518, 0.0259131, -0.0050439, 0.0046101
4: -0.0026594, -0.0018966, -0.0026639, -0.0019291, -0.0003506, 0.0003836
5: 0.0145825, 0.0153535, 0.0145780, 0.0153207, -0.0003544, 0.0003877
6: 0.0044388, 0.0048139, 0.0044548, 0.0048160, -0.0001886, 0.0001724
7: -0.0144788, -0.0118794, -0.0144939, -0.0119900, -0.0011947, 0.0013072
8: 0.0052424, 0.0073046, 0.0052304, 0.0072168, -0.0009479, 0.0010370
9: 0.0071535, 0.0108627, 0.0071320, 0.0107047, -0.0017048, 0.0018652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004520
time: 0.91 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004521
time: 0.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041052, -0.0040758, -0.0000140, 0.0000149
1: -0.0063960, -0.0052729, -0.0063975, -0.0052939, -0.0005250, 0.0005580
2: 0.9687881, 0.9701357, 0.9687861, 0.9701107, -0.0006300, 0.0006696
3: 0.0160918, 0.0260320, 0.0160782, 0.0258468, -0.0046466, 0.0049390
4: -0.0026729, -0.0019169, -0.0026588, -0.0019159, -0.0003756, 0.0003534
5: 0.0145689, 0.0153330, 0.0145831, 0.0153340, -0.0003797, 0.0003572
6: 0.0044488, 0.0048205, 0.0044483, 0.0048136, -0.0001737, 0.0001847
7: -0.0145247, -0.0119486, -0.0144767, -0.0119451, -0.0012800, 0.0012042
8: 0.0052059, 0.0072497, 0.0052440, 0.0072525, -0.0010155, 0.0009554
9: 0.0070880, 0.0107639, 0.0071565, 0.0107689, -0.0018264, 0.0017183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004521
time: 0.78 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004521
time: 0.80 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040754, -0.0041052, -0.0040759, -0.0000143, 0.0000147
1: -0.0064070, -0.0052800, -0.0063964, -0.0052991, -0.0005344, 0.0005507
2: 0.9687748, 0.9701272, 0.9687874, 0.9701043, -0.0006413, 0.0006608
3: 0.0159942, 0.0259694, 0.0160875, 0.0258000, -0.0047303, 0.0048743
4: -0.0026682, -0.0019095, -0.0026553, -0.0019166, -0.0003707, 0.0003598
5: 0.0145737, 0.0153405, 0.0145867, 0.0153333, -0.0003747, 0.0003636
6: 0.0044452, 0.0048181, 0.0044487, 0.0048118, -0.0001769, 0.0001822
7: -0.0145084, -0.0119233, -0.0144645, -0.0119475, -0.0012632, 0.0012259
8: 0.0052188, 0.0072698, 0.0052537, 0.0072506, -0.0010022, 0.0009726
9: 0.0071112, 0.0108000, 0.0071738, 0.0107655, -0.0018025, 0.0017492

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004521
time: 0.81 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004521
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041048, -0.0040754, -0.0000142, 0.0000143
1: -0.0063960, -0.0052729, -0.0063789, -0.0052813, -0.0005322, 0.0005344
2: 0.9687881, 0.9701357, 0.9688084, 0.9701257, -0.0006387, 0.0006413
3: 0.0160918, 0.0260320, 0.0162426, 0.0259579, -0.0047107, 0.0047300
4: -0.0026729, -0.0019169, -0.0026673, -0.0019284, -0.0003597, 0.0003583
5: 0.0145689, 0.0153330, 0.0145746, 0.0153214, -0.0003636, 0.0003621
6: 0.0044488, 0.0048205, 0.0044545, 0.0048177, -0.0001761, 0.0001768
7: -0.0145247, -0.0119486, -0.0145055, -0.0119877, -0.0012258, 0.0012208
8: 0.0052059, 0.0072497, 0.0052212, 0.0072187, -0.0009725, 0.0009685
9: 0.0070880, 0.0107639, 0.0071154, 0.0107081, -0.0017492, 0.0017420

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004734, upper bound: 0.0004591
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004734, upper bound: 0.0004594
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040754, -0.0041047, -0.0040756, -0.0000145, 0.0000141
1: -0.0064070, -0.0052800, -0.0063779, -0.0052864, -0.0005413, 0.0005266
2: 0.9687748, 0.9701272, 0.9688098, 0.9701197, -0.0006495, 0.0006320
3: 0.0159942, 0.0259694, 0.0162518, 0.0259131, -0.0047908, 0.0046612
4: -0.0026682, -0.0019095, -0.0026639, -0.0019291, -0.0003545, 0.0003644
5: 0.0145737, 0.0153405, 0.0145780, 0.0153207, -0.0003583, 0.0003683
6: 0.0044452, 0.0048181, 0.0044548, 0.0048160, -0.0001791, 0.0001743
7: -0.0145084, -0.0119233, -0.0144939, -0.0119900, -0.0012080, 0.0012416
8: 0.0052188, 0.0072698, 0.0052304, 0.0072168, -0.0009584, 0.0009850
9: 0.0071112, 0.0108000, 0.0071320, 0.0107047, -0.0017237, 0.0017716

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004745, upper bound: 0.0004591
time: 0.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004745, upper bound: 0.0004594
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041050, -0.0040754, -0.0000151, 0.0000145
1: -0.0064156, -0.0052854, -0.0063885, -0.0052811, -0.0005643, 0.0005437
2: 0.9687644, 0.9701208, 0.9687970, 0.9701260, -0.0006771, 0.0006524
3: 0.0159176, 0.0259219, 0.0161576, 0.0259601, -0.0049945, 0.0048122
4: -0.0026645, -0.0019037, -0.0026674, -0.0019219, -0.0003660, 0.0003799
5: 0.0145774, 0.0153464, 0.0145744, 0.0153279, -0.0003699, 0.0003839
6: 0.0044423, 0.0048164, 0.0044513, 0.0048178, -0.0001867, 0.0001799
7: -0.0144961, -0.0119034, -0.0145060, -0.0119657, -0.0012471, 0.0012944
8: 0.0052286, 0.0072855, 0.0052207, 0.0072362, -0.0009894, 0.0010269
9: 0.0071288, 0.0108283, 0.0071146, 0.0107396, -0.0017795, 0.0018469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
time: 0.84 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004451
time: 0.84 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041050, -0.0040756, -0.0000153, 0.0000143
1: -0.0064261, -0.0052930, -0.0063874, -0.0052865, -0.0005732, 0.0005358
2: 0.9687519, 0.9701117, 0.9687982, 0.9701194, -0.0006878, 0.0006430
3: 0.0158247, 0.0258548, 0.0161673, 0.0259117, -0.0050732, 0.0047423
4: -0.0026594, -0.0018966, -0.0026638, -0.0019226, -0.0003607, 0.0003858
5: 0.0145825, 0.0153535, 0.0145781, 0.0153272, -0.0003645, 0.0003900
6: 0.0044388, 0.0048139, 0.0044517, 0.0048160, -0.0001897, 0.0001773
7: -0.0144788, -0.0118794, -0.0144935, -0.0119682, -0.0012290, 0.0013148
8: 0.0052424, 0.0073046, 0.0052307, 0.0072342, -0.0009750, 0.0010431
9: 0.0071535, 0.0108627, 0.0071325, 0.0107360, -0.0017537, 0.0018761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
time: 0.87 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
time: 0.85 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041045, -0.0040751, -0.0000156, 0.0000141
1: -0.0064156, -0.0052854, -0.0063681, -0.0052694, -0.0005829, 0.0005293
2: 0.9687644, 0.9701208, 0.9688215, 0.9701400, -0.0006996, 0.0006352
3: 0.0159176, 0.0259219, 0.0163382, 0.0260635, -0.0051598, 0.0046850
4: -0.0026645, -0.0019037, -0.0026753, -0.0019356, -0.0003563, 0.0003924
5: 0.0145774, 0.0153464, 0.0145665, 0.0153140, -0.0003601, 0.0003966
6: 0.0044423, 0.0048164, 0.0044580, 0.0048217, -0.0001929, 0.0001752
7: -0.0144961, -0.0119034, -0.0145328, -0.0120124, -0.0012141, 0.0013372
8: 0.0052286, 0.0072855, 0.0051995, 0.0071990, -0.0009632, 0.0010609
9: 0.0071288, 0.0108283, 0.0070764, 0.0106728, -0.0017325, 0.0019081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004465
time: 0.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004469
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041044, -0.0040753, -0.0000158, 0.0000139
1: -0.0064261, -0.0052930, -0.0063670, -0.0052748, -0.0005922, 0.0005214
2: 0.9687519, 0.9701117, 0.9688227, 0.9701335, -0.0007106, 0.0006258
3: 0.0158247, 0.0258548, 0.0163478, 0.0260155, -0.0052416, 0.0046154
4: -0.0026594, -0.0018966, -0.0026717, -0.0019364, -0.0003510, 0.0003987
5: 0.0145825, 0.0153535, 0.0145702, 0.0153133, -0.0003548, 0.0004029
6: 0.0044388, 0.0048139, 0.0044584, 0.0048199, -0.0001960, 0.0001726
7: -0.0144788, -0.0118794, -0.0145204, -0.0120149, -0.0011961, 0.0013584
8: 0.0052424, 0.0073046, 0.0052093, 0.0071971, -0.0009490, 0.0010777
9: 0.0071535, 0.0108627, 0.0070941, 0.0106692, -0.0017068, 0.0019383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004465
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004469
time: 0.88 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041050, -0.0040754, -0.0000148, 0.0000151
1: -0.0063960, -0.0052729, -0.0063885, -0.0052811, -0.0005558, 0.0005667
2: 0.9687881, 0.9701357, 0.9687970, 0.9701260, -0.0006669, 0.0006800
3: 0.0160918, 0.0260320, 0.0161576, 0.0259601, -0.0049193, 0.0050159
4: -0.0026729, -0.0019169, -0.0026674, -0.0019219, -0.0003815, 0.0003741
5: 0.0145689, 0.0153330, 0.0145744, 0.0153279, -0.0003856, 0.0003781
6: 0.0044488, 0.0048205, 0.0044513, 0.0048178, -0.0001839, 0.0001875
7: -0.0145247, -0.0119486, -0.0145060, -0.0119657, -0.0012999, 0.0012749
8: 0.0052059, 0.0072497, 0.0052207, 0.0072362, -0.0010313, 0.0010114
9: 0.0070880, 0.0107639, 0.0071146, 0.0107396, -0.0018549, 0.0018191

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
time: 0.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
time: 0.82 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040754, -0.0041050, -0.0040756, -0.0000151, 0.0000149
1: -0.0064070, -0.0052800, -0.0063874, -0.0052865, -0.0005651, 0.0005593
2: 0.9687748, 0.9701272, 0.9687982, 0.9701194, -0.0006782, 0.0006712
3: 0.0159942, 0.0259694, 0.0161673, 0.0259117, -0.0050022, 0.0049504
4: -0.0026682, -0.0019095, -0.0026638, -0.0019226, -0.0003765, 0.0003804
5: 0.0145737, 0.0153405, 0.0145781, 0.0153272, -0.0003805, 0.0003845
6: 0.0044452, 0.0048181, 0.0044517, 0.0048160, -0.0001870, 0.0001851
7: -0.0145084, -0.0119233, -0.0144935, -0.0119682, -0.0012829, 0.0012964
8: 0.0052188, 0.0072698, 0.0052307, 0.0072342, -0.0010178, 0.0010285
9: 0.0071112, 0.0108000, 0.0071325, 0.0107360, -0.0018307, 0.0018498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
time: 0.85 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
time: 0.83 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041045, -0.0040751, -0.0000151, 0.0000145
1: -0.0063960, -0.0052729, -0.0063681, -0.0052694, -0.0005643, 0.0005427
2: 0.9687881, 0.9701357, 0.9688215, 0.9701400, -0.0006771, 0.0006513
3: 0.0160918, 0.0260320, 0.0163382, 0.0260635, -0.0049944, 0.0048039
4: -0.0026729, -0.0019169, -0.0026753, -0.0019356, -0.0003654, 0.0003799
5: 0.0145689, 0.0153330, 0.0145665, 0.0153140, -0.0003693, 0.0003839
6: 0.0044488, 0.0048205, 0.0044580, 0.0048217, -0.0001867, 0.0001796
7: -0.0145247, -0.0119486, -0.0145328, -0.0120124, -0.0012450, 0.0012943
8: 0.0052059, 0.0072497, 0.0051995, 0.0071990, -0.0009877, 0.0010269
9: 0.0070880, 0.0107639, 0.0070764, 0.0106728, -0.0017765, 0.0018469

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004556
time: 0.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004566
time: 0.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040754, -0.0041044, -0.0040753, -0.0000153, 0.0000143
1: -0.0064070, -0.0052800, -0.0063670, -0.0052748, -0.0005732, 0.0005349
2: 0.9687748, 0.9701272, 0.9688227, 0.9701335, -0.0006879, 0.0006419
3: 0.0159942, 0.0259694, 0.0163478, 0.0260155, -0.0050737, 0.0047345
4: -0.0026682, -0.0019095, -0.0026717, -0.0019364, -0.0003601, 0.0003859
5: 0.0145737, 0.0153405, 0.0145702, 0.0153133, -0.0003639, 0.0003900
6: 0.0044452, 0.0048181, 0.0044584, 0.0048199, -0.0001897, 0.0001770
7: -0.0145084, -0.0119233, -0.0145204, -0.0120149, -0.0012270, 0.0013149
8: 0.0052188, 0.0072698, 0.0052093, 0.0071971, -0.0009734, 0.0010432
9: 0.0071112, 0.0108000, 0.0070941, 0.0106692, -0.0017508, 0.0018762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=10, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004740, upper bound: 0.0004556
time: 0.88 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004740, upper bound: 0.0004566
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041058, -0.0040755, -0.0000143, 0.0000142
1: -0.0063961, -0.0052976, -0.0064171, -0.0052819, -0.0005371, 0.0005315
2: 0.9687880, 0.9701061, 0.9687627, 0.9701250, -0.0006445, 0.0006378
3: 0.0160907, 0.0258135, 0.0159048, 0.0259526, -0.0047539, 0.0047044
4: -0.0026563, -0.0019168, -0.0026669, -0.0019027, -0.0003578, 0.0003616
5: 0.0145857, 0.0153331, 0.0145750, 0.0153473, -0.0003616, 0.0003654
6: 0.0044488, 0.0048123, 0.0044418, 0.0048175, -0.0001777, 0.0001759
7: -0.0144680, -0.0119483, -0.0145041, -0.0119001, -0.0012192, 0.0012320
8: 0.0052509, 0.0072499, 0.0052223, 0.0072882, -0.0009672, 0.0009774
9: 0.0071688, 0.0107643, 0.0071174, 0.0108331, -0.0017397, 0.0017580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
time: 0.83 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041057, -0.0040756, -0.0000147, 0.0000140
1: -0.0064074, -0.0053039, -0.0064160, -0.0052874, -0.0005495, 0.0005238
2: 0.9687743, 0.9700986, 0.9687641, 0.9701183, -0.0006594, 0.0006285
3: 0.0159906, 0.0257579, 0.0159148, 0.0259039, -0.0048639, 0.0046359
4: -0.0026521, -0.0019092, -0.0026632, -0.0019034, -0.0003526, 0.0003699
5: 0.0145900, 0.0153408, 0.0145787, 0.0153466, -0.0003564, 0.0003739
6: 0.0044450, 0.0048102, 0.0044422, 0.0048157, -0.0001819, 0.0001733
7: -0.0144536, -0.0119224, -0.0144915, -0.0119027, -0.0012014, 0.0012605
8: 0.0052623, 0.0072705, 0.0052323, 0.0072861, -0.0009532, 0.0010000
9: 0.0071894, 0.0108013, 0.0071354, 0.0108294, -0.0017143, 0.0017987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041052, -0.0040751, -0.0000150, 0.0000140
1: -0.0063961, -0.0052976, -0.0063974, -0.0052693, -0.0005604, 0.0005229
2: 0.9687880, 0.9701061, 0.9687864, 0.9701401, -0.0006725, 0.0006275
3: 0.0160907, 0.0258135, 0.0160793, 0.0260644, -0.0049601, 0.0046285
4: -0.0026563, -0.0019168, -0.0026754, -0.0019159, -0.0003520, 0.0003772
5: 0.0145857, 0.0153331, 0.0145664, 0.0153339, -0.0003558, 0.0003813
6: 0.0044488, 0.0048123, 0.0044484, 0.0048217, -0.0001855, 0.0001731
7: -0.0144680, -0.0119483, -0.0145331, -0.0119453, -0.0011995, 0.0012855
8: 0.0052509, 0.0072499, 0.0051993, 0.0072523, -0.0009516, 0.0010198
9: 0.0071688, 0.0107643, 0.0070760, 0.0107685, -0.0017116, 0.0018343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004672
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004674
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041052, -0.0040753, -0.0000153, 0.0000138
1: -0.0064074, -0.0053039, -0.0063963, -0.0052744, -0.0005730, 0.0005153
2: 0.9687743, 0.9700986, 0.9687877, 0.9701339, -0.0006876, 0.0006183
3: 0.0159906, 0.0257579, 0.0160890, 0.0260187, -0.0050716, 0.0045607
4: -0.0026521, -0.0019092, -0.0026719, -0.0019167, -0.0003469, 0.0003857
5: 0.0145900, 0.0153408, 0.0145699, 0.0153332, -0.0003506, 0.0003898
6: 0.0044450, 0.0048102, 0.0044487, 0.0048200, -0.0001896, 0.0001705
7: -0.0144536, -0.0119224, -0.0145212, -0.0119479, -0.0011819, 0.0013144
8: 0.0052623, 0.0072705, 0.0052087, 0.0072503, -0.0009377, 0.0010427
9: 0.0071894, 0.0108013, 0.0070929, 0.0107649, -0.0016865, 0.0018755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004672
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004674
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041058, -0.0040755, -0.0000142, 0.0000149
1: -0.0063776, -0.0052852, -0.0064171, -0.0052819, -0.0005308, 0.0005581
2: 0.9688101, 0.9701211, 0.9687627, 0.9701250, -0.0006370, 0.0006697
3: 0.0162545, 0.0259238, 0.0159048, 0.0259526, -0.0046982, 0.0049398
4: -0.0026647, -0.0019293, -0.0026669, -0.0019027, -0.0003757, 0.0003573
5: 0.0145772, 0.0153205, 0.0145750, 0.0153473, -0.0003797, 0.0003611
6: 0.0044549, 0.0048164, 0.0044418, 0.0048175, -0.0001757, 0.0001847
7: -0.0144966, -0.0119907, -0.0145041, -0.0119001, -0.0012802, 0.0012176
8: 0.0052282, 0.0072163, 0.0052223, 0.0072882, -0.0010157, 0.0009660
9: 0.0071280, 0.0107037, 0.0071174, 0.0108331, -0.0018267, 0.0017374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004660
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004661
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041057, -0.0040756, -0.0000145, 0.0000147
1: -0.0063897, -0.0052909, -0.0064160, -0.0052874, -0.0005442, 0.0005514
2: 0.9687956, 0.9701142, 0.9687641, 0.9701183, -0.0006531, 0.0006618
3: 0.0161476, 0.0258727, 0.0159148, 0.0259039, -0.0048173, 0.0048810
4: -0.0026608, -0.0019211, -0.0026632, -0.0019034, -0.0003712, 0.0003664
5: 0.0145811, 0.0153287, 0.0145787, 0.0153466, -0.0003752, 0.0003703
6: 0.0044509, 0.0048145, 0.0044422, 0.0048157, -0.0001801, 0.0001825
7: -0.0144834, -0.0119630, -0.0144915, -0.0119027, -0.0012650, 0.0012484
8: 0.0052387, 0.0072382, 0.0052323, 0.0072861, -0.0010036, 0.0009905
9: 0.0071469, 0.0107433, 0.0071354, 0.0108294, -0.0018050, 0.0017814

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004660
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004661
time: 0.79 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041052, -0.0040751, -0.0000143, 0.0000142
1: -0.0063776, -0.0052852, -0.0063974, -0.0052693, -0.0005365, 0.0005303
2: 0.9688101, 0.9701211, 0.9687864, 0.9701401, -0.0006438, 0.0006363
3: 0.0162545, 0.0259238, 0.0160793, 0.0260644, -0.0047485, 0.0046935
4: -0.0026647, -0.0019293, -0.0026754, -0.0019159, -0.0003570, 0.0003612
5: 0.0145772, 0.0153205, 0.0145664, 0.0153339, -0.0003608, 0.0003650
6: 0.0044549, 0.0048164, 0.0044484, 0.0048217, -0.0001775, 0.0001755
7: -0.0144966, -0.0119907, -0.0145331, -0.0119453, -0.0012163, 0.0012306
8: 0.0052282, 0.0072163, 0.0051993, 0.0072523, -0.0009650, 0.0009763
9: 0.0071280, 0.0107037, 0.0070760, 0.0107685, -0.0017356, 0.0017560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004732
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004738
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041052, -0.0040753, -0.0000146, 0.0000140
1: -0.0063897, -0.0052909, -0.0063963, -0.0052744, -0.0005484, 0.0005228
2: 0.9687956, 0.9701142, 0.9687877, 0.9701339, -0.0006581, 0.0006274
3: 0.0161476, 0.0258727, 0.0160890, 0.0260187, -0.0048538, 0.0046274
4: -0.0026608, -0.0019211, -0.0026719, -0.0019167, -0.0003519, 0.0003692
5: 0.0145811, 0.0153287, 0.0145699, 0.0153332, -0.0003557, 0.0003731
6: 0.0044509, 0.0048145, 0.0044487, 0.0048200, -0.0001815, 0.0001730
7: -0.0144834, -0.0119630, -0.0145212, -0.0119479, -0.0011992, 0.0012579
8: 0.0052387, 0.0072382, 0.0052087, 0.0072503, -0.0009514, 0.0009980
9: 0.0071469, 0.0107433, 0.0070929, 0.0107649, -0.0017112, 0.0017949

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004731
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004738
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041056, -0.0040751, -0.0000150, 0.0000143
1: -0.0063961, -0.0052976, -0.0064096, -0.0052675, -0.0005624, 0.0005360
2: 0.9687880, 0.9701061, 0.9687717, 0.9701422, -0.0006750, 0.0006432
3: 0.0160907, 0.0258135, 0.0159711, 0.0260805, -0.0049783, 0.0047442
4: -0.0026563, -0.0019168, -0.0026766, -0.0019077, -0.0003608, 0.0003786
5: 0.0145857, 0.0153331, 0.0145652, 0.0153423, -0.0003647, 0.0003827
6: 0.0044488, 0.0048123, 0.0044443, 0.0048223, -0.0001861, 0.0001774
7: -0.0144680, -0.0119483, -0.0145372, -0.0119173, -0.0012295, 0.0012902
8: 0.0052509, 0.0072499, 0.0051960, 0.0072745, -0.0009754, 0.0010236
9: 0.0071688, 0.0107643, 0.0070701, 0.0108085, -0.0017544, 0.0018410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041055, -0.0040752, -0.0000153, 0.0000141
1: -0.0064074, -0.0053039, -0.0064085, -0.0052732, -0.0005748, 0.0005283
2: 0.9687743, 0.9700986, 0.9687730, 0.9701354, -0.0006897, 0.0006340
3: 0.0159906, 0.0257579, 0.0159811, 0.0260298, -0.0050874, 0.0046760
4: -0.0026521, -0.0019092, -0.0026727, -0.0019085, -0.0003556, 0.0003869
5: 0.0145900, 0.0153408, 0.0145691, 0.0153415, -0.0003594, 0.0003911
6: 0.0044450, 0.0048102, 0.0044447, 0.0048204, -0.0001902, 0.0001748
7: -0.0144536, -0.0119224, -0.0145241, -0.0119199, -0.0012118, 0.0013185
8: 0.0052623, 0.0072705, 0.0052064, 0.0072725, -0.0009614, 0.0010460
9: 0.0071894, 0.0108013, 0.0070889, 0.0108048, -0.0017292, 0.0018813

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
time: 0.86 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040759, -0.0041050, -0.0040748, -0.0000155, 0.0000139
1: -0.0063961, -0.0052976, -0.0063890, -0.0052558, -0.0005792, 0.0005216
2: 0.9687880, 0.9701061, 0.9687964, 0.9701563, -0.0006951, 0.0006259
3: 0.0160907, 0.0258135, 0.0161538, 0.0261836, -0.0051266, 0.0046165
4: -0.0026563, -0.0019168, -0.0026844, -0.0019216, -0.0003511, 0.0003899
5: 0.0145857, 0.0153331, 0.0145572, 0.0153282, -0.0003549, 0.0003941
6: 0.0044488, 0.0048123, 0.0044511, 0.0048261, -0.0001917, 0.0001726
7: -0.0144680, -0.0119483, -0.0145640, -0.0119646, -0.0011964, 0.0013286
8: 0.0052509, 0.0072499, 0.0051748, 0.0072370, -0.0009492, 0.0010541
9: 0.0071688, 0.0107643, 0.0070320, 0.0107410, -0.0017072, 0.0018958

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004613
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004621
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040760, -0.0041050, -0.0040749, -0.0000158, 0.0000137
1: -0.0064074, -0.0053039, -0.0063878, -0.0052613, -0.0005918, 0.0005139
2: 0.9687743, 0.9700986, 0.9687979, 0.9701496, -0.0007101, 0.0006167
3: 0.0159906, 0.0257579, 0.0161641, 0.0261346, -0.0052378, 0.0045485
4: -0.0026521, -0.0019092, -0.0026807, -0.0019224, -0.0003459, 0.0003984
5: 0.0145900, 0.0153408, 0.0145610, 0.0153274, -0.0003496, 0.0004026
6: 0.0044450, 0.0048102, 0.0044515, 0.0048243, -0.0001958, 0.0001701
7: -0.0144536, -0.0119224, -0.0145513, -0.0119673, -0.0011788, 0.0013574
8: 0.0052623, 0.0072705, 0.0051849, 0.0072348, -0.0009352, 0.0010769
9: 0.0071894, 0.0108013, 0.0070501, 0.0107372, -0.0016820, 0.0019369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004613
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004621
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041056, -0.0040751, -0.0000149, 0.0000150
1: -0.0063776, -0.0052852, -0.0064096, -0.0052675, -0.0005561, 0.0005626
2: 0.9688101, 0.9701211, 0.9687717, 0.9701422, -0.0006674, 0.0006751
3: 0.0162545, 0.0259238, 0.0159711, 0.0260805, -0.0049226, 0.0049797
4: -0.0026647, -0.0019293, -0.0026766, -0.0019077, -0.0003787, 0.0003744
5: 0.0145772, 0.0153205, 0.0145652, 0.0153423, -0.0003828, 0.0003784
6: 0.0044549, 0.0048164, 0.0044443, 0.0048223, -0.0001840, 0.0001862
7: -0.0144966, -0.0119907, -0.0145372, -0.0119173, -0.0012905, 0.0012757
8: 0.0052282, 0.0072163, 0.0051960, 0.0072745, -0.0010238, 0.0010121
9: 0.0071280, 0.0107037, 0.0070701, 0.0108085, -0.0018415, 0.0018204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
time: 0.84 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041055, -0.0040752, -0.0000152, 0.0000148
1: -0.0063897, -0.0052909, -0.0064085, -0.0052732, -0.0005695, 0.0005560
2: 0.9687956, 0.9701142, 0.9687730, 0.9701354, -0.0006834, 0.0006672
3: 0.0161476, 0.0258727, 0.0159811, 0.0260298, -0.0050408, 0.0049211
4: -0.0026608, -0.0019211, -0.0026727, -0.0019085, -0.0003743, 0.0003834
5: 0.0145811, 0.0153287, 0.0145691, 0.0153415, -0.0003783, 0.0003875
6: 0.0044509, 0.0048145, 0.0044447, 0.0048204, -0.0001885, 0.0001840
7: -0.0144834, -0.0119630, -0.0145241, -0.0119199, -0.0012753, 0.0013064
8: 0.0052387, 0.0072382, 0.0052064, 0.0072725, -0.0010118, 0.0010364
9: 0.0071469, 0.0107433, 0.0070889, 0.0108048, -0.0018198, 0.0018641

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
time: 0.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041047, -0.0040755, -0.0041050, -0.0040748, -0.0000150, 0.0000143
1: -0.0063776, -0.0052852, -0.0063890, -0.0052558, -0.0005628, 0.0005348
2: 0.9688101, 0.9701211, 0.9687964, 0.9701563, -0.0006754, 0.0006417
3: 0.0162545, 0.0259238, 0.0161538, 0.0261836, -0.0049815, 0.0047334
4: -0.0026647, -0.0019293, -0.0026844, -0.0019216, -0.0003600, 0.0003789
5: 0.0145772, 0.0153205, 0.0145572, 0.0153282, -0.0003638, 0.0003829
6: 0.0044549, 0.0048164, 0.0044511, 0.0048261, -0.0001862, 0.0001770
7: -0.0144966, -0.0119907, -0.0145640, -0.0119646, -0.0012267, 0.0012910
8: 0.0052282, 0.0072163, 0.0051748, 0.0072370, -0.0009732, 0.0010242
9: 0.0071280, 0.0107037, 0.0070320, 0.0107410, -0.0017504, 0.0018421

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004587, upper bound: 0.0004693
time: 0.96 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004587, upper bound: 0.0004710
time: 0.95 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041050, -0.0040757, -0.0041050, -0.0040749, -0.0000153, 0.0000141
1: -0.0063897, -0.0052909, -0.0063878, -0.0052613, -0.0005746, 0.0005273
2: 0.9687956, 0.9701142, 0.9687979, 0.9701496, -0.0006895, 0.0006328
3: 0.0161476, 0.0258727, 0.0161641, 0.0261346, -0.0050856, 0.0046672
4: -0.0026608, -0.0019211, -0.0026807, -0.0019224, -0.0003550, 0.0003868
5: 0.0145811, 0.0153287, 0.0145610, 0.0153274, -0.0003588, 0.0003909
6: 0.0044509, 0.0048145, 0.0044515, 0.0048243, -0.0001901, 0.0001745
7: -0.0144834, -0.0119630, -0.0145513, -0.0119673, -0.0012095, 0.0013180
8: 0.0052387, 0.0072382, 0.0051849, 0.0072348, -0.0009596, 0.0010456
9: 0.0071469, 0.0107433, 0.0070501, 0.0107372, -0.0017259, 0.0018806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004693
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004710
time: 0.90 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041058, -0.0040755, -0.0000138, 0.0000137
1: -0.0064156, -0.0052854, -0.0064171, -0.0052819, -0.0005157, 0.0005135
2: 0.9687644, 0.9701208, 0.9687627, 0.9701250, -0.0006189, 0.0006162
3: 0.0159176, 0.0259219, 0.0159048, 0.0259526, -0.0045646, 0.0045449
4: -0.0026645, -0.0019037, -0.0026669, -0.0019027, -0.0003457, 0.0003472
5: 0.0145774, 0.0153464, 0.0145750, 0.0153473, -0.0003494, 0.0003509
6: 0.0044423, 0.0048164, 0.0044418, 0.0048175, -0.0001707, 0.0001699
7: -0.0144961, -0.0119034, -0.0145041, -0.0119001, -0.0011779, 0.0011830
8: 0.0052286, 0.0072855, 0.0052223, 0.0072882, -0.0009345, 0.0009385
9: 0.0071288, 0.0108283, 0.0071174, 0.0108331, -0.0016807, 0.0016880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
time: 0.88 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041057, -0.0040756, -0.0000141, 0.0000135
1: -0.0064261, -0.0052930, -0.0064160, -0.0052874, -0.0005276, 0.0005055
2: 0.9687519, 0.9701117, 0.9687641, 0.9701183, -0.0006332, 0.0006067
3: 0.0158247, 0.0258548, 0.0159148, 0.0259039, -0.0046700, 0.0044747
4: -0.0026594, -0.0018966, -0.0026632, -0.0019034, -0.0003403, 0.0003552
5: 0.0145825, 0.0153535, 0.0145787, 0.0153466, -0.0003440, 0.0003590
6: 0.0044388, 0.0048139, 0.0044422, 0.0048157, -0.0001746, 0.0001673
7: -0.0144788, -0.0118794, -0.0144915, -0.0119027, -0.0011597, 0.0012103
8: 0.0052424, 0.0073046, 0.0052323, 0.0072861, -0.0009200, 0.0009602
9: 0.0071535, 0.0108627, 0.0071354, 0.0108294, -0.0016548, 0.0017270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
time: 0.78 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
time: 0.80 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041052, -0.0040751, -0.0000145, 0.0000135
1: -0.0064156, -0.0052854, -0.0063974, -0.0052693, -0.0005423, 0.0005068
2: 0.9687644, 0.9701208, 0.9687864, 0.9701401, -0.0006507, 0.0006082
3: 0.0159176, 0.0259219, 0.0160793, 0.0260644, -0.0047998, 0.0044862
4: -0.0026645, -0.0019037, -0.0026754, -0.0019159, -0.0003412, 0.0003650
5: 0.0145774, 0.0153464, 0.0145664, 0.0153339, -0.0003448, 0.0003689
6: 0.0044423, 0.0048164, 0.0044484, 0.0048217, -0.0001795, 0.0001677
7: -0.0144961, -0.0119034, -0.0145331, -0.0119453, -0.0011626, 0.0012439
8: 0.0052286, 0.0072855, 0.0051993, 0.0072523, -0.0009224, 0.0009868
9: 0.0071288, 0.0108283, 0.0070760, 0.0107685, -0.0016590, 0.0017749

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004538
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004539
time: 0.99 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041052, -0.0040753, -0.0000148, 0.0000133
1: -0.0064261, -0.0052930, -0.0063963, -0.0052744, -0.0005546, 0.0004989
2: 0.9687519, 0.9701117, 0.9687877, 0.9701339, -0.0006656, 0.0005987
3: 0.0158247, 0.0258548, 0.0160890, 0.0260187, -0.0049090, 0.0044162
4: -0.0026594, -0.0018966, -0.0026719, -0.0019167, -0.0003359, 0.0003734
5: 0.0145825, 0.0153535, 0.0145699, 0.0153332, -0.0003395, 0.0003773
6: 0.0044388, 0.0048139, 0.0044487, 0.0048200, -0.0001835, 0.0001651
7: -0.0144788, -0.0118794, -0.0145212, -0.0119479, -0.0011445, 0.0012722
8: 0.0052424, 0.0073046, 0.0052087, 0.0072503, -0.0009080, 0.0010093
9: 0.0071535, 0.0108627, 0.0070929, 0.0107649, -0.0016331, 0.0018154

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004669, upper bound: 0.0004538
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004669, upper bound: 0.0004539
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041058, -0.0040755, -0.0000136, 0.0000144
1: -0.0063960, -0.0052729, -0.0064171, -0.0052819, -0.0005091, 0.0005397
2: 0.9687881, 0.9701357, 0.9687627, 0.9701250, -0.0006109, 0.0006476
3: 0.0160918, 0.0260320, 0.0159048, 0.0259526, -0.0045061, 0.0047769
4: -0.0026729, -0.0019169, -0.0026669, -0.0019027, -0.0003633, 0.0003427
5: 0.0145689, 0.0153330, 0.0145750, 0.0153473, -0.0003672, 0.0003464
6: 0.0044488, 0.0048205, 0.0044418, 0.0048175, -0.0001685, 0.0001786
7: -0.0145247, -0.0119486, -0.0145041, -0.0119001, -0.0012380, 0.0011678
8: 0.0052059, 0.0072497, 0.0052223, 0.0072882, -0.0009821, 0.0009265
9: 0.0070880, 0.0107639, 0.0071174, 0.0108331, -0.0017665, 0.0016663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004538
time: 0.80 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004538
time: 0.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040754, -0.0041057, -0.0040756, -0.0000140, 0.0000142
1: -0.0064070, -0.0052800, -0.0064160, -0.0052874, -0.0005226, 0.0005329
2: 0.9687748, 0.9701272, 0.9687641, 0.9701183, -0.0006271, 0.0006395
3: 0.0159942, 0.0259694, 0.0159148, 0.0259039, -0.0046255, 0.0047172
4: -0.0026682, -0.0019095, -0.0026632, -0.0019034, -0.0003588, 0.0003518
5: 0.0145737, 0.0153405, 0.0145787, 0.0153466, -0.0003626, 0.0003556
6: 0.0044452, 0.0048181, 0.0044422, 0.0048157, -0.0001729, 0.0001764
7: -0.0145084, -0.0119233, -0.0144915, -0.0119027, -0.0012225, 0.0011987
8: 0.0052188, 0.0072698, 0.0052323, 0.0072861, -0.0009699, 0.0009510
9: 0.0071112, 0.0108000, 0.0071354, 0.0108294, -0.0017444, 0.0017105

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004538
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004538
time: 0.84 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041052, -0.0040751, -0.0000137, 0.0000137
1: -0.0063960, -0.0052729, -0.0063974, -0.0052693, -0.0005146, 0.0005124
2: 0.9687881, 0.9701357, 0.9687864, 0.9701401, -0.0006175, 0.0006149
3: 0.0160918, 0.0260320, 0.0160793, 0.0260644, -0.0045549, 0.0045353
4: -0.0026729, -0.0019169, -0.0026754, -0.0019159, -0.0003449, 0.0003464
5: 0.0145689, 0.0153330, 0.0145664, 0.0153339, -0.0003486, 0.0003501
6: 0.0044488, 0.0048205, 0.0044484, 0.0048217, -0.0001703, 0.0001696
7: -0.0145247, -0.0119486, -0.0145331, -0.0119453, -0.0011754, 0.0011804
8: 0.0052059, 0.0072497, 0.0051993, 0.0072523, -0.0009325, 0.0009365
9: 0.0070880, 0.0107639, 0.0070760, 0.0107685, -0.0016772, 0.0016844

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004595
time: 0.89 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004598
time: 0.87 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040754, -0.0041052, -0.0040753, -0.0000141, 0.0000135
1: -0.0064070, -0.0052800, -0.0063963, -0.0052744, -0.0005262, 0.0005045
2: 0.9687748, 0.9701272, 0.9687877, 0.9701339, -0.0006315, 0.0006055
3: 0.0159942, 0.0259694, 0.0160890, 0.0260187, -0.0046579, 0.0044659
4: -0.0026682, -0.0019095, -0.0026719, -0.0019167, -0.0003397, 0.0003543
5: 0.0145737, 0.0153405, 0.0145699, 0.0153332, -0.0003433, 0.0003580
6: 0.0044452, 0.0048181, 0.0044487, 0.0048200, -0.0001742, 0.0001670
7: -0.0145084, -0.0119233, -0.0145212, -0.0119479, -0.0011574, 0.0012071
8: 0.0052188, 0.0072698, 0.0052087, 0.0072503, -0.0009182, 0.0009577
9: 0.0071112, 0.0108000, 0.0070929, 0.0107649, -0.0016515, 0.0017225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004595
time: 0.93 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004597
time: 1.02 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041056, -0.0040751, -0.0000146, 0.0000139
1: -0.0064156, -0.0052854, -0.0064096, -0.0052675, -0.0005468, 0.0005223
2: 0.9687644, 0.9701208, 0.9687717, 0.9701422, -0.0006561, 0.0006268
3: 0.0159176, 0.0259219, 0.0159711, 0.0260805, -0.0048395, 0.0046229
4: -0.0026645, -0.0019037, -0.0026766, -0.0019077, -0.0003516, 0.0003681
5: 0.0145774, 0.0153464, 0.0145652, 0.0153423, -0.0003554, 0.0003720
6: 0.0044423, 0.0048164, 0.0044443, 0.0048223, -0.0001809, 0.0001728
7: -0.0144961, -0.0119034, -0.0145372, -0.0119173, -0.0011981, 0.0012542
8: 0.0052286, 0.0072855, 0.0051960, 0.0072745, -0.0009505, 0.0009950
9: 0.0071288, 0.0108283, 0.0070701, 0.0108085, -0.0017095, 0.0017897

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
time: 0.81 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041055, -0.0040752, -0.0000149, 0.0000137
1: -0.0064261, -0.0052930, -0.0064085, -0.0052732, -0.0005586, 0.0005143
2: 0.9687519, 0.9701117, 0.9687730, 0.9701354, -0.0006703, 0.0006172
3: 0.0158247, 0.0258548, 0.0159811, 0.0260298, -0.0049439, 0.0045520
4: -0.0026594, -0.0018966, -0.0026727, -0.0019085, -0.0003462, 0.0003760
5: 0.0145825, 0.0153535, 0.0145691, 0.0153415, -0.0003499, 0.0003800
6: 0.0044388, 0.0048139, 0.0044447, 0.0048204, -0.0001848, 0.0001702
7: -0.0144788, -0.0118794, -0.0145241, -0.0119199, -0.0011797, 0.0012813
8: 0.0052424, 0.0073046, 0.0052064, 0.0072725, -0.0009359, 0.0010165
9: 0.0071535, 0.0108627, 0.0070889, 0.0108048, -0.0016833, 0.0018283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
time: 0.87 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
time: 0.86 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041057, -0.0040756, -0.0041050, -0.0040748, -0.0000151, 0.0000136
1: -0.0064156, -0.0052854, -0.0063890, -0.0052558, -0.0005649, 0.0005076
2: 0.9687644, 0.9701208, 0.9687964, 0.9701563, -0.0006779, 0.0006091
3: 0.0159176, 0.0259219, 0.0161538, 0.0261836, -0.0050002, 0.0044928
4: -0.0026645, -0.0019037, -0.0026844, -0.0019216, -0.0003417, 0.0003803
5: 0.0145774, 0.0153464, 0.0145572, 0.0153282, -0.0003454, 0.0003844
6: 0.0044423, 0.0048164, 0.0044511, 0.0048261, -0.0001869, 0.0001680
7: -0.0144961, -0.0119034, -0.0145640, -0.0119646, -0.0011644, 0.0012958
8: 0.0052286, 0.0072855, 0.0051748, 0.0072370, -0.0009237, 0.0010281
9: 0.0071288, 0.0108283, 0.0070320, 0.0107410, -0.0016614, 0.0018491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004477
time: 1.05 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004480
time: 0.91 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.0041060, -0.0040758, -0.0041050, -0.0040749, -0.0000154, 0.0000133
1: -0.0064261, -0.0052930, -0.0063878, -0.0052613, -0.0005771, 0.0004996
2: 0.9687519, 0.9701117, 0.9687979, 0.9701496, -0.0006925, 0.0005995
3: 0.0158247, 0.0258548, 0.0161641, 0.0261346, -0.0051079, 0.0044219
4: -0.0026594, -0.0018966, -0.0026807, -0.0019224, -0.0003363, 0.0003885
5: 0.0145825, 0.0153535, 0.0145610, 0.0153274, -0.0003399, 0.0003926
6: 0.0044388, 0.0048139, 0.0044515, 0.0048243, -0.0001910, 0.0001653
7: -0.0144788, -0.0118794, -0.0145513, -0.0119673, -0.0011460, 0.0013238
8: 0.0052424, 0.0073046, 0.0051849, 0.0072348, -0.0009092, 0.0010502
9: 0.0071535, 0.0108627, 0.0070501, 0.0107372, -0.0016352, 0.0018889

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004477
time: 0.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004480
time: 0.85 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041056, -0.0040751, -0.0000144, 0.0000146
1: -0.0063960, -0.0052729, -0.0064096, -0.0052675, -0.0005401, 0.0005485
2: 0.9687881, 0.9701357, 0.9687717, 0.9701422, -0.0006482, 0.0006582
3: 0.0160918, 0.0260320, 0.0159711, 0.0260805, -0.0047810, 0.0048549
4: -0.0026729, -0.0019169, -0.0026766, -0.0019077, -0.0003692, 0.0003636
5: 0.0145689, 0.0153330, 0.0145652, 0.0153423, -0.0003732, 0.0003675
6: 0.0044488, 0.0048205, 0.0044443, 0.0048223, -0.0001788, 0.0001815
7: -0.0145247, -0.0119486, -0.0145372, -0.0119173, -0.0012582, 0.0012390
8: 0.0052059, 0.0072497, 0.0051960, 0.0072745, -0.0009982, 0.0009830
9: 0.0070880, 0.0107639, 0.0070701, 0.0108085, -0.0017953, 0.0017680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004471
time: 0.83 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004472
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -0.0041055, -0.0040754, -0.0041055, -0.0040752, -0.0000148, 0.0000145
1: -0.0064070, -0.0052800, -0.0064085, -0.0052732, -0.0005535, 0.0005417
2: 0.9687748, 0.9701272, 0.9687730, 0.9701354, -0.0006643, 0.0006500
3: 0.0159942, 0.0259694, 0.0159811, 0.0260298, -0.0048994, 0.0047945
4: -0.0026682, -0.0019095, -0.0026727, -0.0019085, -0.0003646, 0.0003726
5: 0.0145737, 0.0153405, 0.0145691, 0.0153415, -0.0003685, 0.0003766
6: 0.0044452, 0.0048181, 0.0044447, 0.0048204, -0.0001832, 0.0001793
7: -0.0145084, -0.0119233, -0.0145241, -0.0119199, -0.0012425, 0.0012697
8: 0.0052188, 0.0072698, 0.0052064, 0.0072725, -0.0009858, 0.0010073
9: 0.0071112, 0.0108000, 0.0070889, 0.0108048, -0.0017730, 0.0018118

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004471
time: 0.82 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004472
time: 0.82 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.0041052, -0.0040752, -0.0041050, -0.0040748, -0.0000146, 0.0000139
1: -0.0063960, -0.0052729, -0.0063890, -0.0052558, -0.0005466, 0.0005210
2: 0.9687881, 0.9701357, 0.9687964, 0.9701563, -0.0006559, 0.0006252
3: 0.0160918, 0.0260320, 0.0161538, 0.0261836, -0.0048378, 0.0046115
4: -0.0026729, -0.0019169, -0.0026844, -0.0019216, -0.0003507, 0.0003679
5: 0.0145689, 0.0153330, 0.0145572, 0.0153282, -0.0003545, 0.0003719
6: 0.0044488, 0.0048205, 0.0044511, 0.0048261, -0.0001809, 0.0001724
7: -0.0145247, -0.0119486, -0.0145640, -0.0119646, -0.0011951, 0.0012538
8: 0.0052059, 0.0072497, 0.0051748, 0.0072370, -0.0009481, 0.0009947
9: 0.0070880, 0.0107639, 0.0070320, 0.0107410, -0.0017053, 0.0017890

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 249
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 189
type: B, layer: 1, pos: 37

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 249

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004731, upper bound: 0.0004559
time: 0.94 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 2, lower bound: -0.0004731, upper bound: 0.0004568
time: 1.19 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.63 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004555, upper bound: 0.0004555
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004558, upper bound: 0.0004566
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004566, upper bound: 0.0004558
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004649, upper bound: 0.0004646
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004649, upper bound: 0.0004653
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004657, upper bound: 0.0004646
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004657, upper bound: 0.0004653
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004522, upper bound: 0.0004484
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004503
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004504
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004503
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004538, upper bound: 0.0004504
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004535, upper bound: 0.0004491
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004606
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004646, upper bound: 0.0004620
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004606
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004653, upper bound: 0.0004620
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004517
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004662, upper bound: 0.0004517
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004663, upper bound: 0.0004516
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004663, upper bound: 0.0004517
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004660, upper bound: 0.0004520
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004660, upper bound: 0.0004521
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004520
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004661, upper bound: 0.0004521
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004521
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004672, upper bound: 0.0004521
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004521
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004674, upper bound: 0.0004521
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004734, upper bound: 0.0004591
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004734, upper bound: 0.0004594
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004745, upper bound: 0.0004591
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004745, upper bound: 0.0004594
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004451
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004627, upper bound: 0.0004450
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004465
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004469
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004465
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004633, upper bound: 0.0004469
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004641, upper bound: 0.0004458
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004556
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004730, upper bound: 0.0004566
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004740, upper bound: 0.0004556
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004740, upper bound: 0.0004566
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004517, upper bound: 0.0004663
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004672
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004674
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004672
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004674
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004660
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004520, upper bound: 0.0004661
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004660
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004521, upper bound: 0.0004661
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004732
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004591, upper bound: 0.0004738
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004731
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004738
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004489, upper bound: 0.0004590
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004613
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004621
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004613
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004498, upper bound: 0.0004621
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004494, upper bound: 0.0004591
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004587, upper bound: 0.0004693
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004587, upper bound: 0.0004710
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004693
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004710
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004673, upper bound: 0.0004537
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004538
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004668, upper bound: 0.0004539
IS_A1_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004669, upper bound: 0.0004538
IS_A1_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004669, upper bound: 0.0004539
IS_A1_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004538
IS_A1_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004679, upper bound: 0.0004538
IS_A1_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004538
IS_A1_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004682, upper bound: 0.0004538
IS_A1_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004595
IS_A1_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004736, upper bound: 0.0004598
IS_A1_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004595
IS_A1_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004747, upper bound: 0.0004597
IS_A1_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
IS_A1_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
IS_A1_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
IS_A1_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004631, upper bound: 0.0004468
IS_A1_B2_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004477
IS_A1_B2_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004480
IS_A1_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004477
IS_A1_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004636, upper bound: 0.0004480
IS_A1_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004471
IS_A1_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004644, upper bound: 0.0004472
IS_A1_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004471
IS_A1_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004472
IS_A1_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004731, upper bound: 0.0004559
IS_A1_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.63
Output dim: 2, lower bound: -0.0004731, upper bound: 0.0004568
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004645, upper bound: 0.0004568
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004610
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004522
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004621
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004535
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004601
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004538
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004681
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004650
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004592
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004484
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004608
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004484, upper bound: 0.0004505
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004580
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004491
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004503, upper bound: 0.0004665
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004504, upper bound: 0.0004620
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004562
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004489
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004563
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004494
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004557
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004498
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004624
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004589
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004548
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004451
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004552
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004589, upper bound: 0.0004469
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004538
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004458
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004613, upper bound: 0.0004611
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004621, upper bound: 0.0004566
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004726
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004627
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004728
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004641
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004703
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004633
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004771
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004733
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004715
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004590
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004717
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004450, upper bound: 0.0004621
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004682
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004591
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004465, upper bound: 0.0004753
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004469, upper bound: 0.0004710
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004570
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004502
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004570
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004507
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004563
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004508
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004626
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004592
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004556
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004468
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004557
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004593, upper bound: 0.0004480
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004545
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004472
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004617, upper bound: 0.0004613
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.63
Output dim: 2, lower bound: -0.0004625, upper bound: 0.0004568

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 2.95 + 597.89 = 600.84 seconds
