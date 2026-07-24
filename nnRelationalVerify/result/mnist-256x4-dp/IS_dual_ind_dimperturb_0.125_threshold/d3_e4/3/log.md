## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00049488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9875610, 0.9898446, 0.9875610, 0.9898446, -0.0017392, 0.0017392)
1: (-0.0043634, -0.0037944, -0.0043634, -0.0037944, -0.0004334, 0.0004334)
2: (0.0100544, 0.0130698, 0.0100544, 0.0130698, -0.0022966, 0.0022966)
3: (-0.0072219, -0.0058495, -0.0072219, -0.0058495, -0.0010453, 0.0010453)
4: (0.0024739, 0.0030575, 0.0024739, 0.0030575, -0.0004445, 0.0004445)
5: (0.0116053, 0.0153978, 0.0116053, 0.0153978, -0.0028885, 0.0028885)
6: (-0.0023673, -0.0014047, -0.0023673, -0.0014047, -0.0007331, 0.0007331)
7: (-0.0092626, -0.0067720, -0.0092626, -0.0067720, -0.0018968, 0.0018968)
8: (-0.0044352, -0.0031255, -0.0044352, -0.0031255, -0.0009975, 0.0009975)
9: (0.0017603, 0.0032790, 0.0017603, 0.0032790, -0.0011567, 0.0011567)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.00 + 1.43 = 3.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0006844, upper bound: 0.0006845

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 167

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006503, upper bound: 0.0006350
time: 0.57 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006503, upper bound: 0.0006503
time: 0.55 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.32 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 0, lower bound: -0.0006503, upper bound: 0.0006350
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.32
Output dim: 0, lower bound: -0.0006503, upper bound: 0.0006503

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.9875754, 0.9897330, 0.9875613, 0.9897987, -0.0016446, 0.0016279
1: -0.0043598, -0.0038222, -0.0043633, -0.0038059, -0.0004098, 0.0004056
2: 0.0102018, 0.0130508, 0.0101150, 0.0130694, -0.0021496, 0.0021717
3: -0.0072133, -0.0059166, -0.0072217, -0.0058770, -0.0009885, 0.0009784
4: 0.0025024, 0.0030538, 0.0024856, 0.0030574, -0.0004161, 0.0004203
5: 0.0117907, 0.0153739, 0.0116815, 0.0153973, -0.0027037, 0.0027315
6: -0.0023612, -0.0014518, -0.0023672, -0.0014241, -0.0006933, 0.0006862
7: -0.0092469, -0.0068938, -0.0092622, -0.0068221, -0.0017937, 0.0017755
8: -0.0044270, -0.0031895, -0.0044351, -0.0031518, -0.0009433, 0.0009337
9: 0.0018346, 0.0032694, 0.0017908, 0.0032788, -0.0010827, 0.0010938

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005934, upper bound: 0.0006030
time: 0.56 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0006031
time: 0.57 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.9875618, 0.9897736, 0.9875613, 0.9898169, -0.0017039, 0.0015965
1: -0.0043632, -0.0038121, -0.0043633, -0.0038013, -0.0004246, 0.0003978
2: 0.0101482, 0.0130688, 0.0100910, 0.0130694, -0.0021081, 0.0022499
3: -0.0072215, -0.0058921, -0.0072218, -0.0058661, -0.0010241, 0.0009595
4: 0.0024920, 0.0030573, 0.0024810, 0.0030574, -0.0004080, 0.0004355
5: 0.0117232, 0.0153965, 0.0116512, 0.0153973, -0.0026514, 0.0028298
6: -0.0023670, -0.0014346, -0.0023672, -0.0014164, -0.0007182, 0.0006730
7: -0.0092617, -0.0068495, -0.0092622, -0.0068022, -0.0018583, 0.0017412
8: -0.0044348, -0.0031662, -0.0044351, -0.0031414, -0.0009773, 0.0009157
9: 0.0018075, 0.0032785, 0.0017787, 0.0032788, -0.0010618, 0.0011332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 179
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 179

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005934, upper bound: 0.0006167
time: 0.57 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006168
time: 0.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.16 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0005934, upper bound: 0.0006030
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0006031
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0005934, upper bound: 0.0006167
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.16
Output dim: 0, lower bound: -0.0006168, upper bound: 0.0006168

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.9875759, 0.9896736, 0.9875555, 0.9896563, -0.0014662, 0.0014850
1: -0.0043597, -0.0038370, -0.0043648, -0.0038413, -0.0003653, 0.0003700
2: 0.0102802, 0.0130501, 0.0103030, 0.0130771, -0.0019609, 0.0019361
3: -0.0072130, -0.0059522, -0.0072253, -0.0059626, -0.0008812, 0.0008925
4: 0.0025176, 0.0030537, 0.0025220, 0.0030589, -0.0003795, 0.0003747
5: 0.0118893, 0.0153731, 0.0119179, 0.0154070, -0.0024663, 0.0024351
6: -0.0023610, -0.0014768, -0.0023696, -0.0014841, -0.0006181, 0.0006260
7: -0.0092463, -0.0069585, -0.0092686, -0.0069773, -0.0015991, 0.0016196
8: -0.0044267, -0.0032236, -0.0044384, -0.0032335, -0.0008409, 0.0008517
9: 0.0018740, 0.0032691, 0.0018855, 0.0032827, -0.0009876, 0.0009751

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0005800
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005749, upper bound: 0.0005852
time: 0.57 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.9875758, 0.9897136, 0.9875624, 0.9897392, -0.0014281, 0.0016151
1: -0.0043597, -0.0038270, -0.0043631, -0.0038206, -0.0003558, 0.0004024
2: 0.0102274, 0.0130504, 0.0101934, 0.0130680, -0.0021327, 0.0018857
3: -0.0072131, -0.0059282, -0.0072211, -0.0059127, -0.0008583, 0.0009707
4: 0.0025074, 0.0030538, 0.0025008, 0.0030572, -0.0004128, 0.0003650
5: 0.0118228, 0.0153734, 0.0117801, 0.0153955, -0.0026824, 0.0023718
6: -0.0023611, -0.0014599, -0.0023667, -0.0014491, -0.0006020, 0.0006808
7: -0.0092465, -0.0069149, -0.0092610, -0.0068868, -0.0015575, 0.0017615
8: -0.0044268, -0.0032006, -0.0044344, -0.0031859, -0.0008191, 0.0009263
9: 0.0018474, 0.0032692, 0.0018303, 0.0032781, -0.0010741, 0.0009498

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0005807
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0006031
time: 0.56 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.9875623, 0.9897138, 0.9875554, 0.9896737, -0.0015239, 0.0014664
1: -0.0043631, -0.0038270, -0.0043648, -0.0038370, -0.0003797, 0.0003654
2: 0.0102270, 0.0130681, 0.0102801, 0.0130771, -0.0019363, 0.0020123
3: -0.0072212, -0.0059280, -0.0072253, -0.0059522, -0.0009159, 0.0008813
4: 0.0025073, 0.0030572, 0.0025176, 0.0030589, -0.0003748, 0.0003895
5: 0.0118223, 0.0153957, 0.0118891, 0.0154070, -0.0024354, 0.0025310
6: -0.0023668, -0.0014598, -0.0023696, -0.0014767, -0.0006424, 0.0006181
7: -0.0092612, -0.0069146, -0.0092686, -0.0069584, -0.0016621, 0.0015993
8: -0.0044345, -0.0032005, -0.0044384, -0.0032235, -0.0008741, 0.0008410
9: 0.0018472, 0.0032782, 0.0018740, 0.0032827, -0.0009752, 0.0010135

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0005927
time: 0.55 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005749, upper bound: 0.0005982
time: 0.60 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.9875622, 0.9897527, 0.9875625, 0.9897558, -0.0014733, 0.0015831
1: -0.0043631, -0.0038173, -0.0043631, -0.0038165, -0.0003671, 0.0003945
2: 0.0101757, 0.0130683, 0.0101716, 0.0130680, -0.0020905, 0.0019454
3: -0.0072213, -0.0059047, -0.0072211, -0.0059028, -0.0008855, 0.0009515
4: 0.0024974, 0.0030572, 0.0024966, 0.0030572, -0.0004046, 0.0003765
5: 0.0117578, 0.0153960, 0.0117527, 0.0153955, -0.0026293, 0.0024468
6: -0.0023668, -0.0014434, -0.0023667, -0.0014421, -0.0006210, 0.0006673
7: -0.0092613, -0.0068722, -0.0092610, -0.0068688, -0.0016068, 0.0017266
8: -0.0044346, -0.0031782, -0.0044344, -0.0031764, -0.0008450, 0.0009080
9: 0.0018214, 0.0032783, 0.0018193, 0.0032781, -0.0010529, 0.0009798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=10, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0005934
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0006168
time: 0.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.07 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0005800
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0005749, upper bound: 0.0005852
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0005807
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0006031
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0005477, upper bound: 0.0005927
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0005749, upper bound: 0.0005982
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0005934
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.07
Output dim: 0, lower bound: -0.0006167, upper bound: 0.0006168

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876693, 0.9896546, 0.9875960, 0.9896551, -0.0013485, 0.0013644
1: -0.0043364, -0.0038418, -0.0043547, -0.0038416, -0.0003360, 0.0003400
2: 0.0103053, 0.0129267, 0.0103045, 0.0130238, -0.0018016, 0.0017807
3: -0.0071568, -0.0059637, -0.0072010, -0.0059633, -0.0008105, 0.0008200
4: 0.0025225, 0.0030298, 0.0025223, 0.0030486, -0.0003487, 0.0003446
5: 0.0119208, 0.0152179, 0.0119198, 0.0153399, -0.0022660, 0.0022396
6: -0.0023216, -0.0014848, -0.0023526, -0.0014845, -0.0005684, 0.0005751
7: -0.0091444, -0.0069793, -0.0092245, -0.0069786, -0.0014707, 0.0014880
8: -0.0043731, -0.0032345, -0.0044152, -0.0032341, -0.0007734, 0.0007825
9: 0.0018867, 0.0032070, 0.0018862, 0.0032558, -0.0009074, 0.0008968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
time: 0.55 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876173, 0.9896721, 0.9875706, 0.9896559, -0.0013049, 0.0014418
1: -0.0043494, -0.0038374, -0.0043610, -0.0038414, -0.0003251, 0.0003593
2: 0.0102821, 0.0129955, 0.0103037, 0.0130572, -0.0019039, 0.0017231
3: -0.0071881, -0.0059531, -0.0072162, -0.0059629, -0.0007843, 0.0008666
4: 0.0025180, 0.0030431, 0.0025221, 0.0030551, -0.0003685, 0.0003335
5: 0.0118916, 0.0153043, 0.0119187, 0.0153819, -0.0023945, 0.0021672
6: -0.0023436, -0.0014774, -0.0023633, -0.0014843, -0.0005501, 0.0006078
7: -0.0092012, -0.0069601, -0.0092521, -0.0069779, -0.0014231, 0.0015725
8: -0.0044029, -0.0032244, -0.0044297, -0.0032338, -0.0007484, 0.0008269
9: 0.0018750, 0.0032416, 0.0018858, 0.0032726, -0.0009589, 0.0008678

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005851
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005852
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9875759, 0.9895904, 0.9875624, 0.9897392, -0.0015599, 0.0014560
1: -0.0043597, -0.0038577, -0.0043631, -0.0038206, -0.0003887, 0.0003628
2: 0.0103900, 0.0130502, 0.0101934, 0.0130680, -0.0019227, 0.0020598
3: -0.0072130, -0.0060022, -0.0072211, -0.0059127, -0.0009375, 0.0008751
4: 0.0025389, 0.0030537, 0.0025008, 0.0030572, -0.0003721, 0.0003987
5: 0.0120274, 0.0153731, 0.0117801, 0.0153955, -0.0024182, 0.0025907
6: -0.0023610, -0.0015118, -0.0023667, -0.0014491, -0.0006575, 0.0006138
7: -0.0092463, -0.0070492, -0.0092610, -0.0068868, -0.0017013, 0.0015880
8: -0.0044267, -0.0032713, -0.0044344, -0.0031859, -0.0008947, 0.0008351
9: 0.0019293, 0.0032691, 0.0018303, 0.0032781, -0.0009684, 0.0010374

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005807
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005807
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9875764, 0.9896760, 0.9875624, 0.9897392, -0.0014274, 0.0014056
1: -0.0043596, -0.0038364, -0.0043631, -0.0038206, -0.0003557, 0.0003502
2: 0.0102771, 0.0130495, 0.0101934, 0.0130680, -0.0018561, 0.0018849
3: -0.0072127, -0.0059508, -0.0072211, -0.0059127, -0.0008579, 0.0008448
4: 0.0025170, 0.0030536, 0.0025008, 0.0030572, -0.0003593, 0.0003648
5: 0.0118853, 0.0153723, 0.0117801, 0.0153955, -0.0023345, 0.0023707
6: -0.0023608, -0.0014758, -0.0023667, -0.0014491, -0.0006017, 0.0005925
7: -0.0092458, -0.0069560, -0.0092610, -0.0068868, -0.0015568, 0.0015330
8: -0.0044264, -0.0032222, -0.0044344, -0.0031859, -0.0008187, 0.0008062
9: 0.0018725, 0.0032688, 0.0018303, 0.0032781, -0.0009348, 0.0009494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006031
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006030
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876568, 0.9896931, 0.9875959, 0.9896726, -0.0014150, 0.0013380
1: -0.0043396, -0.0038322, -0.0043547, -0.0038373, -0.0003526, 0.0003334
2: 0.0102544, 0.0129435, 0.0102815, 0.0130238, -0.0017668, 0.0018684
3: -0.0071644, -0.0059405, -0.0072010, -0.0059528, -0.0008504, 0.0008042
4: 0.0025126, 0.0030331, 0.0025178, 0.0030486, -0.0003420, 0.0003616
5: 0.0118569, 0.0152389, 0.0118908, 0.0153399, -0.0022222, 0.0023500
6: -0.0023270, -0.0014686, -0.0023526, -0.0014772, -0.0005965, 0.0005640
7: -0.0091582, -0.0069372, -0.0092245, -0.0069596, -0.0015432, 0.0014593
8: -0.0043804, -0.0032124, -0.0044152, -0.0032241, -0.0008116, 0.0007674
9: 0.0018611, 0.0032154, 0.0018747, 0.0032558, -0.0008899, 0.0009410

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005361, upper bound: 0.0005928
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005361, upper bound: 0.0005928
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876037, 0.9897126, 0.9875706, 0.9896732, -0.0013855, 0.0014303
1: -0.0043528, -0.0038273, -0.0043610, -0.0038371, -0.0003452, 0.0003564
2: 0.0102287, 0.0130135, 0.0102807, 0.0130572, -0.0018886, 0.0018295
3: -0.0071963, -0.0059288, -0.0072162, -0.0059525, -0.0008327, 0.0008596
4: 0.0025076, 0.0030466, 0.0025177, 0.0030551, -0.0003655, 0.0003541
5: 0.0118245, 0.0153271, 0.0118899, 0.0153819, -0.0023754, 0.0023011
6: -0.0023493, -0.0014604, -0.0023633, -0.0014770, -0.0005840, 0.0006029
7: -0.0092161, -0.0069160, -0.0092521, -0.0069589, -0.0015111, 0.0015599
8: -0.0044108, -0.0032012, -0.0044297, -0.0032238, -0.0007947, 0.0008203
9: 0.0018481, 0.0032507, 0.0018743, 0.0032727, -0.0009512, 0.0009214

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005982
time: 0.58 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005981
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9875560, 0.9896292, 0.9875625, 0.9897558, -0.0016076, 0.0014251
1: -0.0043647, -0.0038481, -0.0043631, -0.0038165, -0.0004006, 0.0003551
2: 0.0103389, 0.0130765, 0.0101716, 0.0130680, -0.0018818, 0.0021228
3: -0.0072250, -0.0059789, -0.0072211, -0.0059028, -0.0009662, 0.0008565
4: 0.0025290, 0.0030588, 0.0024966, 0.0030572, -0.0003642, 0.0004109
5: 0.0119630, 0.0154062, 0.0117527, 0.0153955, -0.0023668, 0.0026699
6: -0.0023694, -0.0014955, -0.0023667, -0.0014421, -0.0006776, 0.0006007
7: -0.0092681, -0.0070070, -0.0092610, -0.0068688, -0.0017533, 0.0015543
8: -0.0044381, -0.0032490, -0.0044344, -0.0031764, -0.0009220, 0.0008174
9: 0.0019036, 0.0032824, 0.0018193, 0.0032781, -0.0009478, 0.0010691

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005934
time: 0.62 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005934
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9875629, 0.9897111, 0.9875625, 0.9897558, -0.0014726, 0.0013886
1: -0.0043630, -0.0038277, -0.0043631, -0.0038165, -0.0003669, 0.0003460
2: 0.0102307, 0.0130674, 0.0101716, 0.0130680, -0.0018336, 0.0019446
3: -0.0072208, -0.0059297, -0.0072211, -0.0059028, -0.0008851, 0.0008346
4: 0.0025080, 0.0030570, 0.0024966, 0.0030572, -0.0003549, 0.0003764
5: 0.0118270, 0.0153947, 0.0117527, 0.0153955, -0.0023061, 0.0024458
6: -0.0023665, -0.0014610, -0.0023667, -0.0014421, -0.0006208, 0.0005853
7: -0.0092605, -0.0069176, -0.0092610, -0.0068688, -0.0016061, 0.0015144
8: -0.0044342, -0.0032021, -0.0044344, -0.0031764, -0.0008446, 0.0007964
9: 0.0018491, 0.0032778, 0.0018193, 0.0032781, -0.0009235, 0.0009794

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=10, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 167

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006168
time: 0.57 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006168
time: 0.64 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.13 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005851
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005852
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005807
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005807
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006031
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006030
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005361, upper bound: 0.0005928
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005361, upper bound: 0.0005928
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005982
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005981
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005934
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0005934
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006168
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.13
Output dim: 0, lower bound: -0.0005807, upper bound: 0.0006168

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876693, 0.9896546, 0.9876131, 0.9895893, -0.0012908, 0.0013216
1: -0.0043364, -0.0038418, -0.0043505, -0.0038580, -0.0003216, 0.0003293
2: 0.0103053, 0.0129267, 0.0103915, 0.0130012, -0.0017451, 0.0017045
3: -0.0071568, -0.0059637, -0.0071907, -0.0060029, -0.0007758, 0.0007943
4: 0.0025225, 0.0030298, 0.0025391, 0.0030442, -0.0003378, 0.0003299
5: 0.0119208, 0.0152179, 0.0120292, 0.0153115, -0.0021949, 0.0021438
6: -0.0023216, -0.0014848, -0.0023454, -0.0015123, -0.0005441, 0.0005571
7: -0.0091444, -0.0069793, -0.0092058, -0.0070504, -0.0014078, 0.0014413
8: -0.0043731, -0.0032345, -0.0044054, -0.0032719, -0.0007404, 0.0007580
9: 0.0018867, 0.0032070, 0.0019301, 0.0032444, -0.0008789, 0.0008585

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005601
time: 0.57 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
time: 0.56 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876693, 0.9896546, 0.9875963, 0.9896281, -0.0013273, 0.0013583
1: -0.0043364, -0.0038418, -0.0043546, -0.0038484, -0.0003307, 0.0003385
2: 0.0103053, 0.0129267, 0.0103403, 0.0130232, -0.0017937, 0.0017527
3: -0.0071568, -0.0059637, -0.0072007, -0.0059796, -0.0007977, 0.0008164
4: 0.0025225, 0.0030298, 0.0025292, 0.0030485, -0.0003472, 0.0003392
5: 0.0119208, 0.0152179, 0.0119648, 0.0153392, -0.0022560, 0.0022044
6: -0.0023216, -0.0014848, -0.0023524, -0.0014960, -0.0005595, 0.0005726
7: -0.0091444, -0.0069793, -0.0092241, -0.0070081, -0.0014476, 0.0014815
8: -0.0043731, -0.0032345, -0.0044150, -0.0032496, -0.0007613, 0.0007791
9: 0.0018867, 0.0032070, 0.0019043, 0.0032556, -0.0009034, 0.0008827

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005601
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876173, 0.9896721, 0.9875925, 0.9895899, -0.0012331, 0.0014126
1: -0.0043494, -0.0038374, -0.0043556, -0.0038579, -0.0003072, 0.0003520
2: 0.0102821, 0.0129955, 0.0103907, 0.0130282, -0.0018654, 0.0016282
3: -0.0071881, -0.0059531, -0.0072030, -0.0060025, -0.0007411, 0.0008490
4: 0.0025180, 0.0030431, 0.0025390, 0.0030495, -0.0003610, 0.0003151
5: 0.0118916, 0.0153043, 0.0120282, 0.0153455, -0.0023461, 0.0020479
6: -0.0023436, -0.0014774, -0.0023540, -0.0015121, -0.0005198, 0.0005955
7: -0.0092012, -0.0069601, -0.0092282, -0.0070498, -0.0013448, 0.0015407
8: -0.0044029, -0.0032244, -0.0044172, -0.0032716, -0.0007072, 0.0008102
9: 0.0018750, 0.0032416, 0.0019297, 0.0032581, -0.0009395, 0.0008201

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005653
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005852
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876173, 0.9896721, 0.9875711, 0.9896287, -0.0013022, 0.0014365
1: -0.0043494, -0.0038374, -0.0043609, -0.0038482, -0.0003245, 0.0003579
2: 0.0102821, 0.0129955, 0.0103395, 0.0130566, -0.0018969, 0.0017195
3: -0.0071881, -0.0059531, -0.0072159, -0.0059792, -0.0007826, 0.0008634
4: 0.0025180, 0.0030431, 0.0025291, 0.0030550, -0.0003671, 0.0003328
5: 0.0118916, 0.0153043, 0.0119638, 0.0153812, -0.0023858, 0.0021627
6: -0.0023436, -0.0014774, -0.0023631, -0.0014957, -0.0005489, 0.0006056
7: -0.0092012, -0.0069601, -0.0092516, -0.0070075, -0.0014202, 0.0015667
8: -0.0044029, -0.0032244, -0.0044295, -0.0032493, -0.0007469, 0.0008239
9: 0.0018750, 0.0032416, 0.0019039, 0.0032723, -0.0009554, 0.0008660

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005653
time: 0.73 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005852
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9875759, 0.9895904, 0.9875764, 0.9896760, -0.0014923, 0.0014079
1: -0.0043597, -0.0038577, -0.0043596, -0.0038364, -0.0003718, 0.0003508
2: 0.0103900, 0.0130502, 0.0102771, 0.0130495, -0.0018591, 0.0019705
3: -0.0072130, -0.0060022, -0.0072127, -0.0059508, -0.0008969, 0.0008462
4: 0.0025389, 0.0030537, 0.0025170, 0.0030536, -0.0003598, 0.0003814
5: 0.0120274, 0.0153731, 0.0118853, 0.0153723, -0.0023382, 0.0024784
6: -0.0023610, -0.0015118, -0.0023608, -0.0014758, -0.0006290, 0.0005935
7: -0.0092463, -0.0070492, -0.0092458, -0.0069560, -0.0016275, 0.0015355
8: -0.0044267, -0.0032713, -0.0044264, -0.0032222, -0.0008559, 0.0008075
9: 0.0019293, 0.0032691, 0.0018725, 0.0032688, -0.0009363, 0.0009924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005476, upper bound: 0.0005565
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005848, upper bound: 0.0005618
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9875759, 0.9895904, 0.9875629, 0.9897111, -0.0015361, 0.0014510
1: -0.0043597, -0.0038577, -0.0043630, -0.0038277, -0.0003828, 0.0003615
2: 0.0103900, 0.0130502, 0.0102307, 0.0130674, -0.0019160, 0.0020284
3: -0.0072130, -0.0060022, -0.0072208, -0.0059297, -0.0009232, 0.0008721
4: 0.0025389, 0.0030537, 0.0025080, 0.0030570, -0.0003708, 0.0003926
5: 0.0120274, 0.0153731, 0.0118270, 0.0153947, -0.0024099, 0.0025512
6: -0.0023610, -0.0015118, -0.0023665, -0.0014610, -0.0006475, 0.0006116
7: -0.0092463, -0.0070492, -0.0092605, -0.0069176, -0.0016753, 0.0015825
8: -0.0044267, -0.0032713, -0.0044342, -0.0032021, -0.0008810, 0.0008322
9: 0.0019293, 0.0032691, 0.0018491, 0.0032778, -0.0009650, 0.0010216

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005476, upper bound: 0.0005564
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005848, upper bound: 0.0005618
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9875764, 0.9896760, 0.9875764, 0.9896760, -0.0013697, 0.0013697
1: -0.0043596, -0.0038364, -0.0043596, -0.0038364, -0.0003413, 0.0003413
2: 0.0102771, 0.0130495, 0.0102771, 0.0130495, -0.0018086, 0.0018086
3: -0.0072127, -0.0059508, -0.0072127, -0.0059508, -0.0008232, 0.0008232
4: 0.0025170, 0.0030536, 0.0025170, 0.0030536, -0.0003501, 0.0003501
5: 0.0118853, 0.0153723, 0.0118853, 0.0153723, -0.0022748, 0.0022748
6: -0.0023608, -0.0014758, -0.0023608, -0.0014758, -0.0005774, 0.0005774
7: -0.0092458, -0.0069560, -0.0092458, -0.0069560, -0.0014938, 0.0014938
8: -0.0044264, -0.0032222, -0.0044264, -0.0032222, -0.0007856, 0.0007856
9: 0.0018725, 0.0032688, 0.0018725, 0.0032688, -0.0009109, 0.0009109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005373, upper bound: 0.0005800
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005619, upper bound: 0.0005852
time: 0.58 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9875764, 0.9896760, 0.9875629, 0.9897111, -0.0014070, 0.0014003
1: -0.0043596, -0.0038364, -0.0043630, -0.0038277, -0.0003506, 0.0003489
2: 0.0102771, 0.0130495, 0.0102307, 0.0130674, -0.0018491, 0.0018579
3: -0.0072127, -0.0059508, -0.0072208, -0.0059297, -0.0008456, 0.0008416
4: 0.0025170, 0.0030536, 0.0025080, 0.0030570, -0.0003579, 0.0003596
5: 0.0118853, 0.0153723, 0.0118270, 0.0153947, -0.0023257, 0.0023367
6: -0.0023608, -0.0014758, -0.0023665, -0.0014610, -0.0005931, 0.0005903
7: -0.0092458, -0.0069560, -0.0092605, -0.0069176, -0.0015345, 0.0015273
8: -0.0044264, -0.0032222, -0.0044342, -0.0032021, -0.0008070, 0.0008032
9: 0.0018725, 0.0032688, 0.0018491, 0.0032778, -0.0009313, 0.0009357

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005373, upper bound: 0.0005800
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005619, upper bound: 0.0005852
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876568, 0.9896931, 0.9876131, 0.9895893, -0.0013424, 0.0013869
1: -0.0043396, -0.0038322, -0.0043505, -0.0038580, -0.0003345, 0.0003456
2: 0.0102544, 0.0129435, 0.0103915, 0.0130012, -0.0018313, 0.0017727
3: -0.0071644, -0.0059405, -0.0071907, -0.0060029, -0.0008068, 0.0008335
4: 0.0025126, 0.0030331, 0.0025391, 0.0030442, -0.0003545, 0.0003431
5: 0.0118569, 0.0152389, 0.0120292, 0.0153115, -0.0023033, 0.0022295
6: -0.0023270, -0.0014686, -0.0023454, -0.0015123, -0.0005659, 0.0005846
7: -0.0091582, -0.0069372, -0.0092058, -0.0070504, -0.0014641, 0.0015126
8: -0.0043804, -0.0032124, -0.0044054, -0.0032719, -0.0007700, 0.0007954
9: 0.0018611, 0.0032154, 0.0019301, 0.0032444, -0.0009224, 0.0008928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005739
time: 0.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005928
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876568, 0.9896931, 0.9875963, 0.9896281, -0.0013052, 0.0013351
1: -0.0043396, -0.0038322, -0.0043546, -0.0038484, -0.0003252, 0.0003327
2: 0.0102544, 0.0129435, 0.0103403, 0.0130232, -0.0017630, 0.0017235
3: -0.0071644, -0.0059405, -0.0072007, -0.0059796, -0.0007845, 0.0008024
4: 0.0025126, 0.0030331, 0.0025292, 0.0030485, -0.0003412, 0.0003336
5: 0.0118569, 0.0152389, 0.0119648, 0.0153392, -0.0022174, 0.0021677
6: -0.0023270, -0.0014686, -0.0023524, -0.0014960, -0.0005502, 0.0005628
7: -0.0091582, -0.0069372, -0.0092241, -0.0070081, -0.0014235, 0.0014561
8: -0.0043804, -0.0032124, -0.0044150, -0.0032496, -0.0007486, 0.0007658
9: 0.0018611, 0.0032154, 0.0019043, 0.0032556, -0.0008879, 0.0008680

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005739
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005927
time: 0.56 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876037, 0.9897126, 0.9875925, 0.9895899, -0.0012871, 0.0014544
1: -0.0043528, -0.0038273, -0.0043556, -0.0038579, -0.0003207, 0.0003624
2: 0.0102287, 0.0130135, 0.0103907, 0.0130282, -0.0019205, 0.0016996
3: -0.0071963, -0.0059288, -0.0072030, -0.0060025, -0.0007736, 0.0008741
4: 0.0025076, 0.0030466, 0.0025390, 0.0030495, -0.0003717, 0.0003290
5: 0.0118245, 0.0153271, 0.0120282, 0.0153455, -0.0024155, 0.0021376
6: -0.0023493, -0.0014604, -0.0023540, -0.0015121, -0.0005426, 0.0006131
7: -0.0092161, -0.0069160, -0.0092282, -0.0070498, -0.0014037, 0.0015862
8: -0.0044108, -0.0032012, -0.0044172, -0.0032716, -0.0007382, 0.0008342
9: 0.0018481, 0.0032507, 0.0019297, 0.0032581, -0.0009673, 0.0008560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005792
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005982
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876037, 0.9897126, 0.9875711, 0.9896287, -0.0012494, 0.0014277
1: -0.0043528, -0.0038273, -0.0043609, -0.0038482, -0.0003113, 0.0003557
2: 0.0102287, 0.0130135, 0.0103395, 0.0130566, -0.0018852, 0.0016498
3: -0.0071963, -0.0059288, -0.0072159, -0.0059792, -0.0007509, 0.0008581
4: 0.0025076, 0.0030466, 0.0025291, 0.0030550, -0.0003649, 0.0003193
5: 0.0118245, 0.0153271, 0.0119638, 0.0153812, -0.0023711, 0.0020750
6: -0.0023493, -0.0014604, -0.0023631, -0.0014957, -0.0005267, 0.0006018
7: -0.0092161, -0.0069160, -0.0092516, -0.0070075, -0.0013626, 0.0015571
8: -0.0044108, -0.0032012, -0.0044295, -0.0032493, -0.0007166, 0.0008188
9: 0.0018481, 0.0032507, 0.0019039, 0.0032723, -0.0009495, 0.0008309

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 179
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 179

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005792
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005982
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9875560, 0.9896292, 0.9875764, 0.9896760, -0.0015230, 0.0014456
1: -0.0043647, -0.0038481, -0.0043596, -0.0038364, -0.0003795, 0.0003602
2: 0.0103389, 0.0130765, 0.0102771, 0.0130495, -0.0019089, 0.0020111
3: -0.0072250, -0.0059789, -0.0072127, -0.0059508, -0.0009154, 0.0008689
4: 0.0025290, 0.0030588, 0.0025170, 0.0030536, -0.0003695, 0.0003893
5: 0.0119630, 0.0154062, 0.0118853, 0.0153723, -0.0024009, 0.0025295
6: -0.0023694, -0.0014955, -0.0023608, -0.0014758, -0.0006420, 0.0006094
7: -0.0092681, -0.0070070, -0.0092458, -0.0069560, -0.0016611, 0.0015766
8: -0.0044381, -0.0032490, -0.0044264, -0.0032222, -0.0008735, 0.0008291
9: 0.0019036, 0.0032824, 0.0018725, 0.0032688, -0.0009614, 0.0010129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005461, upper bound: 0.0005696
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005749
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9875560, 0.9896292, 0.9875629, 0.9897111, -0.0015063, 0.0014221
1: -0.0043647, -0.0038481, -0.0043630, -0.0038277, -0.0003753, 0.0003544
2: 0.0103389, 0.0130765, 0.0102307, 0.0130674, -0.0018779, 0.0019890
3: -0.0072250, -0.0059789, -0.0072208, -0.0059297, -0.0009053, 0.0008547
4: 0.0025290, 0.0030588, 0.0025080, 0.0030570, -0.0003635, 0.0003850
5: 0.0119630, 0.0154062, 0.0118270, 0.0153947, -0.0023619, 0.0025017
6: -0.0023694, -0.0014955, -0.0023665, -0.0014610, -0.0006350, 0.0005995
7: -0.0092681, -0.0070070, -0.0092605, -0.0069176, -0.0016428, 0.0015510
8: -0.0044381, -0.0032490, -0.0044342, -0.0032021, -0.0008639, 0.0008157
9: 0.0019036, 0.0032824, 0.0018491, 0.0032778, -0.0009458, 0.0010018

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005461, upper bound: 0.0005696
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005749
time: 0.58 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9875629, 0.9897111, 0.9875764, 0.9896760, -0.0014003, 0.0014070
1: -0.0043630, -0.0038277, -0.0043596, -0.0038364, -0.0003489, 0.0003506
2: 0.0102307, 0.0130674, 0.0102771, 0.0130495, -0.0018579, 0.0018491
3: -0.0072208, -0.0059297, -0.0072127, -0.0059508, -0.0008416, 0.0008456
4: 0.0025080, 0.0030570, 0.0025170, 0.0030536, -0.0003596, 0.0003579
5: 0.0118270, 0.0153947, 0.0118853, 0.0153723, -0.0023367, 0.0023257
6: -0.0023665, -0.0014610, -0.0023608, -0.0014758, -0.0005903, 0.0005931
7: -0.0092605, -0.0069176, -0.0092458, -0.0069560, -0.0015273, 0.0015345
8: -0.0044342, -0.0032021, -0.0044264, -0.0032222, -0.0008032, 0.0008070
9: 0.0018491, 0.0032778, 0.0018725, 0.0032688, -0.0009357, 0.0009313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005364, upper bound: 0.0005928
time: 0.58 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005982
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9875629, 0.9897111, 0.9875629, 0.9897111, -0.0013859, 0.0013859
1: -0.0043630, -0.0038277, -0.0043630, -0.0038277, -0.0003453, 0.0003453
2: 0.0102307, 0.0130674, 0.0102307, 0.0130674, -0.0018301, 0.0018301
3: -0.0072208, -0.0059297, -0.0072208, -0.0059297, -0.0008330, 0.0008330
4: 0.0025080, 0.0030570, 0.0025080, 0.0030570, -0.0003542, 0.0003542
5: 0.0118270, 0.0153947, 0.0118270, 0.0153947, -0.0023018, 0.0023018
6: -0.0023665, -0.0014610, -0.0023665, -0.0014610, -0.0005842, 0.0005842
7: -0.0092605, -0.0069176, -0.0092605, -0.0069176, -0.0015116, 0.0015116
8: -0.0044342, -0.0032021, -0.0044342, -0.0032021, -0.0007949, 0.0007949
9: 0.0018491, 0.0032778, 0.0018491, 0.0032778, -0.0009217, 0.0009217

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 69
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 69

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005364, upper bound: 0.0005928
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005982
time: 0.60 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.24 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005601
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005601
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005371, upper bound: 0.0005800
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005653
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005852
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005653
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005852
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005476, upper bound: 0.0005565
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005848, upper bound: 0.0005618
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005476, upper bound: 0.0005564
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005848, upper bound: 0.0005618
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005373, upper bound: 0.0005800
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005619, upper bound: 0.0005852
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005373, upper bound: 0.0005800
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005619, upper bound: 0.0005852
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005739
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005928
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005739
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005362, upper bound: 0.0005927
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005792
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005982
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005792
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005617, upper bound: 0.0005982
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005461, upper bound: 0.0005696
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005749
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005461, upper bound: 0.0005696
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005847, upper bound: 0.0005749
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005364, upper bound: 0.0005928
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005982
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005364, upper bound: 0.0005928
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.24
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005982

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876652, 0.9895737, 0.9876131, 0.9895893, -0.0012376, 0.0012221
1: -0.0043375, -0.0038619, -0.0043505, -0.0038580, -0.0003084, 0.0003045
2: 0.0104120, 0.0129322, 0.0103915, 0.0130012, -0.0016137, 0.0016342
3: -0.0071593, -0.0060122, -0.0071907, -0.0060029, -0.0007438, 0.0007345
4: 0.0025431, 0.0030309, 0.0025391, 0.0030442, -0.0003123, 0.0003163
5: 0.0120550, 0.0152248, 0.0120292, 0.0153115, -0.0020296, 0.0020554
6: -0.0023234, -0.0015189, -0.0023454, -0.0015123, -0.0005217, 0.0005151
7: -0.0091489, -0.0070674, -0.0092058, -0.0070504, -0.0013497, 0.0013328
8: -0.0043755, -0.0032808, -0.0044054, -0.0032719, -0.0007098, 0.0007009
9: 0.0019404, 0.0032097, 0.0019301, 0.0032444, -0.0008128, 0.0008231

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005162, upper bound: 0.0005384
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005149, upper bound: 0.0005384
time: 0.57 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876700, 0.9896523, 0.9876131, 0.9895893, -0.0012903, 0.0013627
1: -0.0043363, -0.0038423, -0.0043505, -0.0038580, -0.0003215, 0.0003396
2: 0.0103083, 0.0129259, 0.0103915, 0.0130012, -0.0017994, 0.0017038
3: -0.0071564, -0.0059650, -0.0071907, -0.0060029, -0.0007755, 0.0008190
4: 0.0025230, 0.0030297, 0.0025391, 0.0030442, -0.0003483, 0.0003298
5: 0.0119245, 0.0152168, 0.0120292, 0.0153115, -0.0022632, 0.0021430
6: -0.0023214, -0.0014857, -0.0023454, -0.0015123, -0.0005439, 0.0005744
7: -0.0091437, -0.0069817, -0.0092058, -0.0070504, -0.0014073, 0.0014862
8: -0.0043727, -0.0032357, -0.0044054, -0.0032719, -0.0007401, 0.0007816
9: 0.0018882, 0.0032065, 0.0019301, 0.0032444, -0.0009063, 0.0008581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005162, upper bound: 0.0005628
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005149, upper bound: 0.0005628
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876652, 0.9895737, 0.9875963, 0.9896281, -0.0012741, 0.0012588
1: -0.0043375, -0.0038619, -0.0043546, -0.0038484, -0.0003175, 0.0003137
2: 0.0104120, 0.0129322, 0.0103403, 0.0130232, -0.0016623, 0.0016824
3: -0.0071593, -0.0060122, -0.0072007, -0.0059796, -0.0007657, 0.0007566
4: 0.0025431, 0.0030309, 0.0025292, 0.0030485, -0.0003217, 0.0003256
5: 0.0120550, 0.0152248, 0.0119648, 0.0153392, -0.0020907, 0.0021160
6: -0.0023234, -0.0015189, -0.0023524, -0.0014960, -0.0005371, 0.0005306
7: -0.0091489, -0.0070674, -0.0092241, -0.0070081, -0.0013895, 0.0013729
8: -0.0043755, -0.0032808, -0.0044150, -0.0032496, -0.0007307, 0.0007220
9: 0.0019404, 0.0032097, 0.0019043, 0.0032556, -0.0008372, 0.0008473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005383
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005256, upper bound: 0.0005384
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876700, 0.9896523, 0.9875963, 0.9896281, -0.0013268, 0.0013995
1: -0.0043363, -0.0038423, -0.0043546, -0.0038484, -0.0003306, 0.0003487
2: 0.0103083, 0.0129259, 0.0103403, 0.0130232, -0.0018480, 0.0017520
3: -0.0071564, -0.0059650, -0.0072007, -0.0059796, -0.0007974, 0.0008411
4: 0.0025230, 0.0030297, 0.0025292, 0.0030485, -0.0003577, 0.0003391
5: 0.0119245, 0.0152168, 0.0119648, 0.0153392, -0.0023243, 0.0022036
6: -0.0023214, -0.0014857, -0.0023524, -0.0014960, -0.0005593, 0.0005899
7: -0.0091437, -0.0069817, -0.0092241, -0.0070081, -0.0014470, 0.0015263
8: -0.0043727, -0.0032357, -0.0044150, -0.0032496, -0.0007610, 0.0008027
9: 0.0018882, 0.0032065, 0.0019043, 0.0032556, -0.0009308, 0.0008824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005627
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005256, upper bound: 0.0005627
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876190, 0.9895890, 0.9875925, 0.9895899, -0.0011830, 0.0013172
1: -0.0043490, -0.0038581, -0.0043556, -0.0038579, -0.0002948, 0.0003282
2: 0.0103919, 0.0129933, 0.0103907, 0.0130282, -0.0017394, 0.0015621
3: -0.0071871, -0.0060031, -0.0072030, -0.0060025, -0.0007110, 0.0007917
4: 0.0025392, 0.0030427, 0.0025390, 0.0030495, -0.0003367, 0.0003023
5: 0.0120297, 0.0153016, 0.0120282, 0.0153455, -0.0021877, 0.0019647
6: -0.0023429, -0.0015124, -0.0023540, -0.0015121, -0.0004987, 0.0005553
7: -0.0091993, -0.0070508, -0.0092282, -0.0070498, -0.0012902, 0.0014366
8: -0.0044020, -0.0032721, -0.0044172, -0.0032716, -0.0006785, 0.0007555
9: 0.0019303, 0.0032405, 0.0019297, 0.0032581, -0.0008760, 0.0007867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005435
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005436
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876177, 0.9896747, 0.9875925, 0.9895899, -0.0012327, 0.0014564
1: -0.0043493, -0.0038367, -0.0043556, -0.0038579, -0.0003072, 0.0003629
2: 0.0102787, 0.0129950, 0.0103907, 0.0130282, -0.0019232, 0.0016278
3: -0.0071879, -0.0059516, -0.0072030, -0.0060025, -0.0007409, 0.0008754
4: 0.0025173, 0.0030430, 0.0025390, 0.0030495, -0.0003722, 0.0003151
5: 0.0118874, 0.0153037, 0.0120282, 0.0153455, -0.0024189, 0.0020473
6: -0.0023434, -0.0014763, -0.0023540, -0.0015121, -0.0005196, 0.0006139
7: -0.0092007, -0.0069573, -0.0092282, -0.0070498, -0.0013445, 0.0015885
8: -0.0044027, -0.0032229, -0.0044172, -0.0032716, -0.0007070, 0.0008354
9: 0.0018733, 0.0032413, 0.0019297, 0.0032581, -0.0009686, 0.0008198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005679
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005679
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876190, 0.9895890, 0.9875711, 0.9896287, -0.0012521, 0.0013411
1: -0.0043490, -0.0038581, -0.0043609, -0.0038482, -0.0003120, 0.0003342
2: 0.0103919, 0.0129933, 0.0103395, 0.0130566, -0.0017709, 0.0016533
3: -0.0071871, -0.0060031, -0.0072159, -0.0059792, -0.0007525, 0.0008061
4: 0.0025392, 0.0030427, 0.0025291, 0.0030550, -0.0003428, 0.0003200
5: 0.0120297, 0.0153016, 0.0119638, 0.0153812, -0.0022274, 0.0020794
6: -0.0023429, -0.0015124, -0.0023631, -0.0014957, -0.0005278, 0.0005653
7: -0.0091993, -0.0070508, -0.0092516, -0.0070075, -0.0013655, 0.0014627
8: -0.0044020, -0.0032721, -0.0044295, -0.0032493, -0.0007181, 0.0007692
9: 0.0019303, 0.0032405, 0.0019039, 0.0032723, -0.0008919, 0.0008327

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005540, upper bound: 0.0005434
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0005436
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876177, 0.9896747, 0.9875711, 0.9896287, -0.0013018, 0.0014803
1: -0.0043493, -0.0038367, -0.0043609, -0.0038482, -0.0003244, 0.0003689
2: 0.0102787, 0.0129950, 0.0103395, 0.0130566, -0.0019548, 0.0017190
3: -0.0071879, -0.0059516, -0.0072159, -0.0059792, -0.0007824, 0.0008897
4: 0.0025173, 0.0030430, 0.0025291, 0.0030550, -0.0003783, 0.0003327
5: 0.0118874, 0.0153037, 0.0119638, 0.0153812, -0.0024586, 0.0021621
6: -0.0023434, -0.0014763, -0.0023631, -0.0014957, -0.0005488, 0.0006240
7: -0.0092007, -0.0069573, -0.0092516, -0.0070075, -0.0014198, 0.0016145
8: -0.0044027, -0.0032229, -0.0044295, -0.0032493, -0.0007467, 0.0008491
9: 0.0018733, 0.0032413, 0.0019039, 0.0032723, -0.0009845, 0.0008658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005540, upper bound: 0.0005679
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0005678
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876652, 0.9895737, 0.9876154, 0.9896749, -0.0013760, 0.0012741
1: -0.0043375, -0.0038619, -0.0043499, -0.0038367, -0.0003429, 0.0003175
2: 0.0104120, 0.0129322, 0.0102784, 0.0129981, -0.0016825, 0.0018169
3: -0.0071593, -0.0060122, -0.0071893, -0.0059514, -0.0008270, 0.0007658
4: 0.0025431, 0.0030309, 0.0025172, 0.0030436, -0.0003256, 0.0003517
5: 0.0120550, 0.0152248, 0.0118870, 0.0153076, -0.0021161, 0.0022852
6: -0.0023234, -0.0015189, -0.0023444, -0.0014762, -0.0005800, 0.0005371
7: -0.0091489, -0.0070674, -0.0092033, -0.0069570, -0.0015007, 0.0013896
8: -0.0043755, -0.0032808, -0.0044041, -0.0032228, -0.0007892, 0.0007308
9: 0.0019404, 0.0032097, 0.0018731, 0.0032429, -0.0008474, 0.0009151

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005246, upper bound: 0.0005337
time: 0.60 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005272, upper bound: 0.0005338
time: 0.57 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876190, 0.9895890, 0.9875920, 0.9896755, -0.0013219, 0.0013692
1: -0.0043490, -0.0038581, -0.0043557, -0.0038366, -0.0003294, 0.0003412
2: 0.0103919, 0.0129933, 0.0102777, 0.0130289, -0.0018081, 0.0017455
3: -0.0071871, -0.0060031, -0.0072033, -0.0059511, -0.0007945, 0.0008230
4: 0.0025392, 0.0030427, 0.0025171, 0.0030496, -0.0003499, 0.0003378
5: 0.0120297, 0.0153016, 0.0118861, 0.0153464, -0.0022741, 0.0021954
6: -0.0023429, -0.0015124, -0.0023543, -0.0014760, -0.0005572, 0.0005772
7: -0.0091993, -0.0070508, -0.0092288, -0.0069565, -0.0014417, 0.0014934
8: -0.0044020, -0.0032721, -0.0044175, -0.0032225, -0.0007582, 0.0007853
9: 0.0019303, 0.0032405, 0.0018728, 0.0032584, -0.0009106, 0.0008791

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005621, upper bound: 0.0005387
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005678, upper bound: 0.0005389
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876652, 0.9895737, 0.9876032, 0.9897101, -0.0014179, 0.0013242
1: -0.0043375, -0.0038619, -0.0043529, -0.0038279, -0.0003533, 0.0003299
2: 0.0104120, 0.0129322, 0.0102320, 0.0130141, -0.0017485, 0.0018724
3: -0.0071593, -0.0060122, -0.0071966, -0.0059303, -0.0008522, 0.0007959
4: 0.0025431, 0.0030309, 0.0025083, 0.0030467, -0.0003384, 0.0003624
5: 0.0120550, 0.0152248, 0.0118286, 0.0153277, -0.0021992, 0.0023550
6: -0.0023234, -0.0015189, -0.0023495, -0.0014614, -0.0005977, 0.0005582
7: -0.0091489, -0.0070674, -0.0092165, -0.0069187, -0.0015465, 0.0014442
8: -0.0043755, -0.0032808, -0.0044110, -0.0032026, -0.0008133, 0.0007595
9: 0.0019404, 0.0032097, 0.0018498, 0.0032509, -0.0008807, 0.0009430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005354, upper bound: 0.0005337
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005338
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876190, 0.9895890, 0.9875782, 0.9897106, -0.0013933, 0.0014067
1: -0.0043490, -0.0038581, -0.0043592, -0.0038278, -0.0003472, 0.0003505
2: 0.0103919, 0.0129933, 0.0102313, 0.0130472, -0.0018575, 0.0018398
3: -0.0071871, -0.0060031, -0.0072117, -0.0059300, -0.0008374, 0.0008454
4: 0.0025392, 0.0030427, 0.0025081, 0.0030532, -0.0003595, 0.0003561
5: 0.0120297, 0.0153016, 0.0118278, 0.0153694, -0.0023362, 0.0023140
6: -0.0023429, -0.0015124, -0.0023601, -0.0014612, -0.0005873, 0.0005930
7: -0.0091993, -0.0070508, -0.0092439, -0.0069181, -0.0015196, 0.0015342
8: -0.0044020, -0.0032721, -0.0044254, -0.0032023, -0.0007991, 0.0008068
9: 0.0019303, 0.0032405, 0.0018494, 0.0032676, -0.0009355, 0.0009266

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005752, upper bound: 0.0005387
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0005389
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876700, 0.9896523, 0.9876154, 0.9896749, -0.0012535, 0.0012384
1: -0.0043363, -0.0038423, -0.0043499, -0.0038367, -0.0003123, 0.0003086
2: 0.0103083, 0.0129259, 0.0102784, 0.0129981, -0.0016352, 0.0016552
3: -0.0071564, -0.0059650, -0.0071893, -0.0059514, -0.0007534, 0.0007443
4: 0.0025230, 0.0030297, 0.0025172, 0.0030436, -0.0003165, 0.0003204
5: 0.0119245, 0.0152168, 0.0118870, 0.0153076, -0.0020567, 0.0020818
6: -0.0023214, -0.0014857, -0.0023444, -0.0014762, -0.0005284, 0.0005220
7: -0.0091437, -0.0069817, -0.0092033, -0.0069570, -0.0013671, 0.0013506
8: -0.0043727, -0.0032357, -0.0044041, -0.0032228, -0.0007189, 0.0007103
9: 0.0018882, 0.0032065, 0.0018731, 0.0032429, -0.0008236, 0.0008336

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005165, upper bound: 0.0005628
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005152, upper bound: 0.0005628
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876177, 0.9896747, 0.9875920, 0.9896755, -0.0012031, 0.0013328
1: -0.0043493, -0.0038367, -0.0043557, -0.0038366, -0.0002998, 0.0003321
2: 0.0102787, 0.0129950, 0.0102777, 0.0130289, -0.0017599, 0.0015887
3: -0.0071879, -0.0059516, -0.0072033, -0.0059511, -0.0007231, 0.0008011
4: 0.0025173, 0.0030430, 0.0025171, 0.0030496, -0.0003406, 0.0003075
5: 0.0118874, 0.0153037, 0.0118861, 0.0153464, -0.0022136, 0.0019982
6: -0.0023434, -0.0014763, -0.0023543, -0.0014760, -0.0005072, 0.0005618
7: -0.0092007, -0.0069573, -0.0092288, -0.0069565, -0.0013122, 0.0014536
8: -0.0044027, -0.0032229, -0.0044175, -0.0032225, -0.0006901, 0.0007644
9: 0.0018733, 0.0032413, 0.0018728, 0.0032584, -0.0008864, 0.0008002

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0005679
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005390, upper bound: 0.0005679
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876700, 0.9896523, 0.9876032, 0.9897101, -0.0012895, 0.0012746
1: -0.0043363, -0.0038423, -0.0043529, -0.0038279, -0.0003213, 0.0003176
2: 0.0103083, 0.0129259, 0.0102320, 0.0130141, -0.0016831, 0.0017027
3: -0.0071564, -0.0059650, -0.0071966, -0.0059303, -0.0007750, 0.0007661
4: 0.0025230, 0.0030297, 0.0025083, 0.0030467, -0.0003258, 0.0003296
5: 0.0119245, 0.0152168, 0.0118286, 0.0153277, -0.0021169, 0.0021416
6: -0.0023214, -0.0014857, -0.0023495, -0.0014614, -0.0005436, 0.0005373
7: -0.0091437, -0.0069817, -0.0092165, -0.0069187, -0.0014063, 0.0013901
8: -0.0043727, -0.0032357, -0.0044110, -0.0032026, -0.0007396, 0.0007311
9: 0.0018882, 0.0032065, 0.0018498, 0.0032509, -0.0008477, 0.0008576

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005268, upper bound: 0.0005627
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005258, upper bound: 0.0005627
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876177, 0.9896747, 0.9875782, 0.9897106, -0.0012722, 0.0013570
1: -0.0043493, -0.0038367, -0.0043592, -0.0038278, -0.0003170, 0.0003381
2: 0.0102787, 0.0129950, 0.0102313, 0.0130472, -0.0017918, 0.0016800
3: -0.0071879, -0.0059516, -0.0072117, -0.0059300, -0.0007647, 0.0008156
4: 0.0025173, 0.0030430, 0.0025081, 0.0030532, -0.0003468, 0.0003252
5: 0.0118874, 0.0153037, 0.0118278, 0.0153694, -0.0022537, 0.0021130
6: -0.0023434, -0.0014763, -0.0023601, -0.0014612, -0.0005363, 0.0005720
7: -0.0092007, -0.0069573, -0.0092439, -0.0069181, -0.0013876, 0.0014800
8: -0.0044027, -0.0032229, -0.0044254, -0.0032023, -0.0007297, 0.0007783
9: 0.0018733, 0.0032413, 0.0018494, 0.0032676, -0.0009025, 0.0008461

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005545, upper bound: 0.0005679
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005516, upper bound: 0.0005679
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876518, 0.9896082, 0.9876131, 0.9895893, -0.0012772, 0.0012883
1: -0.0043408, -0.0038533, -0.0043505, -0.0038580, -0.0003182, 0.0003210
2: 0.0103665, 0.0129499, 0.0103915, 0.0130012, -0.0017012, 0.0016865
3: -0.0071674, -0.0059915, -0.0071907, -0.0060029, -0.0007676, 0.0007743
4: 0.0025343, 0.0030343, 0.0025391, 0.0030442, -0.0003293, 0.0003264
5: 0.0119978, 0.0152470, 0.0120292, 0.0153115, -0.0021397, 0.0021211
6: -0.0023290, -0.0015043, -0.0023454, -0.0015123, -0.0005384, 0.0005431
7: -0.0091635, -0.0070298, -0.0092058, -0.0070504, -0.0013929, 0.0014051
8: -0.0043832, -0.0032611, -0.0044054, -0.0032719, -0.0007325, 0.0007389
9: 0.0019175, 0.0032186, 0.0019301, 0.0032444, -0.0008568, 0.0008494

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005157, upper bound: 0.0005521
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005522
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876574, 0.9896910, 0.9876131, 0.9895893, -0.0013419, 0.0014320
1: -0.0043394, -0.0038327, -0.0043505, -0.0038580, -0.0003344, 0.0003568
2: 0.0102571, 0.0129426, 0.0103915, 0.0130012, -0.0018910, 0.0017719
3: -0.0071640, -0.0059417, -0.0071907, -0.0060029, -0.0008065, 0.0008607
4: 0.0025131, 0.0030329, 0.0025391, 0.0030442, -0.0003660, 0.0003430
5: 0.0118602, 0.0152378, 0.0120292, 0.0153115, -0.0023784, 0.0022286
6: -0.0023267, -0.0014694, -0.0023454, -0.0015123, -0.0005657, 0.0006037
7: -0.0091575, -0.0069395, -0.0092058, -0.0070504, -0.0014635, 0.0015618
8: -0.0043800, -0.0032135, -0.0044054, -0.0032719, -0.0007696, 0.0008214
9: 0.0018624, 0.0032149, 0.0019301, 0.0032444, -0.0009524, 0.0008924

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005157, upper bound: 0.0005720
time: 0.62 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005721
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876518, 0.9896082, 0.9875963, 0.9896281, -0.0012529, 0.0012361
1: -0.0043408, -0.0038533, -0.0043546, -0.0038484, -0.0003122, 0.0003080
2: 0.0103665, 0.0129499, 0.0103403, 0.0130232, -0.0016322, 0.0016545
3: -0.0071674, -0.0059915, -0.0072007, -0.0059796, -0.0007530, 0.0007429
4: 0.0025343, 0.0030343, 0.0025292, 0.0030485, -0.0003159, 0.0003202
5: 0.0119978, 0.0152470, 0.0119648, 0.0153392, -0.0020529, 0.0020809
6: -0.0023290, -0.0015043, -0.0023524, -0.0014960, -0.0005282, 0.0005210
7: -0.0091635, -0.0070298, -0.0092241, -0.0070081, -0.0013665, 0.0013481
8: -0.0043832, -0.0032611, -0.0044150, -0.0032496, -0.0007186, 0.0007089
9: 0.0019175, 0.0032186, 0.0019043, 0.0032556, -0.0008221, 0.0008333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005521
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005522
time: 0.58 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876574, 0.9896910, 0.9875963, 0.9896281, -0.0013047, 0.0013760
1: -0.0043394, -0.0038327, -0.0043546, -0.0038484, -0.0003251, 0.0003429
2: 0.0102571, 0.0129426, 0.0103403, 0.0130232, -0.0018170, 0.0017229
3: -0.0071640, -0.0059417, -0.0072007, -0.0059796, -0.0007842, 0.0008270
4: 0.0025131, 0.0030329, 0.0025292, 0.0030485, -0.0003517, 0.0003335
5: 0.0118602, 0.0152378, 0.0119648, 0.0153392, -0.0022853, 0.0021669
6: -0.0023267, -0.0014694, -0.0023524, -0.0014960, -0.0005500, 0.0005800
7: -0.0091575, -0.0069395, -0.0092241, -0.0070081, -0.0014230, 0.0015007
8: -0.0043800, -0.0032135, -0.0044150, -0.0032496, -0.0007483, 0.0007892
9: 0.0018624, 0.0032149, 0.0019043, 0.0032556, -0.0009151, 0.0008677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005720
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005721
time: 0.59 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9875967, 0.9896278, 0.9875925, 0.9895899, -0.0012249, 0.0013580
1: -0.0043545, -0.0038484, -0.0043556, -0.0038579, -0.0003052, 0.0003384
2: 0.0103406, 0.0130228, 0.0103907, 0.0130282, -0.0017933, 0.0016175
3: -0.0072005, -0.0059797, -0.0072030, -0.0060025, -0.0007362, 0.0008162
4: 0.0025293, 0.0030484, 0.0025390, 0.0030495, -0.0003471, 0.0003131
5: 0.0119653, 0.0153387, 0.0120282, 0.0153455, -0.0022554, 0.0020344
6: -0.0023523, -0.0014961, -0.0023540, -0.0015121, -0.0005164, 0.0005725
7: -0.0092237, -0.0070084, -0.0092282, -0.0070498, -0.0013360, 0.0014811
8: -0.0044148, -0.0032498, -0.0044172, -0.0032716, -0.0007026, 0.0007789
9: 0.0019045, 0.0032553, 0.0019297, 0.0032581, -0.0009032, 0.0008147

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005574
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005575
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876041, 0.9897098, 0.9875925, 0.9895899, -0.0012867, 0.0015044
1: -0.0043527, -0.0038280, -0.0043556, -0.0038579, -0.0003206, 0.0003748
2: 0.0102324, 0.0130129, 0.0103907, 0.0130282, -0.0019865, 0.0016991
3: -0.0071960, -0.0059305, -0.0072030, -0.0060025, -0.0007733, 0.0009042
4: 0.0025083, 0.0030465, 0.0025390, 0.0030495, -0.0003845, 0.0003288
5: 0.0118291, 0.0153262, 0.0120282, 0.0153455, -0.0024985, 0.0021370
6: -0.0023491, -0.0014615, -0.0023540, -0.0015121, -0.0005424, 0.0006341
7: -0.0092155, -0.0069190, -0.0092282, -0.0070498, -0.0014033, 0.0016407
8: -0.0044105, -0.0032028, -0.0044172, -0.0032716, -0.0007380, 0.0008628
9: 0.0018499, 0.0032503, 0.0019297, 0.0032581, -0.0010005, 0.0008557

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005775
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005776
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9875967, 0.9896278, 0.9875711, 0.9896287, -0.0011994, 0.0013328
1: -0.0043545, -0.0038484, -0.0043609, -0.0038482, -0.0002989, 0.0003321
2: 0.0103406, 0.0130228, 0.0103395, 0.0130566, -0.0017599, 0.0015838
3: -0.0072005, -0.0059797, -0.0072159, -0.0059792, -0.0007209, 0.0008010
4: 0.0025293, 0.0030484, 0.0025291, 0.0030550, -0.0003406, 0.0003065
5: 0.0119653, 0.0153387, 0.0119638, 0.0153812, -0.0022135, 0.0019920
6: -0.0023523, -0.0014961, -0.0023631, -0.0014957, -0.0005056, 0.0005618
7: -0.0092237, -0.0070084, -0.0092516, -0.0070075, -0.0013081, 0.0014536
8: -0.0044148, -0.0032498, -0.0044295, -0.0032493, -0.0006879, 0.0007644
9: 0.0019045, 0.0032553, 0.0019039, 0.0032723, -0.0008864, 0.0007977

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005574
time: 0.69 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005388, upper bound: 0.0005575
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876041, 0.9897098, 0.9875711, 0.9896287, -0.0012490, 0.0014706
1: -0.0043527, -0.0038280, -0.0043609, -0.0038482, -0.0003112, 0.0003664
2: 0.0102324, 0.0130129, 0.0103395, 0.0130566, -0.0019419, 0.0016493
3: -0.0071960, -0.0059305, -0.0072159, -0.0059792, -0.0007507, 0.0008839
4: 0.0025083, 0.0030465, 0.0025291, 0.0030550, -0.0003759, 0.0003192
5: 0.0118291, 0.0153262, 0.0119638, 0.0153812, -0.0024424, 0.0020743
6: -0.0023491, -0.0014615, -0.0023631, -0.0014957, -0.0005265, 0.0006199
7: -0.0092155, -0.0069190, -0.0092516, -0.0070075, -0.0013622, 0.0016039
8: -0.0044105, -0.0032028, -0.0044295, -0.0032493, -0.0007164, 0.0008435
9: 0.0018499, 0.0032503, 0.0019039, 0.0032723, -0.0009780, 0.0008307

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005775
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005388, upper bound: 0.0005776
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876518, 0.9896082, 0.9876154, 0.9896749, -0.0014155, 0.0013404
1: -0.0043408, -0.0038533, -0.0043499, -0.0038367, -0.0003527, 0.0003340
2: 0.0103665, 0.0129499, 0.0102784, 0.0129981, -0.0017700, 0.0018692
3: -0.0071674, -0.0059915, -0.0071893, -0.0059514, -0.0008508, 0.0008056
4: 0.0025343, 0.0030343, 0.0025172, 0.0030436, -0.0003426, 0.0003618
5: 0.0119978, 0.0152470, 0.0118870, 0.0153076, -0.0022262, 0.0023510
6: -0.0023290, -0.0015043, -0.0023444, -0.0014762, -0.0005967, 0.0005650
7: -0.0091635, -0.0070298, -0.0092033, -0.0069570, -0.0015439, 0.0014619
8: -0.0043832, -0.0032611, -0.0044041, -0.0032228, -0.0008119, 0.0007688
9: 0.0019175, 0.0032186, 0.0018731, 0.0032429, -0.0008915, 0.0009414

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005225, upper bound: 0.0005452
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005269, upper bound: 0.0005460
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9875967, 0.9896278, 0.9875920, 0.9896755, -0.0013639, 0.0014100
1: -0.0043545, -0.0038484, -0.0043557, -0.0038366, -0.0003398, 0.0003513
2: 0.0103406, 0.0130228, 0.0102777, 0.0130289, -0.0018619, 0.0018010
3: -0.0072005, -0.0059797, -0.0072033, -0.0059511, -0.0008197, 0.0008475
4: 0.0025293, 0.0030484, 0.0025171, 0.0030496, -0.0003604, 0.0003486
5: 0.0119653, 0.0153387, 0.0118861, 0.0153464, -0.0023418, 0.0022651
6: -0.0023523, -0.0014961, -0.0023543, -0.0014760, -0.0005749, 0.0005944
7: -0.0092237, -0.0070084, -0.0092288, -0.0069565, -0.0014875, 0.0015379
8: -0.0044148, -0.0032498, -0.0044175, -0.0032225, -0.0007823, 0.0008087
9: 0.0019045, 0.0032553, 0.0018728, 0.0032584, -0.0009378, 0.0009071

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005621, upper bound: 0.0005506
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005678, upper bound: 0.0005512
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876518, 0.9896082, 0.9876032, 0.9897101, -0.0013899, 0.0012875
1: -0.0043408, -0.0038533, -0.0043529, -0.0038279, -0.0003463, 0.0003208
2: 0.0103665, 0.0129499, 0.0102320, 0.0130141, -0.0017001, 0.0018353
3: -0.0071674, -0.0059915, -0.0071966, -0.0059303, -0.0008354, 0.0007738
4: 0.0025343, 0.0030343, 0.0025083, 0.0030467, -0.0003291, 0.0003552
5: 0.0119978, 0.0152470, 0.0118286, 0.0153277, -0.0021383, 0.0023084
6: -0.0023290, -0.0015043, -0.0023495, -0.0014614, -0.0005859, 0.0005427
7: -0.0091635, -0.0070298, -0.0092165, -0.0069187, -0.0015159, 0.0014042
8: -0.0043832, -0.0032611, -0.0044110, -0.0032026, -0.0007972, 0.0007385
9: 0.0019175, 0.0032186, 0.0018498, 0.0032509, -0.0008563, 0.0009244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005224, upper bound: 0.0005453
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005269, upper bound: 0.0005460
time: 0.62 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9875967, 0.9896278, 0.9875782, 0.9897106, -0.0013381, 0.0013835
1: -0.0043545, -0.0038484, -0.0043592, -0.0038278, -0.0003334, 0.0003447
2: 0.0103406, 0.0130228, 0.0102313, 0.0130472, -0.0018270, 0.0017670
3: -0.0072005, -0.0059797, -0.0072117, -0.0059300, -0.0008043, 0.0008316
4: 0.0025293, 0.0030484, 0.0025081, 0.0030532, -0.0003536, 0.0003420
5: 0.0119653, 0.0153387, 0.0118278, 0.0153694, -0.0022978, 0.0022224
6: -0.0023523, -0.0014961, -0.0023601, -0.0014612, -0.0005641, 0.0005832
7: -0.0092237, -0.0070084, -0.0092439, -0.0069181, -0.0014594, 0.0015090
8: -0.0044148, -0.0032498, -0.0044254, -0.0032023, -0.0007675, 0.0007935
9: 0.0019045, 0.0032553, 0.0018494, 0.0032676, -0.0009202, 0.0008899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.78 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005505
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005679, upper bound: 0.0005512
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876574, 0.9896910, 0.9876154, 0.9896749, -0.0012925, 0.0013046
1: -0.0043394, -0.0038327, -0.0043499, -0.0038367, -0.0003221, 0.0003251
2: 0.0102571, 0.0129426, 0.0102784, 0.0129981, -0.0017228, 0.0017068
3: -0.0071640, -0.0059417, -0.0071893, -0.0059514, -0.0007769, 0.0007841
4: 0.0025131, 0.0030329, 0.0025172, 0.0030436, -0.0003334, 0.0003303
5: 0.0118602, 0.0152378, 0.0118870, 0.0153076, -0.0021668, 0.0021467
6: -0.0023267, -0.0014694, -0.0023444, -0.0014762, -0.0005449, 0.0005499
7: -0.0091575, -0.0069395, -0.0092033, -0.0069570, -0.0014097, 0.0014229
8: -0.0043800, -0.0032135, -0.0044041, -0.0032228, -0.0007413, 0.0007483
9: 0.0018624, 0.0032149, 0.0018731, 0.0032429, -0.0008677, 0.0008596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005159, upper bound: 0.0005721
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005139, upper bound: 0.0005721
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876041, 0.9897098, 0.9875920, 0.9896755, -0.0012454, 0.0013732
1: -0.0043527, -0.0038280, -0.0043557, -0.0038366, -0.0003103, 0.0003422
2: 0.0102324, 0.0130129, 0.0102777, 0.0130289, -0.0018133, 0.0016446
3: -0.0071960, -0.0059305, -0.0072033, -0.0059511, -0.0007485, 0.0008253
4: 0.0025083, 0.0030465, 0.0025171, 0.0030496, -0.0003510, 0.0003183
5: 0.0118291, 0.0153262, 0.0118861, 0.0153464, -0.0022807, 0.0020684
6: -0.0023491, -0.0014615, -0.0023543, -0.0014760, -0.0005250, 0.0005789
7: -0.0092155, -0.0069190, -0.0092288, -0.0069565, -0.0013583, 0.0014977
8: -0.0044105, -0.0032028, -0.0044175, -0.0032225, -0.0007143, 0.0007876
9: 0.0018499, 0.0032503, 0.0018728, 0.0032584, -0.0009133, 0.0008283

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0005776
time: 0.60 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005776
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876574, 0.9896910, 0.9876032, 0.9897101, -0.0012699, 0.0012534
1: -0.0043394, -0.0038327, -0.0043529, -0.0038279, -0.0003164, 0.0003123
2: 0.0102571, 0.0129426, 0.0102320, 0.0130141, -0.0016551, 0.0016769
3: -0.0071640, -0.0059417, -0.0071966, -0.0059303, -0.0007632, 0.0007533
4: 0.0025131, 0.0030329, 0.0025083, 0.0030467, -0.0003203, 0.0003246
5: 0.0118602, 0.0152378, 0.0118286, 0.0153277, -0.0020817, 0.0021090
6: -0.0023267, -0.0014694, -0.0023495, -0.0014614, -0.0005353, 0.0005284
7: -0.0091575, -0.0069395, -0.0092165, -0.0069187, -0.0013850, 0.0013670
8: -0.0043800, -0.0032135, -0.0044110, -0.0032026, -0.0007283, 0.0007189
9: 0.0018624, 0.0032149, 0.0018498, 0.0032509, -0.0008336, 0.0008446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005159, upper bound: 0.0005721
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005140, upper bound: 0.0005721
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876041, 0.9897098, 0.9875782, 0.9897106, -0.0012167, 0.0013494
1: -0.0043527, -0.0038280, -0.0043592, -0.0038278, -0.0003032, 0.0003362
2: 0.0102324, 0.0130129, 0.0102313, 0.0130472, -0.0017819, 0.0016066
3: -0.0071960, -0.0059305, -0.0072117, -0.0059300, -0.0007313, 0.0008110
4: 0.0025083, 0.0030465, 0.0025081, 0.0030532, -0.0003449, 0.0003110
5: 0.0118291, 0.0153262, 0.0118278, 0.0153694, -0.0022411, 0.0020207
6: -0.0023491, -0.0014615, -0.0023601, -0.0014612, -0.0005129, 0.0005688
7: -0.0092155, -0.0069190, -0.0092439, -0.0069181, -0.0013270, 0.0014717
8: -0.0044105, -0.0032028, -0.0044254, -0.0032023, -0.0006978, 0.0007740
9: 0.0018499, 0.0032503, 0.0018494, 0.0032676, -0.0008975, 0.0008092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0005776
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005390, upper bound: 0.0005776
time: 0.65 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.32 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005162, upper bound: 0.0005384
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005149, upper bound: 0.0005384
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005162, upper bound: 0.0005628
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005149, upper bound: 0.0005628
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005383
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005256, upper bound: 0.0005384
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005264, upper bound: 0.0005627
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005256, upper bound: 0.0005627
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005435
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005436
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005679
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005679
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005540, upper bound: 0.0005434
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0005436
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005540, upper bound: 0.0005679
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005512, upper bound: 0.0005678
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005246, upper bound: 0.0005337
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005272, upper bound: 0.0005338
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005621, upper bound: 0.0005387
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005678, upper bound: 0.0005389
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005354, upper bound: 0.0005337
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005338
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005752, upper bound: 0.0005387
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005776, upper bound: 0.0005389
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005165, upper bound: 0.0005628
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005152, upper bound: 0.0005628
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0005679
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005390, upper bound: 0.0005679
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005268, upper bound: 0.0005627
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005258, upper bound: 0.0005627
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005545, upper bound: 0.0005679
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005516, upper bound: 0.0005679
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005157, upper bound: 0.0005521
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005522
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005157, upper bound: 0.0005720
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005721
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005521
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005522
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005156, upper bound: 0.0005720
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005137, upper bound: 0.0005721
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005574
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005575
IS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005775
IS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005776
IS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005574
IS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005388, upper bound: 0.0005575
IS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005419, upper bound: 0.0005775
IS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005388, upper bound: 0.0005776
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005225, upper bound: 0.0005452
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005269, upper bound: 0.0005460
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005621, upper bound: 0.0005506
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005678, upper bound: 0.0005512
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005224, upper bound: 0.0005453
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005269, upper bound: 0.0005460
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005618, upper bound: 0.0005505
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005679, upper bound: 0.0005512
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005159, upper bound: 0.0005721
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005139, upper bound: 0.0005721
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0005776
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005389, upper bound: 0.0005776
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005159, upper bound: 0.0005721
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005140, upper bound: 0.0005721
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005422, upper bound: 0.0005776
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.32
Output dim: 0, lower bound: -0.0005390, upper bound: 0.0005776

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876670, 0.9895522, 0.9876142, 0.9895375, -0.0011703, 0.0011975
1: -0.0043370, -0.0038673, -0.0043502, -0.0038709, -0.0002916, 0.0002984
2: 0.0104405, 0.0129300, 0.0104599, 0.0129997, -0.0015813, 0.0015454
3: -0.0071583, -0.0060252, -0.0071900, -0.0060340, -0.0007034, 0.0007197
4: 0.0025486, 0.0030305, 0.0025524, 0.0030439, -0.0003060, 0.0002991
5: 0.0120908, 0.0152220, 0.0121152, 0.0153096, -0.0019888, 0.0019437
6: -0.0023227, -0.0015280, -0.0023449, -0.0015341, -0.0004933, 0.0005048
7: -0.0091471, -0.0070909, -0.0092046, -0.0071069, -0.0012764, 0.0013060
8: -0.0043745, -0.0032932, -0.0044048, -0.0033016, -0.0006712, 0.0006868
9: 0.0019548, 0.0032086, 0.0019645, 0.0032437, -0.0007964, 0.0007783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004733, upper bound: 0.0005061
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005060
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876665, 0.9895525, 0.9876163, 0.9895376, -0.0011653, 0.0011996
1: -0.0043371, -0.0038672, -0.0043497, -0.0038709, -0.0002904, 0.0002989
2: 0.0104401, 0.0129305, 0.0104597, 0.0129969, -0.0015841, 0.0015387
3: -0.0071585, -0.0060250, -0.0071888, -0.0060339, -0.0007004, 0.0007210
4: 0.0025485, 0.0030306, 0.0025523, 0.0030434, -0.0003066, 0.0002978
5: 0.0120904, 0.0152226, 0.0121150, 0.0153062, -0.0019924, 0.0019353
6: -0.0023228, -0.0015278, -0.0023440, -0.0015341, -0.0004912, 0.0005057
7: -0.0091475, -0.0070906, -0.0092023, -0.0071068, -0.0012709, 0.0013084
8: -0.0043747, -0.0032930, -0.0044036, -0.0033015, -0.0006684, 0.0006881
9: 0.0019546, 0.0032089, 0.0019644, 0.0032423, -0.0007978, 0.0007750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004758, upper bound: 0.0005062
time: 0.59 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004841, upper bound: 0.0005061
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876716, 0.9896285, 0.9876142, 0.9895375, -0.0012190, 0.0013366
1: -0.0043359, -0.0038483, -0.0043502, -0.0038709, -0.0003037, 0.0003330
2: 0.0103399, 0.0129238, 0.0104599, 0.0129997, -0.0017650, 0.0016096
3: -0.0071555, -0.0059794, -0.0071900, -0.0060340, -0.0007326, 0.0008033
4: 0.0025291, 0.0030293, 0.0025524, 0.0030439, -0.0003416, 0.0003115
5: 0.0119643, 0.0152142, 0.0121152, 0.0153096, -0.0022198, 0.0020245
6: -0.0023207, -0.0014958, -0.0023449, -0.0015341, -0.0005138, 0.0005634
7: -0.0091420, -0.0070078, -0.0092046, -0.0071069, -0.0013295, 0.0014577
8: -0.0043718, -0.0032495, -0.0044048, -0.0033016, -0.0006992, 0.0007666
9: 0.0019041, 0.0032055, 0.0019645, 0.0032437, -0.0008889, 0.0008107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004728, upper bound: 0.0005270
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004855, upper bound: 0.0005270
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876713, 0.9896333, 0.9876163, 0.9895376, -0.0012110, 0.0013417
1: -0.0043360, -0.0038471, -0.0043497, -0.0038709, -0.0003017, 0.0003343
2: 0.0103334, 0.0129243, 0.0104597, 0.0129969, -0.0017716, 0.0015991
3: -0.0071557, -0.0059764, -0.0071888, -0.0060339, -0.0007278, 0.0008064
4: 0.0025279, 0.0030294, 0.0025523, 0.0030434, -0.0003429, 0.0003095
5: 0.0119561, 0.0152149, 0.0121150, 0.0153062, -0.0022283, 0.0020113
6: -0.0023209, -0.0014938, -0.0023440, -0.0015341, -0.0005105, 0.0005656
7: -0.0091424, -0.0070024, -0.0092023, -0.0071068, -0.0013208, 0.0014633
8: -0.0043720, -0.0032466, -0.0044036, -0.0033015, -0.0006946, 0.0007695
9: 0.0019008, 0.0032057, 0.0019644, 0.0032423, -0.0008923, 0.0008054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004734, upper bound: 0.0005271
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004836, upper bound: 0.0005271
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876670, 0.9895522, 0.9875999, 0.9895701, -0.0012077, 0.0012313
1: -0.0043370, -0.0038673, -0.0043537, -0.0038628, -0.0003009, 0.0003068
2: 0.0104405, 0.0129300, 0.0104169, 0.0130184, -0.0016260, 0.0015948
3: -0.0071583, -0.0060252, -0.0071985, -0.0060144, -0.0007259, 0.0007401
4: 0.0025486, 0.0030305, 0.0025441, 0.0030476, -0.0003147, 0.0003087
5: 0.0120908, 0.0152220, 0.0120612, 0.0153332, -0.0020451, 0.0020058
6: -0.0023227, -0.0015280, -0.0023509, -0.0015204, -0.0005091, 0.0005191
7: -0.0091471, -0.0070909, -0.0092201, -0.0070714, -0.0013172, 0.0013430
8: -0.0043745, -0.0032932, -0.0044129, -0.0032829, -0.0006927, 0.0007062
9: 0.0019548, 0.0032086, 0.0019429, 0.0032531, -0.0008189, 0.0008032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005061
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005060
time: 0.63 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876665, 0.9895525, 0.9876001, 0.9895779, -0.0012045, 0.0012359
1: -0.0043371, -0.0038672, -0.0043537, -0.0038608, -0.0003001, 0.0003080
2: 0.0104401, 0.0129305, 0.0104064, 0.0130182, -0.0016320, 0.0015905
3: -0.0071585, -0.0060250, -0.0071985, -0.0060097, -0.0007239, 0.0007428
4: 0.0025485, 0.0030306, 0.0025420, 0.0030475, -0.0003159, 0.0003078
5: 0.0120904, 0.0152226, 0.0120480, 0.0153329, -0.0020526, 0.0020004
6: -0.0023228, -0.0015278, -0.0023508, -0.0015171, -0.0005077, 0.0005210
7: -0.0091475, -0.0070906, -0.0092199, -0.0070628, -0.0013136, 0.0013479
8: -0.0043747, -0.0032930, -0.0044128, -0.0032784, -0.0006908, 0.0007089
9: 0.0019546, 0.0032089, 0.0019376, 0.0032530, -0.0008220, 0.0008011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005061
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004944, upper bound: 0.0005061
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876716, 0.9896285, 0.9875999, 0.9895701, -0.0012564, 0.0013705
1: -0.0043359, -0.0038483, -0.0043537, -0.0038628, -0.0003131, 0.0003415
2: 0.0103399, 0.0129238, 0.0104169, 0.0130184, -0.0018097, 0.0016591
3: -0.0071555, -0.0059794, -0.0071985, -0.0060144, -0.0007551, 0.0008237
4: 0.0025291, 0.0030293, 0.0025441, 0.0030476, -0.0003503, 0.0003211
5: 0.0119643, 0.0152142, 0.0120612, 0.0153332, -0.0022761, 0.0020867
6: -0.0023207, -0.0014958, -0.0023509, -0.0015204, -0.0005296, 0.0005777
7: -0.0091420, -0.0070078, -0.0092201, -0.0070714, -0.0013703, 0.0014947
8: -0.0043718, -0.0032495, -0.0044129, -0.0032829, -0.0007206, 0.0007860
9: 0.0019041, 0.0032055, 0.0019429, 0.0032531, -0.0009115, 0.0008356

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005269
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005269
time: 0.59 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876713, 0.9896333, 0.9876001, 0.9895779, -0.0012502, 0.0013779
1: -0.0043360, -0.0038471, -0.0043537, -0.0038608, -0.0003115, 0.0003433
2: 0.0103334, 0.0129243, 0.0104064, 0.0130182, -0.0018195, 0.0016509
3: -0.0071557, -0.0059764, -0.0071985, -0.0060097, -0.0007514, 0.0008282
4: 0.0025279, 0.0030294, 0.0025420, 0.0030475, -0.0003522, 0.0003195
5: 0.0119561, 0.0152149, 0.0120480, 0.0153329, -0.0022885, 0.0020763
6: -0.0023209, -0.0014938, -0.0023508, -0.0015171, -0.0005270, 0.0005808
7: -0.0091424, -0.0070024, -0.0092199, -0.0070628, -0.0013635, 0.0015028
8: -0.0043720, -0.0032466, -0.0044128, -0.0032784, -0.0007171, 0.0007903
9: 0.0019008, 0.0032057, 0.0019376, 0.0032530, -0.0009164, 0.0008315

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005270
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005270
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876208, 0.9895682, 0.9875922, 0.9895381, -0.0011174, 0.0012932
1: -0.0043485, -0.0038633, -0.0043556, -0.0038708, -0.0002784, 0.0003222
2: 0.0104194, 0.0129908, 0.0104592, 0.0130286, -0.0017077, 0.0014755
3: -0.0071860, -0.0060156, -0.0072032, -0.0060337, -0.0006716, 0.0007773
4: 0.0025445, 0.0030422, 0.0025522, 0.0030496, -0.0003305, 0.0002856
5: 0.0120643, 0.0152985, 0.0121144, 0.0153461, -0.0021478, 0.0018558
6: -0.0023421, -0.0015212, -0.0023542, -0.0015339, -0.0004710, 0.0005451
7: -0.0091973, -0.0070735, -0.0092286, -0.0071063, -0.0012187, 0.0014104
8: -0.0044009, -0.0032840, -0.0044174, -0.0033013, -0.0006409, 0.0007417
9: 0.0019441, 0.0032392, 0.0019642, 0.0032583, -0.0008601, 0.0007431

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005095
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005128, upper bound: 0.0005095
time: 0.68 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876202, 0.9895679, 0.9875958, 0.9895383, -0.0011127, 0.0012930
1: -0.0043487, -0.0038634, -0.0043548, -0.0038707, -0.0002772, 0.0003222
2: 0.0104199, 0.0129917, 0.0104589, 0.0130239, -0.0017074, 0.0014693
3: -0.0071864, -0.0060158, -0.0072010, -0.0060336, -0.0006687, 0.0007771
4: 0.0025446, 0.0030424, 0.0025522, 0.0030486, -0.0003305, 0.0002844
5: 0.0120650, 0.0152995, 0.0121140, 0.0153401, -0.0021475, 0.0018480
6: -0.0023424, -0.0015214, -0.0023526, -0.0015338, -0.0004690, 0.0005450
7: -0.0091980, -0.0070739, -0.0092246, -0.0071061, -0.0012135, 0.0014102
8: -0.0044013, -0.0032842, -0.0044153, -0.0033012, -0.0006382, 0.0007416
9: 0.0019444, 0.0032397, 0.0019640, 0.0032559, -0.0008599, 0.0007400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005096
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005096
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876197, 0.9896494, 0.9875922, 0.9895381, -0.0011643, 0.0014294
1: -0.0043488, -0.0038431, -0.0043556, -0.0038708, -0.0002901, 0.0003562
2: 0.0103122, 0.0129924, 0.0104592, 0.0130286, -0.0018875, 0.0015374
3: -0.0071867, -0.0059668, -0.0072032, -0.0060337, -0.0006998, 0.0008591
4: 0.0025238, 0.0030425, 0.0025522, 0.0030496, -0.0003653, 0.0002976
5: 0.0119295, 0.0153005, 0.0121144, 0.0153461, -0.0023740, 0.0019337
6: -0.0023426, -0.0014870, -0.0023542, -0.0015339, -0.0004908, 0.0006025
7: -0.0091986, -0.0069849, -0.0092286, -0.0071063, -0.0012698, 0.0015590
8: -0.0044016, -0.0032375, -0.0044174, -0.0033013, -0.0006678, 0.0008198
9: 0.0018901, 0.0032400, 0.0019642, 0.0032583, -0.0009507, 0.0007743

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005006, upper bound: 0.0005318
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005318
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876189, 0.9896577, 0.9875958, 0.9895383, -0.0011560, 0.0014364
1: -0.0043490, -0.0038410, -0.0043548, -0.0038707, -0.0002880, 0.0003579
2: 0.0103012, 0.0129934, 0.0104589, 0.0130239, -0.0018968, 0.0015264
3: -0.0071872, -0.0059618, -0.0072010, -0.0060336, -0.0006948, 0.0008633
4: 0.0025217, 0.0030427, 0.0025522, 0.0030486, -0.0003671, 0.0002954
5: 0.0119157, 0.0153017, 0.0121140, 0.0153401, -0.0023857, 0.0019198
6: -0.0023429, -0.0014835, -0.0023526, -0.0015338, -0.0004873, 0.0006055
7: -0.0091994, -0.0069759, -0.0092246, -0.0071061, -0.0012607, 0.0015666
8: -0.0044020, -0.0032327, -0.0044153, -0.0033012, -0.0006630, 0.0008239
9: 0.0018846, 0.0032405, 0.0019640, 0.0032559, -0.0009553, 0.0007688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0005318
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005059, upper bound: 0.0005318
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876208, 0.9895682, 0.9875783, 0.9895706, -0.0011819, 0.0013111
1: -0.0043485, -0.0038633, -0.0043591, -0.0038627, -0.0002945, 0.0003267
2: 0.0104194, 0.0129908, 0.0104163, 0.0130469, -0.0017313, 0.0015607
3: -0.0071860, -0.0060156, -0.0072115, -0.0060142, -0.0007104, 0.0007880
4: 0.0025445, 0.0030422, 0.0025439, 0.0030531, -0.0003351, 0.0003021
5: 0.0120643, 0.0152985, 0.0120604, 0.0153691, -0.0021776, 0.0019629
6: -0.0023421, -0.0015212, -0.0023600, -0.0015202, -0.0004982, 0.0005527
7: -0.0091973, -0.0070735, -0.0092437, -0.0070709, -0.0012890, 0.0014300
8: -0.0044009, -0.0032840, -0.0044253, -0.0032827, -0.0006779, 0.0007520
9: 0.0019441, 0.0032392, 0.0019426, 0.0032675, -0.0008720, 0.0007860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.79 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005177, upper bound: 0.0005095
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005242, upper bound: 0.0005095
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876202, 0.9895679, 0.9875749, 0.9895787, -0.0011781, 0.0013164
1: -0.0043487, -0.0038634, -0.0043600, -0.0038607, -0.0002936, 0.0003280
2: 0.0104199, 0.0129917, 0.0104056, 0.0130515, -0.0017383, 0.0015557
3: -0.0071864, -0.0060158, -0.0072136, -0.0060093, -0.0007081, 0.0007912
4: 0.0025446, 0.0030424, 0.0025419, 0.0030540, -0.0003364, 0.0003011
5: 0.0120650, 0.0152995, 0.0120470, 0.0153748, -0.0021863, 0.0019567
6: -0.0023424, -0.0015214, -0.0023615, -0.0015168, -0.0004966, 0.0005549
7: -0.0091980, -0.0070739, -0.0092474, -0.0070621, -0.0012849, 0.0014357
8: -0.0044013, -0.0032842, -0.0044273, -0.0032780, -0.0006757, 0.0007550
9: 0.0019444, 0.0032397, 0.0019372, 0.0032698, -0.0008755, 0.0007835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005196, upper bound: 0.0005096
time: 0.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005221, upper bound: 0.0005096
time: 0.58 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876197, 0.9896494, 0.9875783, 0.9895706, -0.0012288, 0.0014473
1: -0.0043488, -0.0038431, -0.0043591, -0.0038627, -0.0003062, 0.0003606
2: 0.0103122, 0.0129924, 0.0104163, 0.0130469, -0.0019112, 0.0016226
3: -0.0071867, -0.0059668, -0.0072115, -0.0060142, -0.0007385, 0.0008699
4: 0.0025238, 0.0030425, 0.0025439, 0.0030531, -0.0003699, 0.0003141
5: 0.0119295, 0.0153005, 0.0120604, 0.0153691, -0.0024038, 0.0020408
6: -0.0023426, -0.0014870, -0.0023600, -0.0015202, -0.0005180, 0.0006101
7: -0.0091986, -0.0069849, -0.0092437, -0.0070709, -0.0013402, 0.0015785
8: -0.0044016, -0.0032375, -0.0044253, -0.0032827, -0.0007048, 0.0008301
9: 0.0018901, 0.0032400, 0.0019426, 0.0032675, -0.0009626, 0.0008172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005316
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005224, upper bound: 0.0005316
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876189, 0.9896577, 0.9875749, 0.9895787, -0.0012214, 0.0014598
1: -0.0043490, -0.0038410, -0.0043600, -0.0038607, -0.0003043, 0.0003638
2: 0.0103012, 0.0129934, 0.0104056, 0.0130515, -0.0019277, 0.0016129
3: -0.0071872, -0.0059618, -0.0072136, -0.0060093, -0.0007341, 0.0008774
4: 0.0025217, 0.0030427, 0.0025419, 0.0030540, -0.0003731, 0.0003122
5: 0.0119157, 0.0153017, 0.0120470, 0.0153748, -0.0024245, 0.0020286
6: -0.0023429, -0.0014835, -0.0023615, -0.0015168, -0.0005149, 0.0006154
7: -0.0091994, -0.0069759, -0.0092474, -0.0070621, -0.0013321, 0.0015922
8: -0.0044020, -0.0032327, -0.0044273, -0.0032780, -0.0007006, 0.0008373
9: 0.0018846, 0.0032405, 0.0019372, 0.0032698, -0.0009709, 0.0008123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005317
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005179, upper bound: 0.0005317
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876670, 0.9895522, 0.9876299, 0.9896138, -0.0013011, 0.0012394
1: -0.0043370, -0.0038673, -0.0043463, -0.0038519, -0.0003242, 0.0003088
2: 0.0104405, 0.0129300, 0.0103590, 0.0129789, -0.0016366, 0.0017180
3: -0.0071583, -0.0060252, -0.0071806, -0.0059881, -0.0007820, 0.0007449
4: 0.0025486, 0.0030305, 0.0025329, 0.0030399, -0.0003168, 0.0003325
5: 0.0120908, 0.0152220, 0.0119884, 0.0152835, -0.0020585, 0.0021608
6: -0.0023227, -0.0015280, -0.0023383, -0.0015020, -0.0005484, 0.0005225
7: -0.0091471, -0.0070909, -0.0091875, -0.0070236, -0.0014190, 0.0013518
8: -0.0043745, -0.0032932, -0.0043958, -0.0032578, -0.0007462, 0.0007109
9: 0.0019548, 0.0032086, 0.0019137, 0.0032332, -0.0008243, 0.0008653

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004811, upper bound: 0.0005025
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004933, upper bound: 0.0005025
time: 0.62 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876665, 0.9895525, 0.9876184, 0.9896324, -0.0013170, 0.0012520
1: -0.0043371, -0.0038672, -0.0043491, -0.0038473, -0.0003282, 0.0003120
2: 0.0104401, 0.0129305, 0.0103347, 0.0129940, -0.0016532, 0.0017390
3: -0.0071585, -0.0060250, -0.0071874, -0.0059770, -0.0007915, 0.0007525
4: 0.0025485, 0.0030306, 0.0025281, 0.0030428, -0.0003200, 0.0003366
5: 0.0120904, 0.0152226, 0.0119577, 0.0153024, -0.0020793, 0.0021872
6: -0.0023228, -0.0015278, -0.0023431, -0.0014942, -0.0005551, 0.0005277
7: -0.0091475, -0.0070906, -0.0091999, -0.0070035, -0.0014363, 0.0013654
8: -0.0043747, -0.0032930, -0.0044023, -0.0032472, -0.0007554, 0.0007181
9: 0.0019546, 0.0032089, 0.0019015, 0.0032408, -0.0008326, 0.0008759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0005026
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004941, upper bound: 0.0005026
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876208, 0.9895682, 0.9876063, 0.9896144, -0.0012465, 0.0013321
1: -0.0043485, -0.0038633, -0.0043522, -0.0038518, -0.0003106, 0.0003319
2: 0.0104194, 0.0129908, 0.0103584, 0.0130101, -0.0017591, 0.0016461
3: -0.0071860, -0.0060156, -0.0071948, -0.0059878, -0.0007492, 0.0008006
4: 0.0025445, 0.0030422, 0.0025327, 0.0030460, -0.0003405, 0.0003186
5: 0.0120643, 0.0152985, 0.0119876, 0.0153228, -0.0022124, 0.0020703
6: -0.0023421, -0.0015212, -0.0023483, -0.0015017, -0.0005255, 0.0005615
7: -0.0091973, -0.0070735, -0.0092133, -0.0070231, -0.0013595, 0.0014529
8: -0.0044009, -0.0032840, -0.0044093, -0.0032575, -0.0007150, 0.0007641
9: 0.0019441, 0.0032392, 0.0019134, 0.0032490, -0.0008860, 0.0008290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0005057
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005291, upper bound: 0.0005057
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876202, 0.9895679, 0.9875951, 0.9896328, -0.0012648, 0.0013452
1: -0.0043487, -0.0038634, -0.0043549, -0.0038472, -0.0003152, 0.0003352
2: 0.0104199, 0.0129917, 0.0103340, 0.0130247, -0.0017763, 0.0016702
3: -0.0071864, -0.0060158, -0.0072014, -0.0059767, -0.0007602, 0.0008085
4: 0.0025446, 0.0030424, 0.0025280, 0.0030488, -0.0003438, 0.0003233
5: 0.0120650, 0.0152995, 0.0119569, 0.0153410, -0.0022341, 0.0021006
6: -0.0023424, -0.0015214, -0.0023529, -0.0014940, -0.0005332, 0.0005670
7: -0.0091980, -0.0070739, -0.0092253, -0.0070029, -0.0013794, 0.0014671
8: -0.0044013, -0.0032842, -0.0044156, -0.0032469, -0.0007254, 0.0007715
9: 0.0019444, 0.0032397, 0.0019011, 0.0032563, -0.0008946, 0.0008412

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005059
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005318, upper bound: 0.0005059
time: 0.60 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876670, 0.9895522, 0.9876136, 0.9896492, -0.0013372, 0.0012760
1: -0.0043370, -0.0038673, -0.0043503, -0.0038431, -0.0003332, 0.0003179
2: 0.0104405, 0.0129300, 0.0103123, 0.0130005, -0.0016849, 0.0017658
3: -0.0071583, -0.0060252, -0.0071904, -0.0059669, -0.0008037, 0.0007669
4: 0.0025486, 0.0030305, 0.0025238, 0.0030441, -0.0003261, 0.0003418
5: 0.0120908, 0.0152220, 0.0119297, 0.0153107, -0.0021192, 0.0022209
6: -0.0023227, -0.0015280, -0.0023452, -0.0014870, -0.0005637, 0.0005379
7: -0.0091471, -0.0070909, -0.0092053, -0.0069851, -0.0014584, 0.0013916
8: -0.0043745, -0.0032932, -0.0044051, -0.0032375, -0.0007670, 0.0007318
9: 0.0019548, 0.0032086, 0.0018902, 0.0032441, -0.0008486, 0.0008893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004945, upper bound: 0.0005025
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005035, upper bound: 0.0005025
time: 0.59 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876665, 0.9895525, 0.9876071, 0.9896647, -0.0013556, 0.0013015
1: -0.0043371, -0.0038672, -0.0043519, -0.0038392, -0.0003378, 0.0003243
2: 0.0104401, 0.0129305, 0.0102920, 0.0130091, -0.0017186, 0.0017901
3: -0.0071585, -0.0060250, -0.0071943, -0.0059576, -0.0008148, 0.0007822
4: 0.0025485, 0.0030306, 0.0025199, 0.0030458, -0.0003326, 0.0003465
5: 0.0120904, 0.0152226, 0.0119041, 0.0153214, -0.0021616, 0.0022514
6: -0.0023228, -0.0015278, -0.0023479, -0.0014805, -0.0005714, 0.0005486
7: -0.0091475, -0.0070906, -0.0092124, -0.0069682, -0.0014785, 0.0014195
8: -0.0043747, -0.0032930, -0.0044088, -0.0032287, -0.0007775, 0.0007465
9: 0.0019546, 0.0032089, 0.0018800, 0.0032484, -0.0008656, 0.0009016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0005025
time: 0.58 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005052, upper bound: 0.0005025
time: 0.64 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876208, 0.9895682, 0.9875914, 0.9896497, -0.0013176, 0.0013599
1: -0.0043485, -0.0038633, -0.0043558, -0.0038430, -0.0003283, 0.0003388
2: 0.0104194, 0.0129908, 0.0103117, 0.0130296, -0.0017957, 0.0017399
3: -0.0071860, -0.0060156, -0.0072036, -0.0059666, -0.0007919, 0.0008173
4: 0.0025445, 0.0030422, 0.0025237, 0.0030497, -0.0003476, 0.0003368
5: 0.0120643, 0.0152985, 0.0119289, 0.0153473, -0.0022585, 0.0021883
6: -0.0023421, -0.0015212, -0.0023545, -0.0014868, -0.0005554, 0.0005732
7: -0.0091973, -0.0070735, -0.0092293, -0.0069846, -0.0014370, 0.0014831
8: -0.0044009, -0.0032840, -0.0044178, -0.0032373, -0.0007557, 0.0007800
9: 0.0019441, 0.0032392, 0.0018899, 0.0032588, -0.0009044, 0.0008763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005374, upper bound: 0.0005058
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0005058
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876202, 0.9895679, 0.9875820, 0.9896652, -0.0013317, 0.0013825
1: -0.0043487, -0.0038634, -0.0043582, -0.0038391, -0.0003318, 0.0003445
2: 0.0104199, 0.0129917, 0.0102913, 0.0130422, -0.0018256, 0.0017584
3: -0.0071864, -0.0060158, -0.0072093, -0.0059573, -0.0008004, 0.0008310
4: 0.0025446, 0.0030424, 0.0025197, 0.0030522, -0.0003533, 0.0003403
5: 0.0120650, 0.0152995, 0.0119032, 0.0153631, -0.0022962, 0.0022117
6: -0.0023424, -0.0015214, -0.0023585, -0.0014803, -0.0005613, 0.0005828
7: -0.0091980, -0.0070739, -0.0092397, -0.0069677, -0.0014524, 0.0015079
8: -0.0044013, -0.0032842, -0.0044232, -0.0032284, -0.0007638, 0.0007930
9: 0.0019444, 0.0032397, 0.0018796, 0.0032651, -0.0009195, 0.0008856

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005440, upper bound: 0.0005059
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005418, upper bound: 0.0005059
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876716, 0.9896285, 0.9876299, 0.9896138, -0.0011877, 0.0012125
1: -0.0043359, -0.0038483, -0.0043463, -0.0038519, -0.0002959, 0.0003021
2: 0.0103399, 0.0129238, 0.0103590, 0.0129789, -0.0016010, 0.0015683
3: -0.0071555, -0.0059794, -0.0071806, -0.0059881, -0.0007138, 0.0007287
4: 0.0025291, 0.0030293, 0.0025329, 0.0030399, -0.0003099, 0.0003035
5: 0.0119643, 0.0152142, 0.0119884, 0.0152835, -0.0020137, 0.0019726
6: -0.0023207, -0.0014958, -0.0023383, -0.0015020, -0.0005007, 0.0005111
7: -0.0091420, -0.0070078, -0.0091875, -0.0070236, -0.0012953, 0.0013224
8: -0.0043718, -0.0032495, -0.0043958, -0.0032578, -0.0006812, 0.0006954
9: 0.0019041, 0.0032055, 0.0019137, 0.0032332, -0.0008064, 0.0007899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004738, upper bound: 0.0005276
time: 0.67 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004869, upper bound: 0.0005276
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876713, 0.9896333, 0.9876184, 0.9896324, -0.0012075, 0.0012163
1: -0.0043360, -0.0038471, -0.0043491, -0.0038473, -0.0003009, 0.0003031
2: 0.0103334, 0.0129243, 0.0103347, 0.0129940, -0.0016061, 0.0015945
3: -0.0071557, -0.0059764, -0.0071874, -0.0059770, -0.0007258, 0.0007310
4: 0.0025279, 0.0030294, 0.0025281, 0.0030428, -0.0003109, 0.0003086
5: 0.0119561, 0.0152149, 0.0119577, 0.0153024, -0.0020201, 0.0020055
6: -0.0023209, -0.0014938, -0.0023431, -0.0014942, -0.0005090, 0.0005127
7: -0.0091424, -0.0070024, -0.0091999, -0.0070035, -0.0013170, 0.0013266
8: -0.0043720, -0.0032466, -0.0044023, -0.0032472, -0.0006926, 0.0006976
9: 0.0019008, 0.0032057, 0.0019015, 0.0032408, -0.0008089, 0.0008031

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004746, upper bound: 0.0005279
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004845, upper bound: 0.0005279
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876197, 0.9896494, 0.9876063, 0.9896144, -0.0011383, 0.0013086
1: -0.0043488, -0.0038431, -0.0043522, -0.0038518, -0.0002836, 0.0003261
2: 0.0103122, 0.0129924, 0.0103584, 0.0130101, -0.0017280, 0.0015031
3: -0.0071867, -0.0059668, -0.0071948, -0.0059878, -0.0006841, 0.0007865
4: 0.0025238, 0.0030425, 0.0025327, 0.0030460, -0.0003344, 0.0002909
5: 0.0119295, 0.0153005, 0.0119876, 0.0153228, -0.0021733, 0.0018905
6: -0.0023426, -0.0014870, -0.0023483, -0.0015017, -0.0004798, 0.0005516
7: -0.0091986, -0.0069849, -0.0092133, -0.0070231, -0.0012415, 0.0014272
8: -0.0044016, -0.0032375, -0.0044093, -0.0032575, -0.0006529, 0.0007505
9: 0.0018901, 0.0032400, 0.0019134, 0.0032490, -0.0008703, 0.0007570

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005323
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005125, upper bound: 0.0005323
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876189, 0.9896577, 0.9875951, 0.9896328, -0.0011532, 0.0013089
1: -0.0043490, -0.0038410, -0.0043549, -0.0038472, -0.0002873, 0.0003261
2: 0.0103012, 0.0129934, 0.0103340, 0.0130247, -0.0017284, 0.0015227
3: -0.0071872, -0.0059618, -0.0072014, -0.0059767, -0.0006931, 0.0007867
4: 0.0025217, 0.0030427, 0.0025280, 0.0030488, -0.0003345, 0.0002947
5: 0.0119157, 0.0153017, 0.0119569, 0.0153410, -0.0021739, 0.0019152
6: -0.0023429, -0.0014835, -0.0023529, -0.0014940, -0.0004861, 0.0005518
7: -0.0091994, -0.0069759, -0.0092253, -0.0070029, -0.0012577, 0.0014275
8: -0.0044020, -0.0032327, -0.0044156, -0.0032469, -0.0006614, 0.0007507
9: 0.0018846, 0.0032405, 0.0019011, 0.0032563, -0.0008705, 0.0007669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005324
time: 0.59 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005081, upper bound: 0.0005324
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876716, 0.9896285, 0.9876136, 0.9896492, -0.0012252, 0.0012461
1: -0.0043359, -0.0038483, -0.0043503, -0.0038431, -0.0003053, 0.0003105
2: 0.0103399, 0.0129238, 0.0103123, 0.0130005, -0.0016455, 0.0016179
3: -0.0071555, -0.0059794, -0.0071904, -0.0059669, -0.0007364, 0.0007490
4: 0.0025291, 0.0030293, 0.0025238, 0.0030441, -0.0003185, 0.0003131
5: 0.0119643, 0.0152142, 0.0119297, 0.0153107, -0.0020696, 0.0020349
6: -0.0023207, -0.0014958, -0.0023452, -0.0014870, -0.0005165, 0.0005253
7: -0.0091420, -0.0070078, -0.0092053, -0.0069851, -0.0013363, 0.0013591
8: -0.0043718, -0.0032495, -0.0044051, -0.0032375, -0.0007027, 0.0007147
9: 0.0019041, 0.0032055, 0.0018902, 0.0032441, -0.0008288, 0.0008149

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.09 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004868, upper bound: 0.0005276
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004966, upper bound: 0.0005276
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876713, 0.9896333, 0.9876071, 0.9896647, -0.0012418, 0.0012521
1: -0.0043360, -0.0038471, -0.0043519, -0.0038392, -0.0003094, 0.0003120
2: 0.0103334, 0.0129243, 0.0102920, 0.0130091, -0.0016534, 0.0016398
3: -0.0071557, -0.0059764, -0.0071943, -0.0059576, -0.0007464, 0.0007525
4: 0.0025279, 0.0030294, 0.0025199, 0.0030458, -0.0003200, 0.0003174
5: 0.0119561, 0.0152149, 0.0119041, 0.0153214, -0.0020795, 0.0020625
6: -0.0023209, -0.0014938, -0.0023479, -0.0014805, -0.0005235, 0.0005278
7: -0.0091424, -0.0070024, -0.0092124, -0.0069682, -0.0013544, 0.0013656
8: -0.0043720, -0.0032466, -0.0044088, -0.0032287, -0.0007123, 0.0007181
9: 0.0019008, 0.0032057, 0.0018800, 0.0032484, -0.0008327, 0.0008259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004886, upper bound: 0.0005278
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004947, upper bound: 0.0005278
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876197, 0.9896494, 0.9875914, 0.9896497, -0.0012031, 0.0013255
1: -0.0043488, -0.0038431, -0.0043558, -0.0038430, -0.0002998, 0.0003303
2: 0.0103122, 0.0129924, 0.0103117, 0.0130296, -0.0017503, 0.0015886
3: -0.0071867, -0.0059668, -0.0072036, -0.0059666, -0.0007231, 0.0007967
4: 0.0025238, 0.0030425, 0.0025237, 0.0030497, -0.0003388, 0.0003075
5: 0.0119295, 0.0153005, 0.0119289, 0.0153473, -0.0022014, 0.0019981
6: -0.0023426, -0.0014870, -0.0023545, -0.0014868, -0.0005071, 0.0005587
7: -0.0091986, -0.0069849, -0.0092293, -0.0069846, -0.0013121, 0.0014456
8: -0.0044016, -0.0032375, -0.0044178, -0.0032373, -0.0006900, 0.0007602
9: 0.0018901, 0.0032400, 0.0018899, 0.0032588, -0.0008815, 0.0008001

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005119, upper bound: 0.0005322
time: 0.62 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005249, upper bound: 0.0005322
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876189, 0.9896577, 0.9875820, 0.9896652, -0.0012190, 0.0013327
1: -0.0043490, -0.0038410, -0.0043582, -0.0038391, -0.0003037, 0.0003321
2: 0.0103012, 0.0129934, 0.0102913, 0.0130422, -0.0017598, 0.0016097
3: -0.0071872, -0.0059618, -0.0072093, -0.0059573, -0.0007327, 0.0008010
4: 0.0025217, 0.0030427, 0.0025197, 0.0030522, -0.0003406, 0.0003116
5: 0.0119157, 0.0153017, 0.0119032, 0.0153631, -0.0022134, 0.0020246
6: -0.0023429, -0.0014835, -0.0023585, -0.0014803, -0.0005139, 0.0005618
7: -0.0091994, -0.0069759, -0.0092397, -0.0069677, -0.0013295, 0.0014535
8: -0.0044020, -0.0032327, -0.0044232, -0.0032284, -0.0006992, 0.0007644
9: 0.0018846, 0.0032405, 0.0018796, 0.0032651, -0.0008863, 0.0008107

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005120, upper bound: 0.0005323
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005202, upper bound: 0.0005323
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876535, 0.9895865, 0.9876142, 0.9895375, -0.0012067, 0.0012630
1: -0.0043404, -0.0038587, -0.0043502, -0.0038709, -0.0003007, 0.0003147
2: 0.0103952, 0.0129477, 0.0104599, 0.0129997, -0.0016677, 0.0015934
3: -0.0071663, -0.0060046, -0.0071900, -0.0060340, -0.0007253, 0.0007591
4: 0.0025399, 0.0030339, 0.0025524, 0.0030439, -0.0003228, 0.0003084
5: 0.0120339, 0.0152442, 0.0121152, 0.0153096, -0.0020976, 0.0020041
6: -0.0023283, -0.0015135, -0.0023449, -0.0015341, -0.0005087, 0.0005324
7: -0.0091617, -0.0070535, -0.0092046, -0.0071069, -0.0013161, 0.0013774
8: -0.0043822, -0.0032735, -0.0044048, -0.0033016, -0.0006921, 0.0007244
9: 0.0019319, 0.0032175, 0.0019645, 0.0032437, -0.0008400, 0.0008025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004731, upper bound: 0.0005184
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004856, upper bound: 0.0005184
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876533, 0.9895874, 0.9876163, 0.9895376, -0.0012027, 0.0012660
1: -0.0043404, -0.0038585, -0.0043497, -0.0038709, -0.0002997, 0.0003155
2: 0.0103941, 0.0129480, 0.0104597, 0.0129969, -0.0016717, 0.0015881
3: -0.0071665, -0.0060041, -0.0071888, -0.0060339, -0.0007228, 0.0007609
4: 0.0025396, 0.0030340, 0.0025523, 0.0030434, -0.0003236, 0.0003074
5: 0.0120325, 0.0152446, 0.0121150, 0.0153062, -0.0021026, 0.0019974
6: -0.0023284, -0.0015131, -0.0023440, -0.0015341, -0.0005070, 0.0005337
7: -0.0091620, -0.0070526, -0.0092023, -0.0071068, -0.0013117, 0.0013808
8: -0.0043823, -0.0032730, -0.0044036, -0.0033015, -0.0006898, 0.0007261
9: 0.0019314, 0.0032177, 0.0019644, 0.0032423, -0.0008420, 0.0007999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0005185
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004826, upper bound: 0.0005185
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876592, 0.9896678, 0.9876142, 0.9895375, -0.0012676, 0.0014028
1: -0.0043390, -0.0038385, -0.0043502, -0.0038709, -0.0003158, 0.0003495
2: 0.0102878, 0.0129403, 0.0104599, 0.0129997, -0.0018524, 0.0016738
3: -0.0071630, -0.0059557, -0.0071900, -0.0060340, -0.0007618, 0.0008431
4: 0.0025191, 0.0030324, 0.0025524, 0.0030439, -0.0003585, 0.0003240
5: 0.0118988, 0.0152349, 0.0121152, 0.0153096, -0.0023298, 0.0021052
6: -0.0023259, -0.0014792, -0.0023449, -0.0015341, -0.0005343, 0.0005913
7: -0.0091555, -0.0069648, -0.0092046, -0.0071069, -0.0013824, 0.0015299
8: -0.0043790, -0.0032269, -0.0044048, -0.0033016, -0.0007270, 0.0008046
9: 0.0018779, 0.0032138, 0.0019645, 0.0032437, -0.0009329, 0.0008430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.80 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004726, upper bound: 0.0005368
time: 0.67 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004850, upper bound: 0.0005368
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876588, 0.9896718, 0.9876163, 0.9895376, -0.0012540, 0.0014130
1: -0.0043391, -0.0038375, -0.0043497, -0.0038709, -0.0003125, 0.0003521
2: 0.0102826, 0.0129407, 0.0104597, 0.0129969, -0.0018658, 0.0016558
3: -0.0071632, -0.0059533, -0.0071888, -0.0060339, -0.0007537, 0.0008492
4: 0.0025181, 0.0030325, 0.0025523, 0.0030434, -0.0003611, 0.0003205
5: 0.0118922, 0.0152355, 0.0121150, 0.0153062, -0.0023467, 0.0020826
6: -0.0023261, -0.0014775, -0.0023440, -0.0015341, -0.0005286, 0.0005956
7: -0.0091559, -0.0069605, -0.0092023, -0.0071068, -0.0013676, 0.0015411
8: -0.0043792, -0.0032246, -0.0044036, -0.0033015, -0.0007192, 0.0008104
9: 0.0018752, 0.0032140, 0.0019644, 0.0032423, -0.0009397, 0.0008340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004725, upper bound: 0.0005369
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004821, upper bound: 0.0005369
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876535, 0.9895865, 0.9875999, 0.9895701, -0.0011898, 0.0012113
1: -0.0043404, -0.0038587, -0.0043537, -0.0038628, -0.0002965, 0.0003018
2: 0.0103952, 0.0129477, 0.0104169, 0.0130184, -0.0015995, 0.0015711
3: -0.0071663, -0.0060046, -0.0071985, -0.0060144, -0.0007151, 0.0007280
4: 0.0025399, 0.0030339, 0.0025441, 0.0030476, -0.0003096, 0.0003041
5: 0.0120339, 0.0152442, 0.0120612, 0.0153332, -0.0020118, 0.0019761
6: -0.0023283, -0.0015135, -0.0023509, -0.0015204, -0.0005016, 0.0005106
7: -0.0091617, -0.0070535, -0.0092201, -0.0070714, -0.0012977, 0.0013211
8: -0.0043822, -0.0032735, -0.0044129, -0.0032829, -0.0006824, 0.0006948
9: 0.0019319, 0.0032175, 0.0019429, 0.0032531, -0.0008056, 0.0007913

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004731, upper bound: 0.0005184
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004856, upper bound: 0.0005184
time: 0.61 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876533, 0.9895874, 0.9876001, 0.9895779, -0.0011836, 0.0012135
1: -0.0043404, -0.0038585, -0.0043537, -0.0038608, -0.0002949, 0.0003024
2: 0.0103941, 0.0129480, 0.0104064, 0.0130182, -0.0016024, 0.0015629
3: -0.0071665, -0.0060041, -0.0071985, -0.0060097, -0.0007114, 0.0007293
4: 0.0025396, 0.0030340, 0.0025420, 0.0030475, -0.0003101, 0.0003025
5: 0.0120325, 0.0152446, 0.0120480, 0.0153329, -0.0020153, 0.0019657
6: -0.0023284, -0.0015131, -0.0023508, -0.0015171, -0.0004989, 0.0005115
7: -0.0091620, -0.0070526, -0.0092199, -0.0070628, -0.0012909, 0.0013235
8: -0.0043823, -0.0032730, -0.0044128, -0.0032784, -0.0006788, 0.0006960
9: 0.0019314, 0.0032177, 0.0019376, 0.0032530, -0.0008070, 0.0007872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0005184
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004826, upper bound: 0.0005185
time: 0.60 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876592, 0.9896678, 0.9875999, 0.9895701, -0.0012370, 0.0013506
1: -0.0043390, -0.0038385, -0.0043537, -0.0038628, -0.0003082, 0.0003365
2: 0.0102878, 0.0129403, 0.0104169, 0.0130184, -0.0017834, 0.0016334
3: -0.0071630, -0.0059557, -0.0071985, -0.0060144, -0.0007434, 0.0008117
4: 0.0025191, 0.0030324, 0.0025441, 0.0030476, -0.0003452, 0.0003161
5: 0.0118988, 0.0152349, 0.0120612, 0.0153332, -0.0022430, 0.0020544
6: -0.0023259, -0.0014792, -0.0023509, -0.0015204, -0.0005214, 0.0005693
7: -0.0091555, -0.0069648, -0.0092201, -0.0070714, -0.0013491, 0.0014730
8: -0.0043790, -0.0032269, -0.0044129, -0.0032829, -0.0007095, 0.0007746
9: 0.0018779, 0.0032138, 0.0019429, 0.0032531, -0.0008982, 0.0008227

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004726, upper bound: 0.0005368
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004850, upper bound: 0.0005368
time: 0.65 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876588, 0.9896718, 0.9876001, 0.9895779, -0.0012292, 0.0013545
1: -0.0043391, -0.0038375, -0.0043537, -0.0038608, -0.0003063, 0.0003375
2: 0.0102826, 0.0129407, 0.0104064, 0.0130182, -0.0017886, 0.0016231
3: -0.0071632, -0.0059533, -0.0071985, -0.0060097, -0.0007388, 0.0008141
4: 0.0025181, 0.0030325, 0.0025420, 0.0030475, -0.0003462, 0.0003141
5: 0.0118922, 0.0152355, 0.0120480, 0.0153329, -0.0022496, 0.0020414
6: -0.0023261, -0.0014775, -0.0023508, -0.0015171, -0.0005181, 0.0005710
7: -0.0091559, -0.0069605, -0.0092199, -0.0070628, -0.0013406, 0.0014773
8: -0.0043792, -0.0032246, -0.0044128, -0.0032784, -0.0007050, 0.0007769
9: 0.0018752, 0.0032140, 0.0019376, 0.0032530, -0.0009008, 0.0008175

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004725, upper bound: 0.0005369
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004821, upper bound: 0.0005369
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9875984, 0.9896046, 0.9875922, 0.9895381, -0.0011579, 0.0013328
1: -0.0043541, -0.0038542, -0.0043556, -0.0038708, -0.0002885, 0.0003321
2: 0.0103713, 0.0130204, 0.0104592, 0.0130286, -0.0017600, 0.0015290
3: -0.0071995, -0.0059937, -0.0072032, -0.0060337, -0.0006959, 0.0008011
4: 0.0025352, 0.0030480, 0.0025522, 0.0030496, -0.0003406, 0.0002959
5: 0.0120038, 0.0153357, 0.0121144, 0.0153461, -0.0022136, 0.0019231
6: -0.0023515, -0.0015059, -0.0023542, -0.0015339, -0.0004881, 0.0005618
7: -0.0092218, -0.0070337, -0.0092286, -0.0071063, -0.0012629, 0.0014537
8: -0.0044138, -0.0032631, -0.0044174, -0.0033013, -0.0006641, 0.0007645
9: 0.0019199, 0.0032541, 0.0019642, 0.0032583, -0.0008864, 0.0007701

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005066, upper bound: 0.0005220
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0005220
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9875982, 0.9896081, 0.9875958, 0.9895383, -0.0011534, 0.0013352
1: -0.0043542, -0.0038533, -0.0043548, -0.0038707, -0.0002874, 0.0003327
2: 0.0103667, 0.0130208, 0.0104589, 0.0130239, -0.0017632, 0.0015230
3: -0.0071996, -0.0059916, -0.0072010, -0.0060336, -0.0006932, 0.0008025
4: 0.0025343, 0.0030480, 0.0025522, 0.0030486, -0.0003413, 0.0002948
5: 0.0119980, 0.0153362, 0.0121140, 0.0153401, -0.0022176, 0.0019156
6: -0.0023517, -0.0015044, -0.0023526, -0.0015338, -0.0004862, 0.0005628
7: -0.0092221, -0.0070300, -0.0092246, -0.0071061, -0.0012579, 0.0014563
8: -0.0044140, -0.0032611, -0.0044153, -0.0033012, -0.0006615, 0.0007658
9: 0.0019176, 0.0032543, 0.0019640, 0.0032559, -0.0008880, 0.0007671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005070, upper bound: 0.0005221
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005221
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876060, 0.9896855, 0.9875922, 0.9895381, -0.0012168, 0.0014756
1: -0.0043522, -0.0038341, -0.0043556, -0.0038708, -0.0003032, 0.0003677
2: 0.0102645, 0.0130105, 0.0104592, 0.0130286, -0.0019485, 0.0016068
3: -0.0071949, -0.0059451, -0.0072032, -0.0060337, -0.0007313, 0.0008869
4: 0.0025146, 0.0030460, 0.0025522, 0.0030496, -0.0003771, 0.0003110
5: 0.0118695, 0.0153232, 0.0121144, 0.0153461, -0.0024507, 0.0020209
6: -0.0023484, -0.0014718, -0.0023542, -0.0015339, -0.0005129, 0.0006220
7: -0.0092136, -0.0069456, -0.0092286, -0.0071063, -0.0013271, 0.0016094
8: -0.0044095, -0.0032167, -0.0044174, -0.0033013, -0.0006979, 0.0008463
9: 0.0018661, 0.0032491, 0.0019642, 0.0032583, -0.0009814, 0.0008093

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005418
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005101, upper bound: 0.0005418
time: 0.70 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876057, 0.9896920, 0.9875958, 0.9895383, -0.0012028, 0.0014865
1: -0.0043523, -0.0038324, -0.0043548, -0.0038707, -0.0002997, 0.0003704
2: 0.0102559, 0.0130109, 0.0104589, 0.0130239, -0.0019629, 0.0015882
3: -0.0071951, -0.0059411, -0.0072010, -0.0060336, -0.0007229, 0.0008934
4: 0.0025129, 0.0030461, 0.0025522, 0.0030486, -0.0003799, 0.0003074
5: 0.0118586, 0.0153238, 0.0121140, 0.0153401, -0.0024688, 0.0019976
6: -0.0023485, -0.0014690, -0.0023526, -0.0015338, -0.0005070, 0.0006266
7: -0.0092139, -0.0069384, -0.0092246, -0.0071061, -0.0013118, 0.0016212
8: -0.0044097, -0.0032130, -0.0044153, -0.0033012, -0.0006899, 0.0008526
9: 0.0018618, 0.0032494, 0.0019640, 0.0032559, -0.0009886, 0.0007999

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004990, upper bound: 0.0005419
time: 0.59 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005059, upper bound: 0.0005419
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9875984, 0.9896046, 0.9875783, 0.9895706, -0.0011364, 0.0013091
1: -0.0043541, -0.0038542, -0.0043591, -0.0038627, -0.0002832, 0.0003262
2: 0.0103713, 0.0130204, 0.0104163, 0.0130469, -0.0017286, 0.0015006
3: -0.0071995, -0.0059937, -0.0072115, -0.0060142, -0.0006830, 0.0007868
4: 0.0025352, 0.0030480, 0.0025439, 0.0030531, -0.0003346, 0.0002904
5: 0.0120038, 0.0153357, 0.0120604, 0.0153691, -0.0021742, 0.0018874
6: -0.0023515, -0.0015059, -0.0023600, -0.0015202, -0.0004790, 0.0005518
7: -0.0092218, -0.0070337, -0.0092437, -0.0070709, -0.0012394, 0.0014277
8: -0.0044138, -0.0032631, -0.0044253, -0.0032827, -0.0006518, 0.0007508
9: 0.0019199, 0.0032541, 0.0019426, 0.0032675, -0.0008706, 0.0007558

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.15 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005066, upper bound: 0.0005220
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0005220
time: 0.64 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9875982, 0.9896081, 0.9875749, 0.9895787, -0.0011313, 0.0013084
1: -0.0043542, -0.0038533, -0.0043600, -0.0038607, -0.0002819, 0.0003260
2: 0.0103667, 0.0130208, 0.0104056, 0.0130515, -0.0017278, 0.0014939
3: -0.0071996, -0.0059916, -0.0072136, -0.0060093, -0.0006799, 0.0007864
4: 0.0025343, 0.0030480, 0.0025419, 0.0030540, -0.0003344, 0.0002891
5: 0.0119980, 0.0153362, 0.0120470, 0.0153748, -0.0021731, 0.0018789
6: -0.0023517, -0.0015044, -0.0023615, -0.0015168, -0.0004769, 0.0005516
7: -0.0092221, -0.0070300, -0.0092474, -0.0070621, -0.0012338, 0.0014270
8: -0.0044140, -0.0032611, -0.0044273, -0.0032780, -0.0006489, 0.0007505
9: 0.0019176, 0.0032543, 0.0019372, 0.0032698, -0.0008702, 0.0007524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0005221
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005094, upper bound: 0.0005221
time: 0.62 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876060, 0.9896855, 0.9875783, 0.9895706, -0.0011818, 0.0014442
1: -0.0043522, -0.0038341, -0.0043591, -0.0038627, -0.0002945, 0.0003599
2: 0.0102645, 0.0130105, 0.0104163, 0.0130469, -0.0019071, 0.0015606
3: -0.0071949, -0.0059451, -0.0072115, -0.0060142, -0.0007103, 0.0008680
4: 0.0025146, 0.0030460, 0.0025439, 0.0030531, -0.0003691, 0.0003021
5: 0.0118695, 0.0153232, 0.0120604, 0.0153691, -0.0023986, 0.0019628
6: -0.0023484, -0.0014718, -0.0023600, -0.0015202, -0.0004982, 0.0006088
7: -0.0092136, -0.0069456, -0.0092437, -0.0070709, -0.0012890, 0.0015751
8: -0.0044095, -0.0032167, -0.0044253, -0.0032827, -0.0006779, 0.0008283
9: 0.0018661, 0.0032491, 0.0019426, 0.0032675, -0.0009605, 0.0007860

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005418
time: 0.64 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005101, upper bound: 0.0005418
time: 0.66 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876057, 0.9896920, 0.9875749, 0.9895787, -0.0011747, 0.0014498
1: -0.0043523, -0.0038324, -0.0043600, -0.0038607, -0.0002927, 0.0003613
2: 0.0102559, 0.0130109, 0.0104056, 0.0130515, -0.0019145, 0.0015512
3: -0.0071951, -0.0059411, -0.0072136, -0.0060093, -0.0007060, 0.0008714
4: 0.0025129, 0.0030461, 0.0025419, 0.0030540, -0.0003705, 0.0003002
5: 0.0118586, 0.0153238, 0.0120470, 0.0153748, -0.0024079, 0.0019510
6: -0.0023485, -0.0014690, -0.0023615, -0.0015168, -0.0004952, 0.0006112
7: -0.0092139, -0.0069384, -0.0092474, -0.0070621, -0.0012812, 0.0015812
8: -0.0044097, -0.0032130, -0.0044273, -0.0032780, -0.0006738, 0.0008316
9: 0.0018618, 0.0032494, 0.0019372, 0.0032698, -0.0009642, 0.0007812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004986, upper bound: 0.0005419
time: 0.68 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005057, upper bound: 0.0005419
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876535, 0.9895865, 0.9876299, 0.9896138, -0.0013375, 0.0013049
1: -0.0043404, -0.0038587, -0.0043463, -0.0038519, -0.0003333, 0.0003251
2: 0.0103952, 0.0129477, 0.0103590, 0.0129789, -0.0017231, 0.0017661
3: -0.0071663, -0.0060046, -0.0071806, -0.0059881, -0.0008038, 0.0007843
4: 0.0025399, 0.0030339, 0.0025329, 0.0030399, -0.0003335, 0.0003418
5: 0.0120339, 0.0152442, 0.0119884, 0.0152835, -0.0021672, 0.0022213
6: -0.0023283, -0.0015135, -0.0023383, -0.0015020, -0.0005638, 0.0005501
7: -0.0091617, -0.0070535, -0.0091875, -0.0070236, -0.0014587, 0.0014232
8: -0.0043822, -0.0032735, -0.0043958, -0.0032578, -0.0007671, 0.0007484
9: 0.0019319, 0.0032175, 0.0019137, 0.0032332, -0.0008678, 0.0008895

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004805, upper bound: 0.0005137
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004912, upper bound: 0.0005137
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876533, 0.9895874, 0.9876184, 0.9896324, -0.0013551, 0.0013183
1: -0.0043404, -0.0038585, -0.0043491, -0.0038473, -0.0003377, 0.0003285
2: 0.0103941, 0.0129480, 0.0103347, 0.0129940, -0.0017408, 0.0017894
3: -0.0071665, -0.0060041, -0.0071874, -0.0059770, -0.0008145, 0.0007923
4: 0.0025396, 0.0030340, 0.0025281, 0.0030428, -0.0003369, 0.0003463
5: 0.0120325, 0.0152446, 0.0119577, 0.0153024, -0.0021895, 0.0022506
6: -0.0023284, -0.0015131, -0.0023431, -0.0014942, -0.0005712, 0.0005557
7: -0.0091620, -0.0070526, -0.0091999, -0.0070035, -0.0014779, 0.0014378
8: -0.0043823, -0.0032730, -0.0044023, -0.0032472, -0.0007772, 0.0007561
9: 0.0019314, 0.0032177, 0.0019015, 0.0032408, -0.0008768, 0.0009012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004890, upper bound: 0.0005142
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004939, upper bound: 0.0005142
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9875984, 0.9896046, 0.9876063, 0.9896144, -0.0012871, 0.0013718
1: -0.0043541, -0.0038542, -0.0043522, -0.0038518, -0.0003207, 0.0003418
2: 0.0103713, 0.0130204, 0.0103584, 0.0130101, -0.0018114, 0.0016996
3: -0.0071995, -0.0059937, -0.0071948, -0.0059878, -0.0007736, 0.0008245
4: 0.0025352, 0.0030480, 0.0025327, 0.0030460, -0.0003506, 0.0003289
5: 0.0120038, 0.0153357, 0.0119876, 0.0153228, -0.0022783, 0.0021376
6: -0.0023515, -0.0015059, -0.0023483, -0.0015017, -0.0005425, 0.0005782
7: -0.0092218, -0.0070337, -0.0092133, -0.0070231, -0.0014037, 0.0014961
8: -0.0044138, -0.0032631, -0.0044093, -0.0032575, -0.0007382, 0.0007868
9: 0.0019199, 0.0032541, 0.0019134, 0.0032490, -0.0009123, 0.0008560

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0005175
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005291, upper bound: 0.0005175
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9875982, 0.9896081, 0.9875951, 0.9896328, -0.0013060, 0.0013874
1: -0.0043542, -0.0038533, -0.0043549, -0.0038472, -0.0003254, 0.0003457
2: 0.0103667, 0.0130208, 0.0103340, 0.0130247, -0.0018321, 0.0017246
3: -0.0071996, -0.0059916, -0.0072014, -0.0059767, -0.0007850, 0.0008339
4: 0.0025343, 0.0030480, 0.0025280, 0.0030488, -0.0003546, 0.0003338
5: 0.0119980, 0.0153362, 0.0119569, 0.0153410, -0.0023042, 0.0021691
6: -0.0023517, -0.0015044, -0.0023529, -0.0014940, -0.0005505, 0.0005848
7: -0.0092221, -0.0070300, -0.0092253, -0.0070029, -0.0014244, 0.0015132
8: -0.0044140, -0.0032611, -0.0044156, -0.0032469, -0.0007491, 0.0007958
9: 0.0019176, 0.0032543, 0.0019011, 0.0032563, -0.0009227, 0.0008686

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005180
time: 0.59 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005180
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876535, 0.9895865, 0.9876136, 0.9896492, -0.0013202, 0.0012525
1: -0.0043404, -0.0038587, -0.0043503, -0.0038431, -0.0003290, 0.0003121
2: 0.0103952, 0.0129477, 0.0103123, 0.0130005, -0.0016539, 0.0017433
3: -0.0071663, -0.0060046, -0.0071904, -0.0059669, -0.0007935, 0.0007528
4: 0.0025399, 0.0030339, 0.0025238, 0.0030441, -0.0003201, 0.0003374
5: 0.0120339, 0.0152442, 0.0119297, 0.0153107, -0.0020802, 0.0021926
6: -0.0023283, -0.0015135, -0.0023452, -0.0014870, -0.0005565, 0.0005280
7: -0.0091617, -0.0070535, -0.0092053, -0.0069851, -0.0014399, 0.0013660
8: -0.0043822, -0.0032735, -0.0044051, -0.0032375, -0.0007572, 0.0007184
9: 0.0019319, 0.0032175, 0.0018902, 0.0032441, -0.0008330, 0.0008780

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004805, upper bound: 0.0005138
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004912, upper bound: 0.0005137
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876533, 0.9895874, 0.9876071, 0.9896647, -0.0013325, 0.0012651
1: -0.0043404, -0.0038585, -0.0043519, -0.0038392, -0.0003320, 0.0003152
2: 0.0103941, 0.0129480, 0.0102920, 0.0130091, -0.0016706, 0.0017596
3: -0.0071665, -0.0060041, -0.0071943, -0.0059576, -0.0008009, 0.0007604
4: 0.0025396, 0.0030340, 0.0025199, 0.0030458, -0.0003233, 0.0003406
5: 0.0120325, 0.0152446, 0.0119041, 0.0153214, -0.0021012, 0.0022131
6: -0.0023284, -0.0015131, -0.0023479, -0.0014805, -0.0005617, 0.0005333
7: -0.0091620, -0.0070526, -0.0092124, -0.0069682, -0.0014533, 0.0013798
8: -0.0043823, -0.0032730, -0.0044088, -0.0032287, -0.0007643, 0.0007256
9: 0.0019314, 0.0032177, 0.0018800, 0.0032484, -0.0008414, 0.0008862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004890, upper bound: 0.0005142
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004939, upper bound: 0.0005142
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9875984, 0.9896046, 0.9875914, 0.9896497, -0.0012664, 0.0013474
1: -0.0043541, -0.0038542, -0.0043558, -0.0038430, -0.0003156, 0.0003357
2: 0.0103713, 0.0130204, 0.0103117, 0.0130296, -0.0017792, 0.0016723
3: -0.0071995, -0.0059937, -0.0072036, -0.0059666, -0.0007612, 0.0008098
4: 0.0025352, 0.0030480, 0.0025237, 0.0030497, -0.0003444, 0.0003237
5: 0.0120038, 0.0153357, 0.0119289, 0.0153473, -0.0022378, 0.0021033
6: -0.0023515, -0.0015059, -0.0023545, -0.0014868, -0.0005338, 0.0005680
7: -0.0092218, -0.0070337, -0.0092293, -0.0069846, -0.0013812, 0.0014695
8: -0.0044138, -0.0032631, -0.0044178, -0.0032373, -0.0007264, 0.0007728
9: 0.0019199, 0.0032541, 0.0018899, 0.0032588, -0.0008961, 0.0008423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005246, upper bound: 0.0005175
time: 0.61 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005290, upper bound: 0.0005175
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9875982, 0.9896081, 0.9875820, 0.9896652, -0.0012817, 0.0013593
1: -0.0043542, -0.0038533, -0.0043582, -0.0038391, -0.0003194, 0.0003387
2: 0.0103667, 0.0130208, 0.0102913, 0.0130422, -0.0017950, 0.0016925
3: -0.0071996, -0.0059916, -0.0072093, -0.0059573, -0.0007703, 0.0008170
4: 0.0025343, 0.0030480, 0.0025197, 0.0030522, -0.0003474, 0.0003276
5: 0.0119980, 0.0153362, 0.0119032, 0.0153631, -0.0022576, 0.0021287
6: -0.0023517, -0.0015044, -0.0023585, -0.0014803, -0.0005403, 0.0005730
7: -0.0092221, -0.0070300, -0.0092397, -0.0069677, -0.0013979, 0.0014826
8: -0.0044140, -0.0032611, -0.0044232, -0.0032284, -0.0007351, 0.0007797
9: 0.0019176, 0.0032543, 0.0018796, 0.0032651, -0.0009041, 0.0008524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005180
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005180
time: 0.59 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876592, 0.9896678, 0.9876299, 0.9896138, -0.0012243, 0.0012782
1: -0.0043390, -0.0038385, -0.0043463, -0.0038519, -0.0003051, 0.0003185
2: 0.0102878, 0.0129403, 0.0103590, 0.0129789, -0.0016878, 0.0016167
3: -0.0071630, -0.0059557, -0.0071806, -0.0059881, -0.0007359, 0.0007682
4: 0.0025191, 0.0030324, 0.0025329, 0.0030399, -0.0003267, 0.0003129
5: 0.0118988, 0.0152349, 0.0119884, 0.0152835, -0.0021228, 0.0020334
6: -0.0023259, -0.0014792, -0.0023383, -0.0015020, -0.0005161, 0.0005388
7: -0.0091555, -0.0069648, -0.0091875, -0.0070236, -0.0013353, 0.0013940
8: -0.0043790, -0.0032269, -0.0043958, -0.0032578, -0.0007022, 0.0007331
9: 0.0018779, 0.0032138, 0.0019137, 0.0032332, -0.0008501, 0.0008143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004736, upper bound: 0.0005376
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005376
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876588, 0.9896718, 0.9876184, 0.9896324, -0.0012445, 0.0012820
1: -0.0043391, -0.0038375, -0.0043491, -0.0038473, -0.0003101, 0.0003194
2: 0.0102826, 0.0129407, 0.0103347, 0.0129940, -0.0016928, 0.0016434
3: -0.0071632, -0.0059533, -0.0071874, -0.0059770, -0.0007480, 0.0007705
4: 0.0025181, 0.0030325, 0.0025281, 0.0030428, -0.0003276, 0.0003181
5: 0.0118922, 0.0152355, 0.0119577, 0.0153024, -0.0021291, 0.0020669
6: -0.0023261, -0.0014775, -0.0023431, -0.0014942, -0.0005246, 0.0005404
7: -0.0091559, -0.0069605, -0.0091999, -0.0070035, -0.0013573, 0.0013982
8: -0.0043792, -0.0032246, -0.0044023, -0.0032472, -0.0007138, 0.0007353
9: 0.0018752, 0.0032140, 0.0019015, 0.0032408, -0.0008526, 0.0008277

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004737, upper bound: 0.0005379
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004832, upper bound: 0.0005379
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876060, 0.9896855, 0.9876063, 0.9896144, -0.0011793, 0.0013478
1: -0.0043522, -0.0038341, -0.0043522, -0.0038518, -0.0002939, 0.0003358
2: 0.0102645, 0.0130105, 0.0103584, 0.0130101, -0.0017797, 0.0015573
3: -0.0071949, -0.0059451, -0.0071948, -0.0059878, -0.0007088, 0.0008101
4: 0.0025146, 0.0030460, 0.0025327, 0.0030460, -0.0003445, 0.0003014
5: 0.0118695, 0.0153232, 0.0119876, 0.0153228, -0.0022384, 0.0019587
6: -0.0023484, -0.0014718, -0.0023483, -0.0015017, -0.0004971, 0.0005681
7: -0.0092136, -0.0069456, -0.0092133, -0.0070231, -0.0012862, 0.0014700
8: -0.0044095, -0.0032167, -0.0044093, -0.0032575, -0.0006764, 0.0007730
9: 0.0018661, 0.0032491, 0.0019134, 0.0032490, -0.0008964, 0.0007843

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005426
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005121, upper bound: 0.0005426
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876057, 0.9896920, 0.9875951, 0.9896328, -0.0011940, 0.0013509
1: -0.0043523, -0.0038324, -0.0043549, -0.0038472, -0.0002975, 0.0003366
2: 0.0102559, 0.0130109, 0.0103340, 0.0130247, -0.0017838, 0.0015767
3: -0.0071951, -0.0059411, -0.0072014, -0.0059767, -0.0007177, 0.0008119
4: 0.0025129, 0.0030461, 0.0025280, 0.0030488, -0.0003453, 0.0003052
5: 0.0118586, 0.0153238, 0.0119569, 0.0153410, -0.0022436, 0.0019831
6: -0.0023485, -0.0014690, -0.0023529, -0.0014940, -0.0005033, 0.0005694
7: -0.0092139, -0.0069384, -0.0092253, -0.0070029, -0.0013023, 0.0014733
8: -0.0044097, -0.0032130, -0.0044156, -0.0032469, -0.0006849, 0.0007748
9: 0.0018618, 0.0032494, 0.0019011, 0.0032563, -0.0008984, 0.0007941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005429
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005081, upper bound: 0.0005429
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.9876592, 0.9896678, 0.9876136, 0.9896492, -0.0012053, 0.0012279
1: -0.0043390, -0.0038385, -0.0043503, -0.0038431, -0.0003003, 0.0003060
2: 0.0102878, 0.0129403, 0.0103123, 0.0130005, -0.0016214, 0.0015915
3: -0.0071630, -0.0059557, -0.0071904, -0.0059669, -0.0007244, 0.0007380
4: 0.0025191, 0.0030324, 0.0025238, 0.0030441, -0.0003138, 0.0003080
5: 0.0118988, 0.0152349, 0.0119297, 0.0153107, -0.0020393, 0.0020017
6: -0.0023259, -0.0014792, -0.0023452, -0.0014870, -0.0005081, 0.0005176
7: -0.0091555, -0.0069648, -0.0092053, -0.0069851, -0.0013145, 0.0013392
8: -0.0043790, -0.0032269, -0.0044051, -0.0032375, -0.0006913, 0.0007043
9: 0.0018779, 0.0032138, 0.0018902, 0.0032441, -0.0008166, 0.0008016

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004736, upper bound: 0.0005376
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005376
time: 0.63 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.9876588, 0.9896718, 0.9876071, 0.9896647, -0.0012175, 0.0012311
1: -0.0043391, -0.0038375, -0.0043519, -0.0038392, -0.0003034, 0.0003068
2: 0.0102826, 0.0129407, 0.0102920, 0.0130091, -0.0016257, 0.0016077
3: -0.0071632, -0.0059533, -0.0071943, -0.0059576, -0.0007318, 0.0007399
4: 0.0025181, 0.0030325, 0.0025199, 0.0030458, -0.0003147, 0.0003112
5: 0.0118922, 0.0152355, 0.0119041, 0.0153214, -0.0020447, 0.0020221
6: -0.0023261, -0.0014775, -0.0023479, -0.0014805, -0.0005132, 0.0005190
7: -0.0091559, -0.0069605, -0.0092124, -0.0069682, -0.0013279, 0.0013427
8: -0.0043792, -0.0032246, -0.0044088, -0.0032287, -0.0006983, 0.0007061
9: 0.0018752, 0.0032140, 0.0018800, 0.0032484, -0.0008188, 0.0008097

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004737, upper bound: 0.0005379
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004832, upper bound: 0.0005379
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.9876060, 0.9896855, 0.9875914, 0.9896497, -0.0011521, 0.0013254
1: -0.0043522, -0.0038341, -0.0043558, -0.0038430, -0.0002871, 0.0003303
2: 0.0102645, 0.0130105, 0.0103117, 0.0130296, -0.0017502, 0.0015214
3: -0.0071949, -0.0059451, -0.0072036, -0.0059666, -0.0006925, 0.0007966
4: 0.0025146, 0.0030460, 0.0025237, 0.0030497, -0.0003388, 0.0002945
5: 0.0118695, 0.0153232, 0.0119289, 0.0153473, -0.0022013, 0.0019135
6: -0.0023484, -0.0014718, -0.0023545, -0.0014868, -0.0004857, 0.0005587
7: -0.0092136, -0.0069456, -0.0092293, -0.0069846, -0.0012565, 0.0014456
8: -0.0044095, -0.0032167, -0.0044178, -0.0032373, -0.0006608, 0.0007602
9: 0.0018661, 0.0032491, 0.0018899, 0.0032588, -0.0008815, 0.0007662

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.07 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005426
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005123, upper bound: 0.0005426
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.9876057, 0.9896920, 0.9875820, 0.9896652, -0.0011628, 0.0013254
1: -0.0043523, -0.0038324, -0.0043582, -0.0038391, -0.0002897, 0.0003303
2: 0.0102559, 0.0130109, 0.0102913, 0.0130422, -0.0017502, 0.0015355
3: -0.0071951, -0.0059411, -0.0072093, -0.0059573, -0.0006989, 0.0007966
4: 0.0025129, 0.0030461, 0.0025197, 0.0030522, -0.0003388, 0.0002972
5: 0.0118586, 0.0153238, 0.0119032, 0.0153631, -0.0022013, 0.0019313
6: -0.0023485, -0.0014690, -0.0023585, -0.0014803, -0.0004902, 0.0005587
7: -0.0092139, -0.0069384, -0.0092397, -0.0069677, -0.0012682, 0.0014456
8: -0.0044097, -0.0032130, -0.0044232, -0.0032284, -0.0006670, 0.0007602
9: 0.0018618, 0.0032494, 0.0018796, 0.0032651, -0.0008815, 0.0007734

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 255
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 200
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 183
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 132

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005002, upper bound: 0.0005429
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005078, upper bound: 0.0005429
time: 0.72 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.56 seconds
IS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004733, upper bound: 0.0005061
IS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005060
IS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004758, upper bound: 0.0005062
IS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004841, upper bound: 0.0005061
IS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004728, upper bound: 0.0005270
IS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004855, upper bound: 0.0005270
IS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004734, upper bound: 0.0005271
IS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004836, upper bound: 0.0005271
IS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005061
IS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005060
IS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005061
IS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004944, upper bound: 0.0005061
IS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005269
IS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004954, upper bound: 0.0005269
IS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005270
IS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004935, upper bound: 0.0005270
IS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005095
IS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005128, upper bound: 0.0005095
IS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005096
IS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005096
IS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005006, upper bound: 0.0005318
IS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005318
IS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0005318
IS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005059, upper bound: 0.0005318
IS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005177, upper bound: 0.0005095
IS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005242, upper bound: 0.0005095
IS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005196, upper bound: 0.0005096
IS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005221, upper bound: 0.0005096
IS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005316
IS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005224, upper bound: 0.0005316
IS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005317
IS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005179, upper bound: 0.0005317
IS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004811, upper bound: 0.0005025
IS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004933, upper bound: 0.0005025
IS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0005026
IS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004941, upper bound: 0.0005026
IS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0005057
IS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005291, upper bound: 0.0005057
IS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005059
IS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005318, upper bound: 0.0005059
IS_A1_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004945, upper bound: 0.0005025
IS_A1_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005035, upper bound: 0.0005025
IS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0005025
IS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005052, upper bound: 0.0005025
IS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005374, upper bound: 0.0005058
IS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0005058
IS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005440, upper bound: 0.0005059
IS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005418, upper bound: 0.0005059
IS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004738, upper bound: 0.0005276
IS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004869, upper bound: 0.0005276
IS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004746, upper bound: 0.0005279
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004845, upper bound: 0.0005279
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005323
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005125, upper bound: 0.0005323
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005324
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005081, upper bound: 0.0005324
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004868, upper bound: 0.0005276
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004966, upper bound: 0.0005276
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004886, upper bound: 0.0005278
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004947, upper bound: 0.0005278
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005119, upper bound: 0.0005322
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005249, upper bound: 0.0005322
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005120, upper bound: 0.0005323
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005202, upper bound: 0.0005323
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004731, upper bound: 0.0005184
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004856, upper bound: 0.0005184
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0005185
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004826, upper bound: 0.0005185
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004726, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004850, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004725, upper bound: 0.0005369
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004821, upper bound: 0.0005369
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004731, upper bound: 0.0005184
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004856, upper bound: 0.0005184
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0005184
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004826, upper bound: 0.0005185
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004726, upper bound: 0.0005368
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004850, upper bound: 0.0005368
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004725, upper bound: 0.0005369
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004821, upper bound: 0.0005369
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005066, upper bound: 0.0005220
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0005220
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005070, upper bound: 0.0005221
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005221
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005418
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005101, upper bound: 0.0005418
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004990, upper bound: 0.0005419
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005059, upper bound: 0.0005419
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005066, upper bound: 0.0005220
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0005220
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0005221
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005094, upper bound: 0.0005221
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005418
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005101, upper bound: 0.0005418
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004986, upper bound: 0.0005419
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005057, upper bound: 0.0005419
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004805, upper bound: 0.0005137
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004912, upper bound: 0.0005137
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004890, upper bound: 0.0005142
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004939, upper bound: 0.0005142
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0005175
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005291, upper bound: 0.0005175
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005180
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005180
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004805, upper bound: 0.0005138
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004912, upper bound: 0.0005137
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004890, upper bound: 0.0005142
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004939, upper bound: 0.0005142
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005246, upper bound: 0.0005175
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005290, upper bound: 0.0005175
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005180
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005180
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004736, upper bound: 0.0005376
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005376
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004737, upper bound: 0.0005379
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004832, upper bound: 0.0005379
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005426
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005121, upper bound: 0.0005426
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005429
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005081, upper bound: 0.0005429
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004736, upper bound: 0.0005376
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005376
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004737, upper bound: 0.0005379
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0004832, upper bound: 0.0005379
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005426
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005123, upper bound: 0.0005426
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005002, upper bound: 0.0005429
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.56
Output dim: 0, lower bound: -0.0005078, upper bound: 0.0005429

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876950, 0.9895498, 0.9876153, 0.9895374, -0.0011471, 0.0011832
1: -0.0043300, -0.0038679, -0.0043499, -0.0038709, -0.0002858, 0.0002948
2: 0.0104437, 0.0128929, 0.0104600, 0.0129981, -0.0015624, 0.0015147
3: -0.0071414, -0.0060266, -0.0071893, -0.0060340, -0.0006894, 0.0007111
4: 0.0025492, 0.0030233, 0.0025524, 0.0030436, -0.0003024, 0.0002932
5: 0.0120949, 0.0151753, 0.0121154, 0.0153076, -0.0019651, 0.0019051
6: -0.0023108, -0.0015290, -0.0023444, -0.0015342, -0.0004835, 0.0004988
7: -0.0091164, -0.0070936, -0.0092033, -0.0071070, -0.0012511, 0.0012905
8: -0.0043584, -0.0032946, -0.0044041, -0.0033016, -0.0006579, 0.0006786
9: 0.0019564, 0.0031899, 0.0019646, 0.0032429, -0.0007869, 0.0007629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004733, upper bound: 0.0005021
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004733, upper bound: 0.0005061
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877424, 0.9896185, 0.9876421, 0.9895347, -0.0011419, 0.0011964
1: -0.0043182, -0.0038507, -0.0043432, -0.0038716, -0.0002845, 0.0002981
2: 0.0103529, 0.0128303, 0.0104636, 0.0129628, -0.0015798, 0.0015078
3: -0.0071129, -0.0059853, -0.0071732, -0.0060357, -0.0006863, 0.0007191
4: 0.0025317, 0.0030112, 0.0025531, 0.0030368, -0.0003058, 0.0002918
5: 0.0119807, 0.0150966, 0.0121199, 0.0152632, -0.0019870, 0.0018965
6: -0.0022909, -0.0015000, -0.0023331, -0.0015353, -0.0004813, 0.0005043
7: -0.0090648, -0.0070186, -0.0091742, -0.0071100, -0.0012454, 0.0013048
8: -0.0043312, -0.0032552, -0.0043887, -0.0033032, -0.0006549, 0.0006862
9: 0.0019107, 0.0031584, 0.0019664, 0.0032251, -0.0007957, 0.0007594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004515, upper bound: 0.0004921
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004722, upper bound: 0.0004921
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876947, 0.9895499, 0.9876173, 0.9895375, -0.0011424, 0.0011958
1: -0.0043301, -0.0038678, -0.0043494, -0.0038709, -0.0002847, 0.0002980
2: 0.0104434, 0.0128933, 0.0104599, 0.0129955, -0.0015790, 0.0015085
3: -0.0071416, -0.0060265, -0.0071881, -0.0060340, -0.0006866, 0.0007187
4: 0.0025492, 0.0030234, 0.0025524, 0.0030431, -0.0003056, 0.0002920
5: 0.0120945, 0.0151759, 0.0121152, 0.0153044, -0.0019860, 0.0018973
6: -0.0023110, -0.0015289, -0.0023436, -0.0015341, -0.0004816, 0.0005041
7: -0.0091168, -0.0070933, -0.0092012, -0.0071069, -0.0012459, 0.0013042
8: -0.0043586, -0.0032945, -0.0044030, -0.0033016, -0.0006552, 0.0006859
9: 0.0019562, 0.0031901, 0.0019645, 0.0032416, -0.0007953, 0.0007598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.82 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004758, upper bound: 0.0005022
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004758, upper bound: 0.0005061
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877421, 0.9896185, 0.9876476, 0.9895352, -0.0011381, 0.0012080
1: -0.0043183, -0.0038508, -0.0043418, -0.0038715, -0.0002836, 0.0003010
2: 0.0103530, 0.0128308, 0.0104630, 0.0129555, -0.0015952, 0.0015028
3: -0.0071131, -0.0059853, -0.0071699, -0.0060354, -0.0006840, 0.0007261
4: 0.0025317, 0.0030113, 0.0025530, 0.0030354, -0.0003087, 0.0002909
5: 0.0119808, 0.0150972, 0.0121191, 0.0152541, -0.0020064, 0.0018902
6: -0.0022910, -0.0015000, -0.0023308, -0.0015351, -0.0004797, 0.0005092
7: -0.0090651, -0.0070186, -0.0091681, -0.0071095, -0.0012413, 0.0013175
8: -0.0043314, -0.0032552, -0.0043856, -0.0033029, -0.0006528, 0.0006929
9: 0.0019107, 0.0031586, 0.0019661, 0.0032214, -0.0008034, 0.0007569

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004522, upper bound: 0.0004921
time: 0.60 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0004921
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9877009, 0.9896263, 0.9876153, 0.9895374, -0.0011974, 0.0013333
1: -0.0043286, -0.0038488, -0.0043499, -0.0038709, -0.0002984, 0.0003322
2: 0.0103426, 0.0128851, 0.0104600, 0.0129981, -0.0017606, 0.0015812
3: -0.0071379, -0.0059806, -0.0071893, -0.0060340, -0.0007197, 0.0008014
4: 0.0025297, 0.0030218, 0.0025524, 0.0030436, -0.0003408, 0.0003060
5: 0.0119678, 0.0151655, 0.0121154, 0.0153076, -0.0022144, 0.0019887
6: -0.0023083, -0.0014967, -0.0023444, -0.0015342, -0.0005048, 0.0005620
7: -0.0091100, -0.0070101, -0.0092033, -0.0071070, -0.0013059, 0.0014542
8: -0.0043550, -0.0032507, -0.0044041, -0.0033016, -0.0006868, 0.0007647
9: 0.0019055, 0.0031860, 0.0019646, 0.0032429, -0.0008867, 0.0007964

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.84 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004728, upper bound: 0.0005268
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004728, upper bound: 0.0005270
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877540, 0.9896901, 0.9876421, 0.9895347, -0.0011907, 0.0013339
1: -0.0043153, -0.0038329, -0.0043432, -0.0038716, -0.0002967, 0.0003324
2: 0.0102585, 0.0128150, 0.0104636, 0.0129628, -0.0017614, 0.0015723
3: -0.0071060, -0.0059424, -0.0071732, -0.0060357, -0.0007156, 0.0008017
4: 0.0025134, 0.0030082, 0.0025531, 0.0030368, -0.0003409, 0.0003043
5: 0.0118620, 0.0150774, 0.0121199, 0.0152632, -0.0022154, 0.0019775
6: -0.0022860, -0.0014699, -0.0023331, -0.0015353, -0.0005019, 0.0005623
7: -0.0090521, -0.0069406, -0.0091742, -0.0071100, -0.0012986, 0.0014548
8: -0.0043246, -0.0032141, -0.0043887, -0.0033032, -0.0006829, 0.0007651
9: 0.0018631, 0.0031507, 0.0019664, 0.0032251, -0.0008871, 0.0007919

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004515, upper bound: 0.0005146
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004716, upper bound: 0.0005146
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9877006, 0.9896313, 0.9876173, 0.9895375, -0.0011886, 0.0013385
1: -0.0043287, -0.0038476, -0.0043494, -0.0038709, -0.0002962, 0.0003335
2: 0.0103361, 0.0128856, 0.0104599, 0.0129955, -0.0017675, 0.0015695
3: -0.0071381, -0.0059777, -0.0071881, -0.0060340, -0.0007144, 0.0008045
4: 0.0025284, 0.0030219, 0.0025524, 0.0030431, -0.0003421, 0.0003038
5: 0.0119596, 0.0151661, 0.0121152, 0.0153044, -0.0022231, 0.0019741
6: -0.0023085, -0.0014946, -0.0023436, -0.0015341, -0.0005010, 0.0005642
7: -0.0091104, -0.0070047, -0.0092012, -0.0071069, -0.0012963, 0.0014599
8: -0.0043552, -0.0032478, -0.0044030, -0.0033016, -0.0006817, 0.0007677
9: 0.0019022, 0.0031862, 0.0019645, 0.0032416, -0.0008902, 0.0007905

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004734, upper bound: 0.0005270
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004734, upper bound: 0.0005271
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877536, 0.9896941, 0.9876476, 0.9895352, -0.0011829, 0.0013342
1: -0.0043154, -0.0038319, -0.0043418, -0.0038715, -0.0002947, 0.0003324
2: 0.0102531, 0.0128155, 0.0104630, 0.0129555, -0.0017618, 0.0015620
3: -0.0071062, -0.0059399, -0.0071699, -0.0060354, -0.0007110, 0.0008019
4: 0.0025124, 0.0030083, 0.0025530, 0.0030354, -0.0003410, 0.0003023
5: 0.0118552, 0.0150779, 0.0121191, 0.0152541, -0.0022159, 0.0019646
6: -0.0022861, -0.0014681, -0.0023308, -0.0015351, -0.0004986, 0.0005624
7: -0.0090525, -0.0069361, -0.0091681, -0.0071095, -0.0012901, 0.0014551
8: -0.0043248, -0.0032118, -0.0043856, -0.0033029, -0.0006785, 0.0007652
9: 0.0018604, 0.0031509, 0.0019661, 0.0032214, -0.0008873, 0.0007867

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004522, upper bound: 0.0005146
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004690, upper bound: 0.0005146
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876950, 0.9895498, 0.9876012, 0.9895699, -0.0011845, 0.0012172
1: -0.0043300, -0.0038679, -0.0043534, -0.0038628, -0.0002952, 0.0003033
2: 0.0104437, 0.0128929, 0.0104170, 0.0130169, -0.0016073, 0.0015642
3: -0.0071414, -0.0060266, -0.0071978, -0.0060145, -0.0007119, 0.0007316
4: 0.0025492, 0.0030233, 0.0025441, 0.0030473, -0.0003111, 0.0003027
5: 0.0120949, 0.0151753, 0.0120613, 0.0153312, -0.0020216, 0.0019673
6: -0.0023108, -0.0015290, -0.0023504, -0.0015205, -0.0004993, 0.0005131
7: -0.0091164, -0.0070936, -0.0092188, -0.0070715, -0.0012919, 0.0013276
8: -0.0043584, -0.0032946, -0.0044122, -0.0032830, -0.0006794, 0.0006982
9: 0.0019564, 0.0031899, 0.0019429, 0.0032524, -0.0008095, 0.0007878

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005021
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005060
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877424, 0.9896185, 0.9876275, 0.9895674, -0.0011796, 0.0012302
1: -0.0043182, -0.0038507, -0.0043469, -0.0038635, -0.0002939, 0.0003065
2: 0.0103529, 0.0128303, 0.0104204, 0.0129821, -0.0016244, 0.0015577
3: -0.0071129, -0.0059853, -0.0071820, -0.0060161, -0.0007090, 0.0007394
4: 0.0025317, 0.0030112, 0.0025447, 0.0030405, -0.0003144, 0.0003015
5: 0.0119807, 0.0150966, 0.0120656, 0.0152875, -0.0020431, 0.0019592
6: -0.0022909, -0.0015000, -0.0023393, -0.0015216, -0.0004973, 0.0005186
7: -0.0090648, -0.0070186, -0.0091901, -0.0070743, -0.0012866, 0.0013417
8: -0.0043312, -0.0032552, -0.0043971, -0.0032845, -0.0006766, 0.0007056
9: 0.0019107, 0.0031584, 0.0019447, 0.0032348, -0.0008182, 0.0007845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004723, upper bound: 0.0004921
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004810, upper bound: 0.0004921
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876947, 0.9895499, 0.9876012, 0.9895779, -0.0011816, 0.0012322
1: -0.0043301, -0.0038678, -0.0043534, -0.0038609, -0.0002944, 0.0003070
2: 0.0104434, 0.0128933, 0.0104065, 0.0130168, -0.0016271, 0.0015603
3: -0.0071416, -0.0060265, -0.0071978, -0.0060097, -0.0007102, 0.0007406
4: 0.0025492, 0.0030234, 0.0025421, 0.0030473, -0.0003149, 0.0003020
5: 0.0120945, 0.0151759, 0.0120481, 0.0153311, -0.0020464, 0.0019624
6: -0.0023110, -0.0015289, -0.0023504, -0.0015171, -0.0004981, 0.0005194
7: -0.0091168, -0.0070933, -0.0092187, -0.0070629, -0.0012887, 0.0013439
8: -0.0043586, -0.0032945, -0.0044122, -0.0032784, -0.0006777, 0.0007067
9: 0.0019562, 0.0031901, 0.0019377, 0.0032523, -0.0008195, 0.0007858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.96 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005022
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005061
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877421, 0.9896185, 0.9876313, 0.9895756, -0.0011776, 0.0012416
1: -0.0043183, -0.0038508, -0.0043459, -0.0038614, -0.0002934, 0.0003094
2: 0.0103530, 0.0128308, 0.0104096, 0.0129771, -0.0016395, 0.0015550
3: -0.0071131, -0.0059853, -0.0071797, -0.0060111, -0.0007077, 0.0007462
4: 0.0025317, 0.0030113, 0.0025426, 0.0030396, -0.0003173, 0.0003010
5: 0.0119808, 0.0150972, 0.0120519, 0.0152812, -0.0020621, 0.0019557
6: -0.0022910, -0.0015000, -0.0023377, -0.0015181, -0.0004964, 0.0005234
7: -0.0090651, -0.0070186, -0.0091860, -0.0070653, -0.0012843, 0.0013541
8: -0.0043314, -0.0032552, -0.0043950, -0.0032797, -0.0006754, 0.0007121
9: 0.0019107, 0.0031586, 0.0019392, 0.0032323, -0.0008257, 0.0007832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004724, upper bound: 0.0004921
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004792, upper bound: 0.0004921
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9877009, 0.9896263, 0.9876012, 0.9895699, -0.0012348, 0.0013672
1: -0.0043286, -0.0038488, -0.0043534, -0.0038628, -0.0003077, 0.0003407
2: 0.0103426, 0.0128851, 0.0104170, 0.0130169, -0.0018054, 0.0016306
3: -0.0071379, -0.0059806, -0.0071978, -0.0060145, -0.0007422, 0.0008217
4: 0.0025297, 0.0030218, 0.0025441, 0.0030473, -0.0003494, 0.0003156
5: 0.0119678, 0.0151655, 0.0120613, 0.0153312, -0.0022707, 0.0020509
6: -0.0023083, -0.0014967, -0.0023504, -0.0015205, -0.0005205, 0.0005763
7: -0.0091100, -0.0070101, -0.0092188, -0.0070715, -0.0013468, 0.0014911
8: -0.0043550, -0.0032507, -0.0044122, -0.0032830, -0.0007083, 0.0007842
9: 0.0019055, 0.0031860, 0.0019429, 0.0032524, -0.0009093, 0.0008213

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005268
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005269
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877540, 0.9896901, 0.9876275, 0.9895674, -0.0012284, 0.0013647
1: -0.0043153, -0.0038329, -0.0043469, -0.0038635, -0.0003061, 0.0003401
2: 0.0102585, 0.0128150, 0.0104204, 0.0129821, -0.0018021, 0.0016221
3: -0.0071060, -0.0059424, -0.0071820, -0.0060161, -0.0007383, 0.0008202
4: 0.0025134, 0.0030082, 0.0025447, 0.0030405, -0.0003488, 0.0003140
5: 0.0118620, 0.0150774, 0.0120656, 0.0152875, -0.0022666, 0.0020402
6: -0.0022860, -0.0014699, -0.0023393, -0.0015216, -0.0005178, 0.0005753
7: -0.0090521, -0.0069406, -0.0091901, -0.0070743, -0.0013398, 0.0014884
8: -0.0043246, -0.0032141, -0.0043971, -0.0032845, -0.0007046, 0.0007827
9: 0.0018631, 0.0031507, 0.0019447, 0.0032348, -0.0009076, 0.0008170

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004723, upper bound: 0.0005146
time: 0.68 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004809, upper bound: 0.0005146
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9877006, 0.9896313, 0.9876012, 0.9895779, -0.0012278, 0.0013749
1: -0.0043287, -0.0038476, -0.0043534, -0.0038609, -0.0003059, 0.0003426
2: 0.0103361, 0.0128856, 0.0104065, 0.0130168, -0.0018155, 0.0016213
3: -0.0071381, -0.0059777, -0.0071978, -0.0060097, -0.0007379, 0.0008264
4: 0.0025284, 0.0030219, 0.0025421, 0.0030473, -0.0003514, 0.0003138
5: 0.0119596, 0.0151661, 0.0120481, 0.0153311, -0.0022835, 0.0020392
6: -0.0023085, -0.0014946, -0.0023504, -0.0015171, -0.0005176, 0.0005796
7: -0.0091104, -0.0070047, -0.0092187, -0.0070629, -0.0013391, 0.0014995
8: -0.0043552, -0.0032478, -0.0044122, -0.0032784, -0.0007042, 0.0007886
9: 0.0019022, 0.0031862, 0.0019377, 0.0032523, -0.0009144, 0.0008166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005269
time: 0.63 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005270
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877536, 0.9896941, 0.9876313, 0.9895756, -0.0012224, 0.0013678
1: -0.0043154, -0.0038319, -0.0043459, -0.0038614, -0.0003046, 0.0003408
2: 0.0102531, 0.0128155, 0.0104096, 0.0129771, -0.0018061, 0.0016141
3: -0.0071062, -0.0059399, -0.0071797, -0.0060111, -0.0007347, 0.0008221
4: 0.0025124, 0.0030083, 0.0025426, 0.0030396, -0.0003496, 0.0003124
5: 0.0118552, 0.0150779, 0.0120519, 0.0152812, -0.0022716, 0.0020301
6: -0.0022861, -0.0014681, -0.0023377, -0.0015181, -0.0005153, 0.0005766
7: -0.0090525, -0.0069361, -0.0091860, -0.0070653, -0.0013332, 0.0014917
8: -0.0043248, -0.0032118, -0.0043950, -0.0032797, -0.0007011, 0.0007845
9: 0.0018604, 0.0031509, 0.0019392, 0.0032323, -0.0009097, 0.0008130

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004724, upper bound: 0.0005146
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004784, upper bound: 0.0005146
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876487, 0.9895658, 0.9875935, 0.9895380, -0.0010943, 0.0012895
1: -0.0043416, -0.0038639, -0.0043553, -0.0038708, -0.0002727, 0.0003213
2: 0.0104227, 0.0129540, 0.0104593, 0.0130270, -0.0017028, 0.0014450
3: -0.0071692, -0.0060171, -0.0072024, -0.0060337, -0.0006577, 0.0007751
4: 0.0025452, 0.0030351, 0.0025523, 0.0030492, -0.0003296, 0.0002797
5: 0.0120684, 0.0152522, 0.0121145, 0.0153440, -0.0021417, 0.0018175
6: -0.0023303, -0.0015223, -0.0023536, -0.0015340, -0.0004613, 0.0005436
7: -0.0091669, -0.0070762, -0.0092272, -0.0071064, -0.0011935, 0.0014064
8: -0.0043849, -0.0032854, -0.0044166, -0.0033014, -0.0006277, 0.0007396
9: 0.0019458, 0.0032207, 0.0019642, 0.0032575, -0.0008576, 0.0007278

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.02 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005062
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005095
time: 0.67 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876953, 0.9896342, 0.9876197, 0.9895353, -0.0010854, 0.0013066
1: -0.0043299, -0.0038468, -0.0043488, -0.0038715, -0.0002704, 0.0003256
2: 0.0103323, 0.0128924, 0.0104628, 0.0129923, -0.0017253, 0.0014332
3: -0.0071412, -0.0059759, -0.0071867, -0.0060353, -0.0006523, 0.0007853
4: 0.0025277, 0.0030232, 0.0025529, 0.0030425, -0.0003339, 0.0002774
5: 0.0119547, 0.0151747, 0.0121188, 0.0153004, -0.0021700, 0.0018026
6: -0.0023107, -0.0014934, -0.0023426, -0.0015351, -0.0004575, 0.0005508
7: -0.0091161, -0.0070015, -0.0091986, -0.0071093, -0.0011838, 0.0014250
8: -0.0043582, -0.0032462, -0.0044016, -0.0033029, -0.0006225, 0.0007494
9: 0.0019002, 0.0031897, 0.0019660, 0.0032400, -0.0008690, 0.0007219

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.10 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004788, upper bound: 0.0004946
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004986, upper bound: 0.0004946
time: 0.66 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876481, 0.9895653, 0.9875969, 0.9895382, -0.0010898, 0.0012894
1: -0.0043417, -0.0038640, -0.0043545, -0.0038708, -0.0002715, 0.0003213
2: 0.0104232, 0.0129549, 0.0104590, 0.0130224, -0.0017026, 0.0014390
3: -0.0071696, -0.0060173, -0.0072004, -0.0060336, -0.0006550, 0.0007750
4: 0.0025453, 0.0030353, 0.0025522, 0.0030484, -0.0003295, 0.0002785
5: 0.0120691, 0.0152533, 0.0121142, 0.0153383, -0.0021414, 0.0018099
6: -0.0023306, -0.0015224, -0.0023522, -0.0015339, -0.0004594, 0.0005435
7: -0.0091676, -0.0070766, -0.0092234, -0.0071062, -0.0011886, 0.0014063
8: -0.0043853, -0.0032857, -0.0044147, -0.0033012, -0.0006251, 0.0007395
9: 0.0019461, 0.0032211, 0.0019641, 0.0032552, -0.0008575, 0.0007248

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.01 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005063, upper bound: 0.0005063
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005063, upper bound: 0.0005096
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876949, 0.9896331, 0.9876263, 0.9895359, -0.0010816, 0.0013023
1: -0.0043301, -0.0038471, -0.0043471, -0.0038713, -0.0002695, 0.0003245
2: 0.0103338, 0.0128931, 0.0104620, 0.0129836, -0.0017197, 0.0014282
3: -0.0071415, -0.0059766, -0.0071827, -0.0060350, -0.0006501, 0.0007827
4: 0.0025280, 0.0030233, 0.0025528, 0.0030408, -0.0003328, 0.0002764
5: 0.0119566, 0.0151755, 0.0121179, 0.0152894, -0.0021629, 0.0017963
6: -0.0023109, -0.0014939, -0.0023398, -0.0015348, -0.0004559, 0.0005490
7: -0.0091166, -0.0070027, -0.0091913, -0.0071087, -0.0011796, 0.0014204
8: -0.0043585, -0.0032468, -0.0043978, -0.0033025, -0.0006203, 0.0007470
9: 0.0019010, 0.0031900, 0.0019656, 0.0032356, -0.0008661, 0.0007193

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004777, upper bound: 0.0004946
time: 0.64 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004946, upper bound: 0.0004946
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876453, 0.9896473, 0.9875935, 0.9895380, -0.0011427, 0.0014264
1: -0.0043424, -0.0038436, -0.0043553, -0.0038708, -0.0002847, 0.0003554
2: 0.0103149, 0.0129585, 0.0104593, 0.0130270, -0.0018836, 0.0015089
3: -0.0071713, -0.0059680, -0.0072024, -0.0060337, -0.0006868, 0.0008573
4: 0.0025243, 0.0030360, 0.0025523, 0.0030492, -0.0003646, 0.0002921
5: 0.0119329, 0.0152579, 0.0121145, 0.0153440, -0.0023690, 0.0018979
6: -0.0023318, -0.0014879, -0.0023536, -0.0015340, -0.0004817, 0.0006013
7: -0.0091706, -0.0069872, -0.0092272, -0.0071064, -0.0012463, 0.0015557
8: -0.0043869, -0.0032386, -0.0044166, -0.0033014, -0.0006554, 0.0008181
9: 0.0018915, 0.0032230, 0.0019642, 0.0032575, -0.0009487, 0.0007600

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.90 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005006, upper bound: 0.0005316
time: 0.71 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005006, upper bound: 0.0005318
time: 0.70 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877048, 0.9897090, 0.9876197, 0.9895353, -0.0011324, 0.0014294
1: -0.0043276, -0.0038282, -0.0043488, -0.0038715, -0.0002822, 0.0003562
2: 0.0102335, 0.0128800, 0.0104628, 0.0129923, -0.0018875, 0.0014953
3: -0.0071356, -0.0059310, -0.0071867, -0.0060353, -0.0006806, 0.0008591
4: 0.0025086, 0.0030208, 0.0025529, 0.0030425, -0.0003653, 0.0002894
5: 0.0118305, 0.0151592, 0.0121188, 0.0153004, -0.0023740, 0.0018806
6: -0.0023067, -0.0014619, -0.0023426, -0.0015351, -0.0004773, 0.0006025
7: -0.0091058, -0.0069199, -0.0091986, -0.0071093, -0.0012350, 0.0015590
8: -0.0043528, -0.0032033, -0.0044016, -0.0033029, -0.0006495, 0.0008199
9: 0.0018505, 0.0031834, 0.0019660, 0.0032400, -0.0009507, 0.0007531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004788, upper bound: 0.0005189
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004957, upper bound: 0.0005189
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876446, 0.9896558, 0.9875969, 0.9895382, -0.0011333, 0.0014336
1: -0.0043426, -0.0038415, -0.0043545, -0.0038708, -0.0002824, 0.0003572
2: 0.0103038, 0.0129595, 0.0104590, 0.0130224, -0.0018931, 0.0014965
3: -0.0071717, -0.0059629, -0.0072004, -0.0060336, -0.0006811, 0.0008616
4: 0.0025222, 0.0030362, 0.0025522, 0.0030484, -0.0003664, 0.0002896
5: 0.0119189, 0.0152591, 0.0121142, 0.0153383, -0.0023810, 0.0018822
6: -0.0023321, -0.0014843, -0.0023522, -0.0015339, -0.0004777, 0.0006043
7: -0.0091715, -0.0069780, -0.0092234, -0.0071062, -0.0012360, 0.0015636
8: -0.0043873, -0.0032338, -0.0044147, -0.0033012, -0.0006500, 0.0008223
9: 0.0018859, 0.0032235, 0.0019641, 0.0032552, -0.0009534, 0.0007537

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0005316
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0005318
time: 0.62 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877040, 0.9897111, 0.9876263, 0.9895359, -0.0011243, 0.0014295
1: -0.0043278, -0.0038277, -0.0043471, -0.0038713, -0.0002802, 0.0003562
2: 0.0102307, 0.0128809, 0.0104620, 0.0129836, -0.0018877, 0.0014847
3: -0.0071360, -0.0059297, -0.0071827, -0.0060350, -0.0006758, 0.0008592
4: 0.0025080, 0.0030210, 0.0025528, 0.0030408, -0.0003654, 0.0002874
5: 0.0118270, 0.0151603, 0.0121179, 0.0152894, -0.0023742, 0.0018673
6: -0.0023070, -0.0014610, -0.0023398, -0.0015348, -0.0004740, 0.0006026
7: -0.0091065, -0.0069176, -0.0091913, -0.0071087, -0.0012263, 0.0015591
8: -0.0043532, -0.0032021, -0.0043978, -0.0033025, -0.0006449, 0.0008199
9: 0.0018491, 0.0031839, 0.0019656, 0.0032356, -0.0009507, 0.0007478

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004777, upper bound: 0.0005189
time: 0.66 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004906, upper bound: 0.0005189
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876487, 0.9895658, 0.9875796, 0.9895704, -0.0011588, 0.0013075
1: -0.0043416, -0.0038639, -0.0043588, -0.0038627, -0.0002887, 0.0003258
2: 0.0104227, 0.0129540, 0.0104164, 0.0130453, -0.0017265, 0.0015302
3: -0.0071692, -0.0060171, -0.0072108, -0.0060142, -0.0006965, 0.0007858
4: 0.0025452, 0.0030351, 0.0025440, 0.0030528, -0.0003342, 0.0002962
5: 0.0120684, 0.0152522, 0.0120605, 0.0153670, -0.0021715, 0.0019246
6: -0.0023303, -0.0015223, -0.0023595, -0.0015203, -0.0004885, 0.0005512
7: -0.0091669, -0.0070762, -0.0092423, -0.0070710, -0.0012639, 0.0014260
8: -0.0043849, -0.0032854, -0.0044246, -0.0032827, -0.0006647, 0.0007499
9: 0.0019458, 0.0032207, 0.0019426, 0.0032667, -0.0008696, 0.0007707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005177, upper bound: 0.0005062
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005177, upper bound: 0.0005095
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876953, 0.9896342, 0.9876055, 0.9895681, -0.0011502, 0.0013229
1: -0.0043299, -0.0038468, -0.0043523, -0.0038633, -0.0002866, 0.0003296
2: 0.0103323, 0.0128924, 0.0104196, 0.0130112, -0.0017468, 0.0015188
3: -0.0071412, -0.0059759, -0.0071952, -0.0060157, -0.0006913, 0.0007951
4: 0.0025277, 0.0030232, 0.0025446, 0.0030462, -0.0003381, 0.0002940
5: 0.0119547, 0.0151747, 0.0120646, 0.0153241, -0.0021970, 0.0019103
6: -0.0023107, -0.0014934, -0.0023486, -0.0015213, -0.0004848, 0.0005576
7: -0.0091161, -0.0070015, -0.0092141, -0.0070737, -0.0012544, 0.0014428
8: -0.0043582, -0.0032462, -0.0044098, -0.0032841, -0.0006597, 0.0007587
9: 0.0019002, 0.0031897, 0.0019443, 0.0032495, -0.0008798, 0.0007650

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0004946
time: 0.67 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005088, upper bound: 0.0004946
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876481, 0.9895653, 0.9875761, 0.9895785, -0.0011552, 0.0013129
1: -0.0043417, -0.0038640, -0.0043597, -0.0038607, -0.0002879, 0.0003271
2: 0.0104232, 0.0129549, 0.0104057, 0.0130500, -0.0017337, 0.0015255
3: -0.0071696, -0.0060173, -0.0072129, -0.0060094, -0.0006943, 0.0007891
4: 0.0025453, 0.0030353, 0.0025419, 0.0030537, -0.0003355, 0.0002953
5: 0.0120691, 0.0152533, 0.0120471, 0.0153729, -0.0021805, 0.0019187
6: -0.0023306, -0.0015224, -0.0023610, -0.0015169, -0.0004870, 0.0005534
7: -0.0091676, -0.0070766, -0.0092462, -0.0070622, -0.0012600, 0.0014319
8: -0.0043853, -0.0032857, -0.0044266, -0.0032781, -0.0006626, 0.0007530
9: 0.0019461, 0.0032211, 0.0019373, 0.0032690, -0.0008732, 0.0007683

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005195, upper bound: 0.0005063
time: 0.61 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005195, upper bound: 0.0005096
time: 0.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876949, 0.9896331, 0.9876050, 0.9895763, -0.0011471, 0.0013225
1: -0.0043301, -0.0038471, -0.0043524, -0.0038613, -0.0002858, 0.0003295
2: 0.0103338, 0.0128931, 0.0104086, 0.0130117, -0.0017464, 0.0015148
3: -0.0071415, -0.0059766, -0.0071955, -0.0060107, -0.0006895, 0.0007949
4: 0.0025280, 0.0030233, 0.0025425, 0.0030463, -0.0003380, 0.0002932
5: 0.0119566, 0.0151755, 0.0120508, 0.0153247, -0.0021965, 0.0019052
6: -0.0023109, -0.0014939, -0.0023488, -0.0015178, -0.0004836, 0.0005575
7: -0.0091166, -0.0070027, -0.0092146, -0.0070646, -0.0012511, 0.0014424
8: -0.0043585, -0.0032468, -0.0044100, -0.0032793, -0.0006580, 0.0007586
9: 0.0019010, 0.0031900, 0.0019387, 0.0032497, -0.0008796, 0.0007629

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0004946
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005062, upper bound: 0.0004946
time: 0.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876453, 0.9896473, 0.9875796, 0.9895704, -0.0012072, 0.0014444
1: -0.0043424, -0.0038436, -0.0043588, -0.0038627, -0.0003008, 0.0003599
2: 0.0103149, 0.0129585, 0.0104164, 0.0130453, -0.0019073, 0.0015941
3: -0.0071713, -0.0059680, -0.0072108, -0.0060142, -0.0007256, 0.0008681
4: 0.0025243, 0.0030360, 0.0025440, 0.0030528, -0.0003692, 0.0003085
5: 0.0119329, 0.0152579, 0.0120605, 0.0153670, -0.0023989, 0.0020050
6: -0.0023318, -0.0014879, -0.0023595, -0.0015203, -0.0005089, 0.0006089
7: -0.0091706, -0.0069872, -0.0092423, -0.0070710, -0.0013167, 0.0015753
8: -0.0043869, -0.0032386, -0.0044246, -0.0032827, -0.0006924, 0.0008284
9: 0.0018915, 0.0032230, 0.0019426, 0.0032667, -0.0009606, 0.0008029

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005315
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005317
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877048, 0.9897090, 0.9876055, 0.9895681, -0.0011972, 0.0014457
1: -0.0043276, -0.0038282, -0.0043523, -0.0038633, -0.0002983, 0.0003602
2: 0.0102335, 0.0128800, 0.0104196, 0.0130112, -0.0019090, 0.0015808
3: -0.0071356, -0.0059310, -0.0071952, -0.0060157, -0.0007195, 0.0008689
4: 0.0025086, 0.0030208, 0.0025446, 0.0030462, -0.0003695, 0.0003060
5: 0.0118305, 0.0151592, 0.0120646, 0.0153241, -0.0024010, 0.0019883
6: -0.0023067, -0.0014619, -0.0023486, -0.0015213, -0.0005046, 0.0006094
7: -0.0091058, -0.0069199, -0.0092141, -0.0070737, -0.0013057, 0.0015767
8: -0.0043528, -0.0032033, -0.0044098, -0.0032841, -0.0006866, 0.0008292
9: 0.0018505, 0.0031834, 0.0019443, 0.0032495, -0.0009615, 0.0007962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004986, upper bound: 0.0005188
time: 0.70 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0005188
time: 0.69 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876446, 0.9896558, 0.9875761, 0.9895785, -0.0011988, 0.0014571
1: -0.0043426, -0.0038415, -0.0043597, -0.0038607, -0.0002987, 0.0003631
2: 0.0103038, 0.0129595, 0.0104057, 0.0130500, -0.0019241, 0.0015829
3: -0.0071717, -0.0059629, -0.0072129, -0.0060094, -0.0007205, 0.0008758
4: 0.0025222, 0.0030362, 0.0025419, 0.0030537, -0.0003724, 0.0003064
5: 0.0119189, 0.0152591, 0.0120471, 0.0153729, -0.0024200, 0.0019909
6: -0.0023321, -0.0014843, -0.0023610, -0.0015169, -0.0005053, 0.0006142
7: -0.0091715, -0.0069780, -0.0092462, -0.0070622, -0.0013074, 0.0015892
8: -0.0043873, -0.0032338, -0.0044266, -0.0032781, -0.0006876, 0.0008357
9: 0.0018859, 0.0032235, 0.0019373, 0.0032690, -0.0009691, 0.0007973

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.05 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005315
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005317
time: 0.65 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877040, 0.9897111, 0.9876050, 0.9895763, -0.0011899, 0.0014498
1: -0.0043278, -0.0038277, -0.0043524, -0.0038613, -0.0002965, 0.0003612
2: 0.0102307, 0.0128809, 0.0104086, 0.0130117, -0.0019144, 0.0015713
3: -0.0071360, -0.0059297, -0.0071955, -0.0060107, -0.0007152, 0.0008713
4: 0.0025080, 0.0030210, 0.0025425, 0.0030463, -0.0003705, 0.0003041
5: 0.0118270, 0.0151603, 0.0120508, 0.0153247, -0.0024078, 0.0019763
6: -0.0023070, -0.0014610, -0.0023488, -0.0015178, -0.0005016, 0.0006111
7: -0.0091065, -0.0069176, -0.0092146, -0.0070646, -0.0012978, 0.0015812
8: -0.0043532, -0.0032021, -0.0044100, -0.0032793, -0.0006825, 0.0008315
9: 0.0018491, 0.0031839, 0.0019387, 0.0032497, -0.0009642, 0.0007914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005188
time: 0.65 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005017, upper bound: 0.0005188
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876950, 0.9895498, 0.9876310, 0.9896138, -0.0012779, 0.0012226
1: -0.0043300, -0.0038679, -0.0043460, -0.0038519, -0.0003184, 0.0003046
2: 0.0104437, 0.0128929, 0.0103592, 0.0129774, -0.0016145, 0.0016874
3: -0.0071414, -0.0060266, -0.0071799, -0.0059882, -0.0007680, 0.0007348
4: 0.0025492, 0.0030233, 0.0025329, 0.0030396, -0.0003125, 0.0003266
5: 0.0120949, 0.0151753, 0.0119885, 0.0152816, -0.0020306, 0.0021223
6: -0.0023108, -0.0015290, -0.0023378, -0.0015020, -0.0005387, 0.0005154
7: -0.0091164, -0.0070936, -0.0091862, -0.0070237, -0.0013937, 0.0013334
8: -0.0043584, -0.0032946, -0.0043951, -0.0032579, -0.0007329, 0.0007012
9: 0.0019564, 0.0031899, 0.0019138, 0.0032325, -0.0008131, 0.0008499

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.87 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004811, upper bound: 0.0004943
time: 0.70 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004811, upper bound: 0.0005025
time: 0.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877424, 0.9896185, 0.9876598, 0.9896110, -0.0012731, 0.0012311
1: -0.0043182, -0.0038507, -0.0043388, -0.0038526, -0.0003172, 0.0003068
2: 0.0103529, 0.0128303, 0.0103628, 0.0129395, -0.0016257, 0.0016812
3: -0.0071129, -0.0059853, -0.0071626, -0.0059898, -0.0007652, 0.0007400
4: 0.0025317, 0.0030112, 0.0025336, 0.0030323, -0.0003147, 0.0003254
5: 0.0119807, 0.0150966, 0.0119931, 0.0152339, -0.0020447, 0.0021145
6: -0.0022909, -0.0015000, -0.0023257, -0.0015032, -0.0005367, 0.0005190
7: -0.0090648, -0.0070186, -0.0091549, -0.0070267, -0.0013885, 0.0013427
8: -0.0043312, -0.0032552, -0.0043786, -0.0032594, -0.0007302, 0.0007061
9: 0.0019107, 0.0031584, 0.0019156, 0.0032134, -0.0008188, 0.0008467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.04 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004607, upper bound: 0.0004879
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004795, upper bound: 0.0004879
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876947, 0.9895499, 0.9876196, 0.9896323, -0.0012927, 0.0012482
1: -0.0043301, -0.0038678, -0.0043488, -0.0038473, -0.0003221, 0.0003110
2: 0.0104434, 0.0128933, 0.0103348, 0.0129925, -0.0016483, 0.0017070
3: -0.0071416, -0.0060265, -0.0071868, -0.0059771, -0.0007769, 0.0007502
4: 0.0025492, 0.0030234, 0.0025282, 0.0030426, -0.0003190, 0.0003304
5: 0.0120945, 0.0151759, 0.0119579, 0.0153006, -0.0020731, 0.0021469
6: -0.0023110, -0.0015289, -0.0023426, -0.0014942, -0.0005449, 0.0005262
7: -0.0091168, -0.0070933, -0.0091987, -0.0070036, -0.0014099, 0.0013614
8: -0.0043586, -0.0032945, -0.0044017, -0.0032473, -0.0007414, 0.0007159
9: 0.0019562, 0.0031901, 0.0019015, 0.0032401, -0.0008302, 0.0008597

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.16 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0004945
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0005026
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877421, 0.9896185, 0.9876511, 0.9896302, -0.0012746, 0.0012575
1: -0.0043183, -0.0038508, -0.0043410, -0.0038478, -0.0003176, 0.0003133
2: 0.0103530, 0.0128308, 0.0103375, 0.0129508, -0.0016605, 0.0016830
3: -0.0071131, -0.0059853, -0.0071678, -0.0059783, -0.0007660, 0.0007558
4: 0.0025317, 0.0030113, 0.0025287, 0.0030345, -0.0003214, 0.0003257
5: 0.0119808, 0.0150972, 0.0119614, 0.0152482, -0.0020884, 0.0021168
6: -0.0022910, -0.0015000, -0.0023293, -0.0014951, -0.0005373, 0.0005301
7: -0.0090651, -0.0070186, -0.0091643, -0.0070059, -0.0013901, 0.0013714
8: -0.0043314, -0.0032552, -0.0043836, -0.0032485, -0.0007310, 0.0007212
9: 0.0019107, 0.0031586, 0.0019029, 0.0032191, -0.0008363, 0.0008477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004633, upper bound: 0.0004879
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004808, upper bound: 0.0004879
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876487, 0.9895658, 0.9876074, 0.9896143, -0.0012235, 0.0013285
1: -0.0043416, -0.0038639, -0.0043519, -0.0038518, -0.0003049, 0.0003310
2: 0.0104227, 0.0129540, 0.0103585, 0.0130086, -0.0017543, 0.0016156
3: -0.0071692, -0.0060171, -0.0071941, -0.0059879, -0.0007354, 0.0007985
4: 0.0025452, 0.0030351, 0.0025328, 0.0030457, -0.0003395, 0.0003127
5: 0.0120684, 0.0152522, 0.0119877, 0.0153209, -0.0022065, 0.0020320
6: -0.0023303, -0.0015223, -0.0023478, -0.0015018, -0.0005157, 0.0005600
7: -0.0091669, -0.0070762, -0.0092120, -0.0070232, -0.0013344, 0.0014489
8: -0.0043849, -0.0032854, -0.0044087, -0.0032576, -0.0007017, 0.0007620
9: 0.0019458, 0.0032207, 0.0019135, 0.0032482, -0.0008836, 0.0008137

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0004989
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0005058
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876953, 0.9896342, 0.9876364, 0.9896117, -0.0012150, 0.0013421
1: -0.0043299, -0.0038468, -0.0043446, -0.0038524, -0.0003028, 0.0003344
2: 0.0103323, 0.0128924, 0.0103619, 0.0129702, -0.0017723, 0.0016044
3: -0.0071412, -0.0059759, -0.0071766, -0.0059894, -0.0007303, 0.0008067
4: 0.0025277, 0.0030232, 0.0025334, 0.0030383, -0.0003430, 0.0003105
5: 0.0119547, 0.0151747, 0.0119920, 0.0152726, -0.0022290, 0.0020180
6: -0.0023107, -0.0014934, -0.0023355, -0.0015029, -0.0005122, 0.0005658
7: -0.0091161, -0.0070015, -0.0091803, -0.0070260, -0.0013252, 0.0014638
8: -0.0043582, -0.0032462, -0.0043920, -0.0032591, -0.0006969, 0.0007698
9: 0.0019002, 0.0031897, 0.0019152, 0.0032289, -0.0008926, 0.0008081

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004918, upper bound: 0.0004906
time: 0.64 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005165, upper bound: 0.0004906
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876481, 0.9895653, 0.9875963, 0.9896328, -0.0012419, 0.0013416
1: -0.0043417, -0.0038640, -0.0043546, -0.0038472, -0.0003094, 0.0003343
2: 0.0104232, 0.0129549, 0.0103341, 0.0130232, -0.0017716, 0.0016399
3: -0.0071696, -0.0060173, -0.0072007, -0.0059767, -0.0007464, 0.0008064
4: 0.0025453, 0.0030353, 0.0025280, 0.0030485, -0.0003429, 0.0003174
5: 0.0120691, 0.0152533, 0.0119570, 0.0153393, -0.0022282, 0.0020625
6: -0.0023306, -0.0015224, -0.0023524, -0.0014940, -0.0005235, 0.0005655
7: -0.0091676, -0.0070766, -0.0092241, -0.0070030, -0.0013544, 0.0014632
8: -0.0043853, -0.0032857, -0.0044150, -0.0032470, -0.0007123, 0.0007695
9: 0.0019461, 0.0032211, 0.0019012, 0.0032556, -0.0008923, 0.0008259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0004989
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005059
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876949, 0.9896331, 0.9876285, 0.9896308, -0.0012155, 0.0013511
1: -0.0043301, -0.0038471, -0.0043466, -0.0038477, -0.0003029, 0.0003367
2: 0.0103338, 0.0128931, 0.0103368, 0.0129808, -0.0017842, 0.0016050
3: -0.0071415, -0.0059766, -0.0071814, -0.0059780, -0.0007305, 0.0008121
4: 0.0025280, 0.0030233, 0.0025285, 0.0030403, -0.0003453, 0.0003106
5: 0.0119566, 0.0151755, 0.0119604, 0.0152858, -0.0022440, 0.0020187
6: -0.0023109, -0.0014939, -0.0023389, -0.0014948, -0.0005124, 0.0005696
7: -0.0091166, -0.0070027, -0.0091890, -0.0070052, -0.0013256, 0.0014736
8: -0.0043585, -0.0032468, -0.0043966, -0.0032481, -0.0006971, 0.0007750
9: 0.0019010, 0.0031900, 0.0019025, 0.0032342, -0.0008986, 0.0008084

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.91 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004956, upper bound: 0.0004906
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005188, upper bound: 0.0004906
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876950, 0.9895498, 0.9876148, 0.9896491, -0.0013140, 0.0012606
1: -0.0043300, -0.0038679, -0.0043500, -0.0038431, -0.0003274, 0.0003141
2: 0.0104437, 0.0128929, 0.0103124, 0.0129990, -0.0016646, 0.0017352
3: -0.0071414, -0.0060266, -0.0071897, -0.0059669, -0.0007898, 0.0007576
4: 0.0025492, 0.0030233, 0.0025238, 0.0030438, -0.0003222, 0.0003358
5: 0.0120949, 0.0151753, 0.0119298, 0.0153087, -0.0020936, 0.0021824
6: -0.0023108, -0.0015290, -0.0023447, -0.0014871, -0.0005539, 0.0005314
7: -0.0091164, -0.0070936, -0.0092040, -0.0069851, -0.0014331, 0.0013748
8: -0.0043584, -0.0032946, -0.0044045, -0.0032376, -0.0007537, 0.0007230
9: 0.0019564, 0.0031899, 0.0018903, 0.0032433, -0.0008384, 0.0008739

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004945, upper bound: 0.0004943
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004945, upper bound: 0.0005025
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877424, 0.9896185, 0.9876440, 0.9896466, -0.0013094, 0.0012661
1: -0.0043182, -0.0038507, -0.0043427, -0.0038437, -0.0003263, 0.0003155
2: 0.0103529, 0.0128303, 0.0103158, 0.0129602, -0.0016719, 0.0017290
3: -0.0071129, -0.0059853, -0.0071721, -0.0059684, -0.0007870, 0.0007610
4: 0.0025317, 0.0030112, 0.0025245, 0.0030363, -0.0003236, 0.0003346
5: 0.0119807, 0.0150966, 0.0119341, 0.0152600, -0.0021028, 0.0021746
6: -0.0022909, -0.0015000, -0.0023323, -0.0014882, -0.0005519, 0.0005337
7: -0.0090648, -0.0070186, -0.0091721, -0.0069879, -0.0014280, 0.0013808
8: -0.0043312, -0.0032552, -0.0043876, -0.0032390, -0.0007510, 0.0007262
9: 0.0019107, 0.0031584, 0.0018920, 0.0032238, -0.0008420, 0.0008708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.06 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004798, upper bound: 0.0004879
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004893, upper bound: 0.0004879
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876947, 0.9895499, 0.9876080, 0.9896646, -0.0013313, 0.0012979
1: -0.0043301, -0.0038678, -0.0043517, -0.0038393, -0.0003317, 0.0003234
2: 0.0104434, 0.0128933, 0.0102921, 0.0130077, -0.0017139, 0.0017580
3: -0.0071416, -0.0060265, -0.0071937, -0.0059576, -0.0008002, 0.0007801
4: 0.0025492, 0.0030234, 0.0025199, 0.0030455, -0.0003317, 0.0003403
5: 0.0120945, 0.0151759, 0.0119042, 0.0153197, -0.0021556, 0.0022111
6: -0.0023110, -0.0015289, -0.0023475, -0.0014806, -0.0005612, 0.0005471
7: -0.0091168, -0.0070933, -0.0092113, -0.0069683, -0.0014520, 0.0014155
8: -0.0043586, -0.0032945, -0.0044083, -0.0032287, -0.0007636, 0.0007444
9: 0.0019562, 0.0031901, 0.0018800, 0.0032477, -0.0008632, 0.0008854

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.97 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0004945
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0005025
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.9877421, 0.9896185, 0.9876419, 0.9896622, -0.0013262, 0.0013033
1: -0.0043183, -0.0038508, -0.0043433, -0.0038399, -0.0003305, 0.0003248
2: 0.0103530, 0.0128308, 0.0102952, 0.0129630, -0.0017211, 0.0017513
3: -0.0071131, -0.0059853, -0.0071733, -0.0059590, -0.0007971, 0.0007833
4: 0.0025317, 0.0030113, 0.0025205, 0.0030369, -0.0003331, 0.0003390
5: 0.0119808, 0.0150972, 0.0119081, 0.0152635, -0.0021646, 0.0022026
6: -0.0022910, -0.0015000, -0.0023332, -0.0014816, -0.0005591, 0.0005494
7: -0.0090651, -0.0070186, -0.0091743, -0.0069709, -0.0014464, 0.0014215
8: -0.0043314, -0.0032552, -0.0043888, -0.0032301, -0.0007607, 0.0007475
9: 0.0019107, 0.0031586, 0.0018816, 0.0032252, -0.0008668, 0.0008820

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.98 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004827, upper bound: 0.0004879
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0004905, upper bound: 0.0004879
time: 0.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.9876487, 0.9895658, 0.9875926, 0.9896496, -0.0012945, 0.0013564
1: -0.0043416, -0.0038639, -0.0043555, -0.0038430, -0.0003226, 0.0003380
2: 0.0104227, 0.0129540, 0.0103118, 0.0130281, -0.0017911, 0.0017094
3: -0.0071692, -0.0060171, -0.0072029, -0.0059666, -0.0007781, 0.0008152
4: 0.0025452, 0.0030351, 0.0025237, 0.0030494, -0.0003467, 0.0003309
5: 0.0120684, 0.0152522, 0.0119290, 0.0153453, -0.0022528, 0.0021500
6: -0.0023303, -0.0015223, -0.0023540, -0.0014869, -0.0005457, 0.0005718
7: -0.0091669, -0.0070762, -0.0092281, -0.0069846, -0.0014119, 0.0014794
8: -0.0043849, -0.0032854, -0.0044171, -0.0032373, -0.0007425, 0.0007780
9: 0.0019458, 0.0032207, 0.0018900, 0.0032580, -0.0009021, 0.0008610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005374, upper bound: 0.0004989
time: 0.76 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005374, upper bound: 0.0005058
time: 0.68 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.9876953, 0.9896342, 0.9876215, 0.9896472, -0.0012863, 0.0013658
1: -0.0043299, -0.0038468, -0.0043484, -0.0038436, -0.0003205, 0.0003403
2: 0.0103323, 0.0128924, 0.0103150, 0.0129900, -0.0018036, 0.0016985
3: -0.0071412, -0.0059759, -0.0071856, -0.0059681, -0.0007731, 0.0008209
4: 0.0025277, 0.0030232, 0.0025243, 0.0030421, -0.0003491, 0.0003287
5: 0.0119547, 0.0151747, 0.0119330, 0.0152975, -0.0022684, 0.0021362
6: -0.0023107, -0.0014934, -0.0023418, -0.0014879, -0.0005422, 0.0005758
7: -0.0091161, -0.0070015, -0.0091966, -0.0069873, -0.0014028, 0.0014896
8: -0.0043582, -0.0032462, -0.0044006, -0.0032387, -0.0007377, 0.0007834
9: 0.0019002, 0.0031897, 0.0018916, 0.0032388, -0.0009084, 0.0008554

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005149, upper bound: 0.0004906
time: 0.71 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005269, upper bound: 0.0004906
time: 0.66 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.9876481, 0.9895653, 0.9875830, 0.9896652, -0.0013087, 0.0013791
1: -0.0043417, -0.0038640, -0.0043579, -0.0038391, -0.0003261, 0.0003436
2: 0.0104232, 0.0129549, 0.0102914, 0.0130408, -0.0018211, 0.0017282
3: -0.0071696, -0.0060173, -0.0072087, -0.0059573, -0.0007866, 0.0008289
4: 0.0025453, 0.0030353, 0.0025198, 0.0030519, -0.0003525, 0.0003345
5: 0.0120691, 0.0152533, 0.0119033, 0.0153614, -0.0022905, 0.0021736
6: -0.0023306, -0.0015224, -0.0023580, -0.0014804, -0.0005517, 0.0005813
7: -0.0091676, -0.0070766, -0.0092386, -0.0069677, -0.0014274, 0.0015041
8: -0.0043853, -0.0032857, -0.0044226, -0.0032284, -0.0007506, 0.0007910
9: 0.0019461, 0.0032211, 0.0018797, 0.0032644, -0.0009172, 0.0008704

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0004990
time: 0.61 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0005059
time: 0.61 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.9876949, 0.9896331, 0.9876171, 0.9896629, -0.0012982, 0.0013862
1: -0.0043301, -0.0038471, -0.0043495, -0.0038397, -0.0003235, 0.0003454
2: 0.0103338, 0.0128931, 0.0102944, 0.0129958, -0.0018304, 0.0017143
3: -0.0071415, -0.0059766, -0.0071883, -0.0059587, -0.0007803, 0.0008331
4: 0.0025280, 0.0030233, 0.0025203, 0.0030432, -0.0003543, 0.0003318
5: 0.0119566, 0.0151755, 0.0119070, 0.0153048, -0.0023022, 0.0021562
6: -0.0023109, -0.0014939, -0.0023437, -0.0014813, -0.0005473, 0.0005843
7: -0.0091166, -0.0070027, -0.0092015, -0.0069702, -0.0014159, 0.0015118
8: -0.0043585, -0.0032468, -0.0044031, -0.0032297, -0.0007446, 0.0007950
9: 0.0019010, 0.0031900, 0.0018812, 0.0032418, -0.0009219, 0.0008634

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.99 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005175, upper bound: 0.0004906
time: 0.65 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0005267, upper bound: 0.0004906
time: 0.67 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.9877009, 0.9896263, 0.9876310, 0.9896138, -0.0011645, 0.0012088
1: -0.0043286, -0.0038488, -0.0043460, -0.0038519, -0.0002902, 0.0003012
2: 0.0103426, 0.0128851, 0.0103592, 0.0129774, -0.0015962, 0.0015377
3: -0.0071379, -0.0059806, -0.0071799, -0.0059882, -0.0006999, 0.0007265
4: 0.0025297, 0.0030218, 0.0025329, 0.0030396, -0.0003089, 0.0002976
5: 0.0119678, 0.0151655, 0.0119885, 0.0152816, -0.0020077, 0.0019340
6: -0.0023083, -0.0014967, -0.0023378, -0.0015020, -0.0004909, 0.0005096
7: -0.0091100, -0.0070101, -0.0091862, -0.0070237, -0.0012701, 0.0013184
8: -0.0043550, -0.0032507, -0.0043951, -0.0032579, -0.0006679, 0.0006933
9: 0.0019055, 0.0031860, 0.0019138, 0.0032325, -0.0008040, 0.0007745

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.95 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004738, upper bound: 0.0005275
time: 0.66 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004738, upper bound: 0.0005277
time: 0.65 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.9877540, 0.9896901, 0.9876598, 0.9896110, -0.0011522, 0.0012219
1: -0.0043153, -0.0038329, -0.0043388, -0.0038526, -0.0002871, 0.0003045
2: 0.0102585, 0.0128150, 0.0103628, 0.0129395, -0.0016136, 0.0015215
3: -0.0071060, -0.0059424, -0.0071626, -0.0059898, -0.0006925, 0.0007344
4: 0.0025134, 0.0030082, 0.0025336, 0.0030323, -0.0003123, 0.0002945
5: 0.0118620, 0.0150774, 0.0119931, 0.0152339, -0.0020294, 0.0019137
6: -0.0022860, -0.0014699, -0.0023257, -0.0015032, -0.0004857, 0.0005151
7: -0.0090521, -0.0069406, -0.0091549, -0.0070267, -0.0012567, 0.0013327
8: -0.0043246, -0.0032141, -0.0043786, -0.0032594, -0.0006609, 0.0007009
9: 0.0018631, 0.0031507, 0.0019156, 0.0032134, -0.0008127, 0.0007663

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 255

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004522, upper bound: 0.0005153
time: 0.69 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004732, upper bound: 0.0005155
time: 0.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.9877006, 0.9896313, 0.9876196, 0.9896323, -0.0011831, 0.0012129
1: -0.0043287, -0.0038476, -0.0043488, -0.0038473, -0.0002948, 0.0003022
2: 0.0103361, 0.0128856, 0.0103348, 0.0129925, -0.0016016, 0.0015623
3: -0.0071381, -0.0059777, -0.0071868, -0.0059771, -0.0007111, 0.0007290
4: 0.0025284, 0.0030219, 0.0025282, 0.0030426, -0.0003100, 0.0003024
5: 0.0119596, 0.0151661, 0.0119579, 0.0153006, -0.0020144, 0.0019649
6: -0.0023085, -0.0014946, -0.0023426, -0.0014942, -0.0004987, 0.0005113
7: -0.0091104, -0.0070047, -0.0091987, -0.0070036, -0.0012903, 0.0013228
8: -0.0043552, -0.0032478, -0.0044017, -0.0032473, -0.0006786, 0.0006957
9: 0.0019022, 0.0031862, 0.0019015, 0.0032401, -0.0008067, 0.0007868

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=0, inp2_unstable=0, delta_unstable=10

Time for backsubstitution: 2.03 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 255
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 69
type: B, layer: 1, pos: 200
type: B, layer: 1, pos: 183
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 14

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 132

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004746, upper bound: 0.0005277
time: 0.68 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0004746, upper bound: 0.0005279
time: 0.69 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.62 seconds
IS_A1_B1_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004733, upper bound: 0.0005021
IS_A1_B1_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004733, upper bound: 0.0005061
IS_A1_B1_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004515, upper bound: 0.0004921
IS_A1_B1_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004722, upper bound: 0.0004921
IS_A1_B1_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004758, upper bound: 0.0005022
IS_A1_B1_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004758, upper bound: 0.0005061
IS_A1_B1_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004522, upper bound: 0.0004921
IS_A1_B1_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004696, upper bound: 0.0004921
IS_A1_B1_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004728, upper bound: 0.0005268
IS_A1_B1_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004728, upper bound: 0.0005270
IS_A1_B1_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004515, upper bound: 0.0005146
IS_A1_B1_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004716, upper bound: 0.0005146
IS_A1_B1_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004734, upper bound: 0.0005270
IS_A1_B1_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004734, upper bound: 0.0005271
IS_A1_B1_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004522, upper bound: 0.0005146
IS_A1_B1_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004690, upper bound: 0.0005146
IS_A1_B1_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005021
IS_A1_B1_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005060
IS_A1_B1_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004723, upper bound: 0.0004921
IS_A1_B1_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004810, upper bound: 0.0004921
IS_A1_B1_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005022
IS_A1_B1_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005061
IS_A1_B1_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004724, upper bound: 0.0004921
IS_A1_B1_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004792, upper bound: 0.0004921
IS_A1_B1_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005268
IS_A1_B1_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004857, upper bound: 0.0005269
IS_A1_B1_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004723, upper bound: 0.0005146
IS_A1_B1_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004809, upper bound: 0.0005146
IS_A1_B1_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005269
IS_A1_B1_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004874, upper bound: 0.0005270
IS_A1_B1_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004724, upper bound: 0.0005146
IS_A1_B1_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004784, upper bound: 0.0005146
IS_A1_B1_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005062
IS_A1_B1_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005071, upper bound: 0.0005095
IS_A1_B1_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004788, upper bound: 0.0004946
IS_A1_B1_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004986, upper bound: 0.0004946
IS_A1_B1_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005063, upper bound: 0.0005063
IS_A1_B1_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005063, upper bound: 0.0005096
IS_A1_B1_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004777, upper bound: 0.0004946
IS_A1_B1_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004946, upper bound: 0.0004946
IS_A1_B1_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005006, upper bound: 0.0005316
IS_A1_B1_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005006, upper bound: 0.0005318
IS_A1_B1_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004788, upper bound: 0.0005189
IS_A1_B1_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004957, upper bound: 0.0005189
IS_A1_B1_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0005316
IS_A1_B1_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0005318
IS_A1_B1_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004777, upper bound: 0.0005189
IS_A1_B1_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004906, upper bound: 0.0005189
IS_A1_B1_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005177, upper bound: 0.0005062
IS_A1_B1_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005177, upper bound: 0.0005095
IS_A1_B1_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0004946
IS_A1_B1_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005088, upper bound: 0.0004946
IS_A1_B1_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005195, upper bound: 0.0005063
IS_A1_B1_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005195, upper bound: 0.0005096
IS_A1_B1_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004989, upper bound: 0.0004946
IS_A1_B1_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005062, upper bound: 0.0004946
IS_A1_B1_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005315
IS_A1_B1_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005104, upper bound: 0.0005317
IS_A1_B1_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004986, upper bound: 0.0005188
IS_A1_B1_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0005188
IS_A1_B1_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005315
IS_A1_B1_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005103, upper bound: 0.0005317
IS_A1_B1_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004982, upper bound: 0.0005188
IS_A1_B1_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005017, upper bound: 0.0005188
IS_A1_B2_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004811, upper bound: 0.0004943
IS_A1_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004811, upper bound: 0.0005025
IS_A1_B2_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004607, upper bound: 0.0004879
IS_A1_B2_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004795, upper bound: 0.0004879
IS_A1_B2_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0004945
IS_A1_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004899, upper bound: 0.0005026
IS_A1_B2_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004633, upper bound: 0.0004879
IS_A1_B2_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004808, upper bound: 0.0004879
IS_A1_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0004989
IS_A1_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0005058
IS_A1_B2_A1_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004918, upper bound: 0.0004906
IS_A1_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005165, upper bound: 0.0004906
IS_A1_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0004989
IS_A1_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005059
IS_A1_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004956, upper bound: 0.0004906
IS_A1_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005188, upper bound: 0.0004906
IS_A1_B2_A1_B2_A1_B1_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004945, upper bound: 0.0004943
IS_A1_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004945, upper bound: 0.0005025
IS_A1_B2_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004798, upper bound: 0.0004879
IS_A1_B2_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004893, upper bound: 0.0004879
IS_A1_B2_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0004945
IS_A1_B2_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005012, upper bound: 0.0005025
IS_A1_B2_A1_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004827, upper bound: 0.0004879
IS_A1_B2_A1_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004905, upper bound: 0.0004879
IS_A1_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005374, upper bound: 0.0004989
IS_A1_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005374, upper bound: 0.0005058
IS_A1_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005149, upper bound: 0.0004906
IS_A1_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005269, upper bound: 0.0004906
IS_A1_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0004990
IS_A1_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005415, upper bound: 0.0005059
IS_A1_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005175, upper bound: 0.0004906
IS_A1_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0005267, upper bound: 0.0004906
IS_A1_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004738, upper bound: 0.0005275
IS_A1_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004738, upper bound: 0.0005277
IS_A1_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004522, upper bound: 0.0005153
IS_A1_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004732, upper bound: 0.0005155
IS_A1_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004746, upper bound: 0.0005277
IS_A1_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.62
Output dim: 0, lower bound: -0.0004746, upper bound: 0.0005279
IS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004845, upper bound: 0.0005279
IS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005323
IS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005125, upper bound: 0.0005323
IS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005324
IS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005081, upper bound: 0.0005324
IS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004868, upper bound: 0.0005276
IS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004966, upper bound: 0.0005276
IS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004886, upper bound: 0.0005278
IS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004947, upper bound: 0.0005278
IS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005119, upper bound: 0.0005322
IS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005249, upper bound: 0.0005322
IS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005120, upper bound: 0.0005323
IS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005202, upper bound: 0.0005323
IS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004731, upper bound: 0.0005184
IS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004856, upper bound: 0.0005184
IS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0005185
IS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004826, upper bound: 0.0005185
IS_A2_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004726, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004850, upper bound: 0.0005368
IS_A2_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004725, upper bound: 0.0005369
IS_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004821, upper bound: 0.0005369
IS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004731, upper bound: 0.0005184
IS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004856, upper bound: 0.0005184
IS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004747, upper bound: 0.0005184
IS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004826, upper bound: 0.0005185
IS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004726, upper bound: 0.0005368
IS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004850, upper bound: 0.0005368
IS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004725, upper bound: 0.0005369
IS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004821, upper bound: 0.0005369
IS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005066, upper bound: 0.0005220
IS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0005220
IS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005070, upper bound: 0.0005221
IS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005096, upper bound: 0.0005221
IS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005418
IS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005101, upper bound: 0.0005418
IS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004990, upper bound: 0.0005419
IS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005059, upper bound: 0.0005419
IS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005066, upper bound: 0.0005220
IS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005114, upper bound: 0.0005220
IS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005068, upper bound: 0.0005221
IS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005094, upper bound: 0.0005221
IS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005418
IS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005101, upper bound: 0.0005418
IS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004986, upper bound: 0.0005419
IS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005057, upper bound: 0.0005419
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004805, upper bound: 0.0005137
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004912, upper bound: 0.0005137
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004890, upper bound: 0.0005142
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004939, upper bound: 0.0005142
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005250, upper bound: 0.0005175
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005291, upper bound: 0.0005175
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005180
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005180
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004805, upper bound: 0.0005138
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004912, upper bound: 0.0005137
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004890, upper bound: 0.0005142
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004939, upper bound: 0.0005142
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005246, upper bound: 0.0005175
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005290, upper bound: 0.0005175
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005342, upper bound: 0.0005180
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005316, upper bound: 0.0005180
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004736, upper bound: 0.0005376
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005376
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004737, upper bound: 0.0005379
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004832, upper bound: 0.0005379
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005426
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005121, upper bound: 0.0005426
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005005, upper bound: 0.0005429
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005081, upper bound: 0.0005429
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004736, upper bound: 0.0005376
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004862, upper bound: 0.0005376
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004737, upper bound: 0.0005379
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0004832, upper bound: 0.0005379
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005018, upper bound: 0.0005426
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005123, upper bound: 0.0005426
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005002, upper bound: 0.0005429
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.62
Output dim: 0, lower bound: -0.0005078, upper bound: 0.0005429

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.43 + 597.08 = 600.52 seconds
