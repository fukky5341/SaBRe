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
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=9, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0061564, 0.0087846, 0.0061564, 0.0087846, -0.0023064, 0.0023064)
1: (0.0022117, 0.0025914, 0.0022117, 0.0025914, -0.0003332, 0.0003332)
2: (0.0095031, 0.0109562, 0.0095031, 0.0109562, -0.0012751, 0.0012751)
3: (-0.0048519, -0.0033491, -0.0048519, -0.0033491, -0.0013188, 0.0013188)
4: (-0.0004114, 0.0012155, -0.0004114, 0.0012155, -0.0014277, 0.0014277)
5: (0.0029631, 0.0045026, 0.0029631, 0.0045026, -0.0013511, 0.0013511)
6: (-0.0105438, -0.0044351, -0.0105438, -0.0044351, -0.0053607, 0.0053607)
7: (0.0034836, 0.0118030, 0.0034836, 0.0118030, -0.0073008, 0.0073008)
8: (0.9916677, 0.9975281, 0.9916677, 0.9975281, -0.0051428, 0.0051428)
9: (-0.0136435, -0.0083238, -0.0136435, -0.0083238, -0.0046683, 0.0046683)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.74 + 1.65 = 3.39 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0027039, upper bound: 0.0027039

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023067, upper bound: 0.0024402
time: 0.61 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0025561, upper bound: 0.0025562
time: 0.81 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.59 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 8, lower bound: -0.0023067, upper bound: 0.0024402
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.59
Output dim: 8, lower bound: -0.0025561, upper bound: 0.0025562

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0066004, 0.0089811, 0.0062872, 0.0087803, -0.0017792, 0.0022571
1: 0.0022759, 0.0026198, 0.0022306, 0.0025908, -0.0002570, 0.0003261
2: 0.0093944, 0.0107107, 0.0095055, 0.0108838, -0.0012479, 0.0009837
3: -0.0049643, -0.0036030, -0.0048495, -0.0034239, -0.0012906, 0.0010173
4: -0.0001365, 0.0013372, -0.0003304, 0.0012129, -0.0011013, 0.0013972
5: 0.0028479, 0.0042425, 0.0029656, 0.0044260, -0.0013222, 0.0010422
6: -0.0110006, -0.0054672, -0.0105339, -0.0047393, -0.0052461, 0.0041353
7: 0.0048891, 0.0124252, 0.0038978, 0.0117895, -0.0056319, 0.0071448
8: 0.9926578, 0.9979665, 0.9919596, 0.9975187, -0.0039672, 0.0050329
9: -0.0140414, -0.0092226, -0.0136349, -0.0085887, -0.0045686, 0.0036012

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020806, upper bound: 0.0022350
time: 0.59 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020768, upper bound: 0.0022350
time: 0.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0061983, 0.0087823, 0.0061564, 0.0087846, -0.0019200, 0.0023044
1: 0.0022178, 0.0025911, 0.0022117, 0.0025914, -0.0002774, 0.0003329
2: 0.0095044, 0.0109330, 0.0095031, 0.0109562, -0.0012741, 0.0010615
3: -0.0048506, -0.0033730, -0.0048519, -0.0033491, -0.0013177, 0.0010979
4: -0.0003854, 0.0012141, -0.0004114, 0.0012155, -0.0011885, 0.0014265
5: 0.0029644, 0.0044781, 0.0029631, 0.0045026, -0.0013499, 0.0011247
6: -0.0105384, -0.0045325, -0.0105438, -0.0044351, -0.0053562, 0.0044626
7: 0.0036161, 0.0117957, 0.0034836, 0.0118030, -0.0060777, 0.0072946
8: 0.9917611, 0.9975231, 0.9916677, 0.9975281, -0.0042813, 0.0051385
9: -0.0136388, -0.0084086, -0.0136435, -0.0083238, -0.0046644, 0.0038862

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=9, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024402, upper bound: 0.0023067
time: 0.59 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0024402, upper bound: 0.0025562
time: 0.74 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.08 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 8, lower bound: -0.0020806, upper bound: 0.0022350
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 8, lower bound: -0.0020768, upper bound: 0.0022350
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 8, lower bound: -0.0024402, upper bound: 0.0023067
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.08
Output dim: 8, lower bound: -0.0024402, upper bound: 0.0025562

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: 0.0066008, 0.0089760, 0.0062922, 0.0087047, -0.0016942, 0.0022454
1: 0.0022759, 0.0026191, 0.0022313, 0.0025799, -0.0002448, 0.0003244
2: 0.0093972, 0.0107105, 0.0095473, 0.0108811, -0.0012414, 0.0009367
3: -0.0049614, -0.0036032, -0.0048062, -0.0034267, -0.0012839, 0.0009688
4: -0.0001363, 0.0013340, -0.0003273, 0.0011660, -0.0010487, 0.0013899
5: 0.0028509, 0.0042423, 0.0030099, 0.0044231, -0.0013153, 0.0009925
6: -0.0109888, -0.0054680, -0.0103580, -0.0047507, -0.0052189, 0.0039378
7: 0.0048902, 0.0124091, 0.0039133, 0.0115500, -0.0053630, 0.0071077
8: 0.9926586, 0.9979550, 0.9919705, 0.9973499, -0.0037778, 0.0050068
9: -0.0140311, -0.0092233, -0.0134817, -0.0085987, -0.0045448, 0.0034292

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020684, upper bound: 0.0022235
time: 0.62 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020684, upper bound: 0.0022350
time: 0.59 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: 0.0066041, 0.0089310, 0.0061534, 0.0086363, -0.0016998, 0.0023818
1: 0.0022764, 0.0026126, 0.0022113, 0.0025700, -0.0002456, 0.0003441
2: 0.0094222, 0.0107086, 0.0095851, 0.0109578, -0.0013168, 0.0009398
3: -0.0049356, -0.0036051, -0.0047671, -0.0033474, -0.0013619, 0.0009720
4: -0.0001342, 0.0013061, -0.0004132, 0.0011237, -0.0010522, 0.0014744
5: 0.0028773, 0.0042404, 0.0030499, 0.0045044, -0.0013952, 0.0009958
6: -0.0108840, -0.0054758, -0.0101992, -0.0044282, -0.0055359, 0.0039509
7: 0.0049008, 0.0122664, 0.0034742, 0.0113337, -0.0053807, 0.0075394
8: 0.9926661, 0.9978545, 0.9916612, 0.9971976, -0.0037903, 0.0053109
9: -0.0139398, -0.0092301, -0.0133434, -0.0083178, -0.0048209, 0.0034406

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014373, upper bound: 0.0019487
time: 0.63 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0013824, upper bound: 0.0017543
time: 0.62 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061983, 0.0087823, 0.0066004, 0.0089811, -0.0023827, 0.0017808
1: 0.0022178, 0.0025911, 0.0022759, 0.0026198, -0.0003442, 0.0002573
2: 0.0095044, 0.0109330, 0.0093944, 0.0107107, -0.0009845, 0.0013173
3: -0.0048506, -0.0033730, -0.0049643, -0.0036030, -0.0010183, 0.0013624
4: -0.0003854, 0.0012141, -0.0001365, 0.0013372, -0.0014749, 0.0011023
5: 0.0029644, 0.0044781, 0.0028479, 0.0042425, -0.0010432, 0.0013958
6: -0.0105384, -0.0045325, -0.0110006, -0.0054672, -0.0041390, 0.0055380
7: 0.0036161, 0.0117957, 0.0048891, 0.0124252, -0.0075423, 0.0056370
8: 0.9917611, 0.9975231, 0.9926578, 0.9979665, -0.0053130, 0.0039708
9: -0.0136388, -0.0084086, -0.0140414, -0.0092226, -0.0036044, 0.0048228

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0020806
time: 0.60 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0020768
time: 0.65 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061983, 0.0087823, 0.0061983, 0.0087823, -0.0019180, 0.0019180
1: 0.0022178, 0.0025911, 0.0022178, 0.0025911, -0.0002771, 0.0002771
2: 0.0095044, 0.0109330, 0.0095044, 0.0109330, -0.0010604, 0.0010604
3: -0.0048506, -0.0033730, -0.0048506, -0.0033730, -0.0010967, 0.0010967
4: -0.0003854, 0.0012141, -0.0003854, 0.0012141, -0.0011873, 0.0011873
5: 0.0029644, 0.0044781, 0.0029644, 0.0044781, -0.0011236, 0.0011236
6: -0.0105384, -0.0045325, -0.0105384, -0.0045325, -0.0044579, 0.0044579
7: 0.0036161, 0.0117957, 0.0036161, 0.0117957, -0.0060713, 0.0060713
8: 0.9917611, 0.9975231, 0.9917611, 0.9975231, -0.0042768, 0.0042768
9: -0.0136388, -0.0084086, -0.0136388, -0.0084086, -0.0038822, 0.0038822

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=8, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0022994
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0023136
time: 0.78 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.23 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0020684, upper bound: 0.0022235
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0020684, upper bound: 0.0022350
IS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0014373, upper bound: 0.0019487
IS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0013824, upper bound: 0.0017543
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0020806
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0020768
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0022994
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.23
Output dim: 8, lower bound: -0.0022350, upper bound: 0.0023136

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0066060, 0.0089022, 0.0062922, 0.0087047, -0.0016889, 0.0021583
1: 0.0022767, 0.0026084, 0.0022313, 0.0025799, -0.0002440, 0.0003118
2: 0.0094381, 0.0107076, 0.0095473, 0.0108811, -0.0011932, 0.0009337
3: -0.0049192, -0.0036062, -0.0048062, -0.0034267, -0.0012341, 0.0009657
4: -0.0001330, 0.0012883, -0.0003273, 0.0011660, -0.0010454, 0.0013360
5: 0.0028942, 0.0042393, 0.0030099, 0.0044231, -0.0012643, 0.0009893
6: -0.0108172, -0.0054802, -0.0103580, -0.0047507, -0.0050164, 0.0039254
7: 0.0049068, 0.0121753, 0.0039133, 0.0115500, -0.0053461, 0.0068319
8: 0.9926704, 0.9977904, 0.9919705, 0.9973499, -0.0037659, 0.0048125
9: -0.0138816, -0.0092339, -0.0134817, -0.0085987, -0.0043685, 0.0034184

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016310, upper bound: 0.0017659
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014645, upper bound: 0.0017750
time: 0.61 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064575, 0.0088474, 0.0062922, 0.0087047, -0.0018790, 0.0021617
1: 0.0022552, 0.0026005, 0.0022313, 0.0025799, -0.0002715, 0.0003123
2: 0.0094684, 0.0107897, 0.0095473, 0.0108811, -0.0011952, 0.0010388
3: -0.0048878, -0.0035212, -0.0048062, -0.0034267, -0.0012361, 0.0010744
4: -0.0002250, 0.0012544, -0.0003273, 0.0011660, -0.0011631, 0.0013382
5: 0.0029263, 0.0043263, 0.0030099, 0.0044231, -0.0012663, 0.0011007
6: -0.0106898, -0.0051349, -0.0103580, -0.0047507, -0.0050245, 0.0043673
7: 0.0044366, 0.0120019, 0.0039133, 0.0115500, -0.0059479, 0.0068429
8: 0.9923390, 0.9976682, 0.9919705, 0.9973499, -0.0041898, 0.0048203
9: -0.0137707, -0.0089332, -0.0134817, -0.0085987, -0.0043756, 0.0038032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016310, upper bound: 0.0017659
time: 0.56 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014645, upper bound: 0.0017750
time: 0.57 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062033, 0.0087066, 0.0066008, 0.0089760, -0.0023708, 0.0016959
1: 0.0022185, 0.0025802, 0.0022759, 0.0026191, -0.0003425, 0.0002450
2: 0.0095462, 0.0109302, 0.0093972, 0.0107105, -0.0009376, 0.0013108
3: -0.0048073, -0.0033759, -0.0049614, -0.0036032, -0.0009698, 0.0013557
4: -0.0003823, 0.0011672, -0.0001363, 0.0013340, -0.0014676, 0.0010498
5: 0.0030087, 0.0044752, 0.0028509, 0.0042423, -0.0009935, 0.0013888
6: -0.0103625, -0.0045442, -0.0109888, -0.0054680, -0.0039418, 0.0055105
7: 0.0036321, 0.0115561, 0.0048902, 0.0124091, -0.0075048, 0.0053684
8: 0.9917724, 0.9973543, 0.9926586, 0.9979550, -0.0052865, 0.0037816
9: -0.0134857, -0.0084188, -0.0140311, -0.0092233, -0.0034327, 0.0047988

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022235, upper bound: 0.0020683
time: 0.66 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022235, upper bound: 0.0020684
time: 0.71 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0060700, 0.0086381, 0.0066041, 0.0089310, -0.0025054, 0.0017015
1: 0.0021992, 0.0025703, 0.0022764, 0.0026126, -0.0003620, 0.0002458
2: 0.0095841, 0.0110039, 0.0094222, 0.0107086, -0.0009407, 0.0013852
3: -0.0047682, -0.0032997, -0.0049356, -0.0036051, -0.0009729, 0.0014326
4: -0.0004649, 0.0011248, -0.0001342, 0.0013061, -0.0015509, 0.0010532
5: 0.0030489, 0.0045533, 0.0028773, 0.0042404, -0.0009967, 0.0014676
6: -0.0102033, -0.0042343, -0.0108840, -0.0054758, -0.0039547, 0.0058232
7: 0.0032100, 0.0113393, 0.0049008, 0.0122664, -0.0079307, 0.0053859
8: 0.9914750, 0.9972015, 0.9926661, 0.9978545, -0.0055865, 0.0037940
9: -0.0133470, -0.0081489, -0.0139398, -0.0092301, -0.0034439, 0.0050711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019487, upper bound: 0.0014373
time: 0.65 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017543, upper bound: 0.0013824
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062033, 0.0087066, 0.0061986, 0.0087772, -0.0019069, 0.0018294
1: 0.0022185, 0.0025802, 0.0022178, 0.0025904, -0.0002755, 0.0002643
2: 0.0095462, 0.0109302, 0.0095072, 0.0109328, -0.0010114, 0.0010542
3: -0.0048073, -0.0033759, -0.0048477, -0.0033732, -0.0010461, 0.0010904
4: -0.0003823, 0.0011672, -0.0003853, 0.0012110, -0.0011804, 0.0011324
5: 0.0030087, 0.0044752, 0.0029674, 0.0044779, -0.0010717, 0.0011170
6: -0.0103625, -0.0045442, -0.0105267, -0.0045332, -0.0042521, 0.0044321
7: 0.0036321, 0.0115561, 0.0036172, 0.0117797, -0.0060361, 0.0057910
8: 0.9917724, 0.9973543, 0.9917619, 0.9975117, -0.0042519, 0.0040793
9: -0.0134857, -0.0084188, -0.0136286, -0.0084092, -0.0037029, 0.0038596

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023801, upper bound: 0.0022952
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0023801, upper bound: 0.0022952
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0060700, 0.0086381, 0.0062015, 0.0087294, -0.0020620, 0.0018355
1: 0.0021992, 0.0025703, 0.0022182, 0.0025834, -0.0002979, 0.0002652
2: 0.0095841, 0.0110039, 0.0095336, 0.0109312, -0.0010148, 0.0011400
3: -0.0047682, -0.0032997, -0.0048204, -0.0033749, -0.0010495, 0.0011791
4: -0.0004649, 0.0011248, -0.0003834, 0.0011814, -0.0012764, 0.0011362
5: 0.0030489, 0.0045533, 0.0029954, 0.0044762, -0.0010752, 0.0012079
6: -0.0102033, -0.0042343, -0.0104156, -0.0045400, -0.0042661, 0.0047926
7: 0.0032100, 0.0113393, 0.0036264, 0.0116284, -0.0065272, 0.0058101
8: 0.9914750, 0.9972015, 0.9917684, 0.9974052, -0.0045979, 0.0040928
9: -0.0133470, -0.0081489, -0.0135319, -0.0084152, -0.0037151, 0.0041737

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=8, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0020736
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0021595
time: 0.80 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.19 seconds
IS_A1_B1_A1_B1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0016310, upper bound: 0.0017659
IS_A1_B1_A1_B2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0014645, upper bound: 0.0017750
IS_A1_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0016310, upper bound: 0.0017659
IS_A1_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0014645, upper bound: 0.0017750
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0022235, upper bound: 0.0020683
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0022235, upper bound: 0.0020684
IS_A2_B1_A2_B1, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0019487, upper bound: 0.0014373
IS_A2_B1_A2_B2, status: Status.VERIFIED, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0017543, upper bound: 0.0013824
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0023801, upper bound: 0.0022952
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0023801, upper bound: 0.0022952
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0020736
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.19
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0021595

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062033, 0.0087066, 0.0066060, 0.0089022, -0.0022837, 0.0016906
1: 0.0022185, 0.0025802, 0.0022767, 0.0026084, -0.0003299, 0.0002442
2: 0.0095462, 0.0109302, 0.0094381, 0.0107076, -0.0009347, 0.0012626
3: -0.0048073, -0.0033759, -0.0049192, -0.0036062, -0.0009667, 0.0013058
4: -0.0003823, 0.0011672, -0.0001330, 0.0012883, -0.0014137, 0.0010465
5: 0.0030087, 0.0044752, 0.0028942, 0.0042393, -0.0009904, 0.0013378
6: -0.0103625, -0.0045442, -0.0108172, -0.0054802, -0.0039294, 0.0053080
7: 0.0036321, 0.0115561, 0.0049068, 0.0121753, -0.0072290, 0.0053515
8: 0.9917724, 0.9973543, 0.9926704, 0.9977904, -0.0050923, 0.0037697
9: -0.0134857, -0.0084188, -0.0138816, -0.0092339, -0.0034219, 0.0046224

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017659, upper bound: 0.0016310
time: 0.63 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017750, upper bound: 0.0014645
time: 0.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062033, 0.0087066, 0.0064575, 0.0088474, -0.0022872, 0.0018807
1: 0.0022185, 0.0025802, 0.0022552, 0.0026005, -0.0003304, 0.0002717
2: 0.0095462, 0.0109302, 0.0094684, 0.0107897, -0.0010398, 0.0012645
3: -0.0048073, -0.0033759, -0.0048878, -0.0035212, -0.0010754, 0.0013078
4: -0.0003823, 0.0011672, -0.0002250, 0.0012544, -0.0014158, 0.0011642
5: 0.0030087, 0.0044752, 0.0029263, 0.0043263, -0.0011017, 0.0013398
6: -0.0103625, -0.0045442, -0.0106898, -0.0051349, -0.0043713, 0.0053161
7: 0.0036321, 0.0115561, 0.0044366, 0.0120019, -0.0072401, 0.0059533
8: 0.9917724, 0.9973543, 0.9923390, 0.9976682, -0.0051000, 0.0041936
9: -0.0134857, -0.0084188, -0.0137707, -0.0089332, -0.0038067, 0.0046295

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017659, upper bound: 0.0016310
time: 0.61 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017750, upper bound: 0.0014645
time: 0.60 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062033, 0.0087066, 0.0062033, 0.0087066, -0.0018247, 0.0018247
1: 0.0022185, 0.0025802, 0.0022185, 0.0025802, -0.0002636, 0.0002636
2: 0.0095462, 0.0109302, 0.0095462, 0.0109302, -0.0010088, 0.0010088
3: -0.0048073, -0.0033759, -0.0048073, -0.0033759, -0.0010434, 0.0010434
4: -0.0003823, 0.0011672, -0.0003823, 0.0011672, -0.0011295, 0.0011295
5: 0.0030087, 0.0044752, 0.0030087, 0.0044752, -0.0010689, 0.0010689
6: -0.0103625, -0.0045442, -0.0103625, -0.0045442, -0.0042411, 0.0042411
7: 0.0036321, 0.0115561, 0.0036321, 0.0115561, -0.0057760, 0.0057760
8: 0.9917724, 0.9973543, 0.9917724, 0.9973543, -0.0040688, 0.0040688
9: -0.0134857, -0.0084188, -0.0134857, -0.0084188, -0.0036934, 0.0036934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021622
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021622
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062033, 0.0087066, 0.0060700, 0.0086381, -0.0018193, 0.0020119
1: 0.0022185, 0.0025802, 0.0021992, 0.0025703, -0.0002628, 0.0002907
2: 0.0095462, 0.0109302, 0.0095841, 0.0110039, -0.0011123, 0.0010058
3: -0.0048073, -0.0033759, -0.0047682, -0.0032997, -0.0011504, 0.0010403
4: -0.0003823, 0.0011672, -0.0004649, 0.0011248, -0.0011262, 0.0012454
5: 0.0030087, 0.0044752, 0.0030489, 0.0045533, -0.0011786, 0.0010657
6: -0.0103625, -0.0045442, -0.0102033, -0.0042343, -0.0046762, 0.0042285
7: 0.0036321, 0.0115561, 0.0032100, 0.0113393, -0.0057588, 0.0063686
8: 0.9917724, 0.9973543, 0.9914750, 0.9972015, -0.0040566, 0.0044862
9: -0.0134857, -0.0084188, -0.0133470, -0.0081489, -0.0040723, 0.0036823

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021622
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021622
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0060700, 0.0086381, 0.0062817, 0.0087263, -0.0020590, 0.0017521
1: 0.0021992, 0.0025703, 0.0022298, 0.0025830, -0.0002975, 0.0002531
2: 0.0095841, 0.0110039, 0.0095353, 0.0108868, -0.0009687, 0.0011384
3: -0.0047682, -0.0032997, -0.0048186, -0.0034208, -0.0010019, 0.0011774
4: -0.0004649, 0.0011248, -0.0003338, 0.0011794, -0.0012746, 0.0010846
5: 0.0030489, 0.0045533, 0.0029972, 0.0044292, -0.0010264, 0.0012062
6: -0.0102033, -0.0042343, -0.0104083, -0.0047265, -0.0040723, 0.0047858
7: 0.0032100, 0.0113393, 0.0038804, 0.0116184, -0.0065178, 0.0055462
8: 0.9914750, 0.9972015, 0.9919473, 0.9973981, -0.0045913, 0.0039068
9: -0.0133470, -0.0081489, -0.0135255, -0.0085776, -0.0035464, 0.0041677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0020736
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0020736
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061154, 0.0086348, 0.0063128, 0.0089322, -0.0022740, 0.0017730
1: 0.0022058, 0.0025698, 0.0022343, 0.0026127, -0.0003285, 0.0002561
2: 0.0095859, 0.0109788, 0.0094215, 0.0108697, -0.0009802, 0.0012572
3: -0.0047663, -0.0033256, -0.0049363, -0.0034385, -0.0010138, 0.0013003
4: -0.0004368, 0.0011228, -0.0003145, 0.0013069, -0.0014077, 0.0010975
5: 0.0030508, 0.0045267, 0.0028766, 0.0044110, -0.0010386, 0.0013321
6: -0.0101957, -0.0043398, -0.0108869, -0.0047987, -0.0041209, 0.0052854
7: 0.0033537, 0.0113290, 0.0039787, 0.0122703, -0.0071983, 0.0056123
8: 0.9915763, 0.9971942, 0.9920166, 0.9978573, -0.0050706, 0.0039534
9: -0.0133404, -0.0082408, -0.0139423, -0.0086405, -0.0035886, 0.0046028

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=7, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0021595
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0021595
time: 0.75 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.28 seconds
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017659, upper bound: 0.0016310
IS_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017750, upper bound: 0.0014645
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017659, upper bound: 0.0016310
IS_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0017750, upper bound: 0.0014645
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021622
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021622
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021622
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0022376, upper bound: 0.0021622
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0020736
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0020736
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0021595
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.28
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0021595

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062836, 0.0087035, 0.0062033, 0.0087066, -0.0017412, 0.0018217
1: 0.0022301, 0.0025797, 0.0022185, 0.0025802, -0.0002516, 0.0002632
2: 0.0095479, 0.0108858, 0.0095462, 0.0109302, -0.0010072, 0.0009627
3: -0.0048055, -0.0034218, -0.0048073, -0.0033759, -0.0010416, 0.0009957
4: -0.0003326, 0.0011653, -0.0003823, 0.0011672, -0.0010779, 0.0011276
5: 0.0030106, 0.0044281, 0.0030087, 0.0044752, -0.0010671, 0.0010200
6: -0.0103552, -0.0047308, -0.0103625, -0.0045442, -0.0042341, 0.0040471
7: 0.0038863, 0.0115462, 0.0036321, 0.0115561, -0.0055118, 0.0057664
8: 0.9919515, 0.9973474, 0.9917724, 0.9973543, -0.0038827, 0.0040620
9: -0.0134793, -0.0085813, -0.0134857, -0.0084188, -0.0036872, 0.0035244

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021897, upper bound: 0.0021064
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021897, upper bound: 0.0021926
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063152, 0.0089138, 0.0062480, 0.0087034, -0.0017594, 0.0020376
1: 0.0022347, 0.0026101, 0.0022249, 0.0025797, -0.0002542, 0.0002944
2: 0.0094316, 0.0108684, 0.0095480, 0.0109055, -0.0011266, 0.0009727
3: -0.0049258, -0.0034399, -0.0048055, -0.0034014, -0.0011651, 0.0010060
4: -0.0003131, 0.0012955, -0.0003547, 0.0011653, -0.0010891, 0.0012613
5: 0.0028873, 0.0044096, 0.0030106, 0.0044490, -0.0011936, 0.0010306
6: -0.0108442, -0.0048042, -0.0103551, -0.0046480, -0.0047360, 0.0040893
7: 0.0039862, 0.0122122, 0.0037734, 0.0115460, -0.0055692, 0.0064500
8: 0.9920218, 0.9978164, 0.9918719, 0.9973471, -0.0039231, 0.0045435
9: -0.0139052, -0.0086452, -0.0134792, -0.0085092, -0.0041243, 0.0035611

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022820, upper bound: 0.0021064
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022820, upper bound: 0.0021926
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062836, 0.0087035, 0.0060700, 0.0086381, -0.0017358, 0.0020089
1: 0.0022301, 0.0025797, 0.0021992, 0.0025703, -0.0002508, 0.0002902
2: 0.0095479, 0.0108858, 0.0095841, 0.0110039, -0.0011107, 0.0009597
3: -0.0048055, -0.0034218, -0.0047682, -0.0032997, -0.0011487, 0.0009925
4: -0.0003326, 0.0011653, -0.0004649, 0.0011248, -0.0010745, 0.0012435
5: 0.0030106, 0.0044281, 0.0030489, 0.0045533, -0.0011768, 0.0010168
6: -0.0103552, -0.0047308, -0.0102033, -0.0042343, -0.0046692, 0.0040345
7: 0.0038863, 0.0115462, 0.0032100, 0.0113393, -0.0054946, 0.0063590
8: 0.9919515, 0.9973474, 0.9914750, 0.9972015, -0.0038705, 0.0044794
9: -0.0134793, -0.0085813, -0.0133470, -0.0081489, -0.0040661, 0.0035134

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0020787
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0021622
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063152, 0.0089138, 0.0061154, 0.0086348, -0.0017540, 0.0022254
1: 0.0022347, 0.0026101, 0.0022058, 0.0025698, -0.0002534, 0.0003215
2: 0.0094316, 0.0108684, 0.0095859, 0.0109788, -0.0012304, 0.0009698
3: -0.0049258, -0.0034399, -0.0047663, -0.0033256, -0.0012725, 0.0010030
4: -0.0003131, 0.0012955, -0.0004368, 0.0011228, -0.0010858, 0.0013776
5: 0.0028873, 0.0044096, 0.0030508, 0.0045267, -0.0013036, 0.0010275
6: -0.0108442, -0.0048042, -0.0101957, -0.0043398, -0.0051724, 0.0040768
7: 0.0039862, 0.0122122, 0.0033537, 0.0113290, -0.0055523, 0.0070444
8: 0.9920218, 0.9978164, 0.9915763, 0.9971942, -0.0039111, 0.0049622
9: -0.0139052, -0.0086452, -0.0133404, -0.0082408, -0.0045044, 0.0035503

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0020787
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0021622
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061497, 0.0086350, 0.0062817, 0.0087263, -0.0019757, 0.0017491
1: 0.0022107, 0.0025698, 0.0022298, 0.0025830, -0.0002854, 0.0002527
2: 0.0095858, 0.0109599, 0.0095353, 0.0108868, -0.0009670, 0.0010923
3: -0.0047664, -0.0033452, -0.0048186, -0.0034208, -0.0010001, 0.0011297
4: -0.0004155, 0.0011229, -0.0003338, 0.0011794, -0.0012230, 0.0010827
5: 0.0030507, 0.0045066, 0.0029972, 0.0044292, -0.0010246, 0.0011574
6: -0.0101962, -0.0044195, -0.0104083, -0.0047265, -0.0040653, 0.0045921
7: 0.0034622, 0.0113296, 0.0038804, 0.0116184, -0.0062540, 0.0055366
8: 0.9916527, 0.9971946, 0.9919473, 0.9973981, -0.0044055, 0.0039001
9: -0.0133408, -0.0083102, -0.0135255, -0.0085776, -0.0035403, 0.0039990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061877, 0.0088396, 0.0062817, 0.0087263, -0.0019681, 0.0020018
1: 0.0022162, 0.0025994, 0.0022298, 0.0025830, -0.0002843, 0.0002892
2: 0.0094727, 0.0109388, 0.0095353, 0.0108868, -0.0011068, 0.0010881
3: -0.0048833, -0.0033670, -0.0048186, -0.0034208, -0.0011447, 0.0011254
4: -0.0003920, 0.0012495, -0.0003338, 0.0011794, -0.0012183, 0.0012392
5: 0.0029309, 0.0044843, 0.0029972, 0.0044292, -0.0011727, 0.0011529
6: -0.0106716, -0.0045080, -0.0104083, -0.0047265, -0.0046528, 0.0045745
7: 0.0035827, 0.0119770, 0.0038804, 0.0116184, -0.0062301, 0.0063367
8: 0.9917376, 0.9976507, 0.9919473, 0.9973981, -0.0043886, 0.0044637
9: -0.0137548, -0.0083873, -0.0135255, -0.0085776, -0.0040519, 0.0039837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061497, 0.0086350, 0.0063128, 0.0089322, -0.0022278, 0.0017480
1: 0.0022107, 0.0025698, 0.0022343, 0.0026127, -0.0003218, 0.0002525
2: 0.0095858, 0.0109599, 0.0094215, 0.0108697, -0.0009664, 0.0012317
3: -0.0047664, -0.0033452, -0.0049363, -0.0034385, -0.0009995, 0.0012739
4: -0.0004155, 0.0011229, -0.0003145, 0.0013069, -0.0013790, 0.0010820
5: 0.0030507, 0.0045066, 0.0028766, 0.0044110, -0.0010240, 0.0013050
6: -0.0101962, -0.0044195, -0.0108869, -0.0047987, -0.0040628, 0.0051779
7: 0.0034622, 0.0113296, 0.0039787, 0.0122703, -0.0070519, 0.0055332
8: 0.9916527, 0.9971946, 0.9920166, 0.9978573, -0.0049675, 0.0038977
9: -0.0133408, -0.0083102, -0.0139423, -0.0086405, -0.0035381, 0.0045092

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021595
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021595
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061877, 0.0088396, 0.0063128, 0.0089322, -0.0020746, 0.0018419
1: 0.0022162, 0.0025994, 0.0022343, 0.0026127, -0.0002997, 0.0002661
2: 0.0094727, 0.0109388, 0.0094215, 0.0108697, -0.0010183, 0.0011470
3: -0.0048833, -0.0033670, -0.0049363, -0.0034385, -0.0010532, 0.0011863
4: -0.0003920, 0.0012495, -0.0003145, 0.0013069, -0.0012842, 0.0011402
5: 0.0029309, 0.0044843, 0.0028766, 0.0044110, -0.0010790, 0.0012153
6: -0.0106716, -0.0045080, -0.0108869, -0.0047987, -0.0042810, 0.0048219
7: 0.0035827, 0.0119770, 0.0039787, 0.0122703, -0.0065671, 0.0058304
8: 0.9917376, 0.9976507, 0.9920166, 0.9978573, -0.0046260, 0.0041071
9: -0.0137548, -0.0083873, -0.0139423, -0.0086405, -0.0037281, 0.0041992

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=7, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0020736
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0020736
time: 0.71 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.21 seconds
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021897, upper bound: 0.0021064
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021897, upper bound: 0.0021926
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0022820, upper bound: 0.0021064
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0022820, upper bound: 0.0021926
IS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0020787
IS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021479, upper bound: 0.0021622
IS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0020787
IS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0022438, upper bound: 0.0021622
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
IS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
IS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021931, upper bound: 0.0020736
IS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021595
IS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0021595
IS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0020736
IS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.21
Output dim: 8, lower bound: -0.0021409, upper bound: 0.0020736

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062836, 0.0087035, 0.0062836, 0.0087035, -0.0017382, 0.0017382
1: 0.0022301, 0.0025797, 0.0022301, 0.0025797, -0.0002511, 0.0002511
2: 0.0095479, 0.0108858, 0.0095479, 0.0108858, -0.0009610, 0.0009610
3: -0.0048055, -0.0034218, -0.0048055, -0.0034218, -0.0009939, 0.0009939
4: -0.0003326, 0.0011653, -0.0003326, 0.0011653, -0.0010760, 0.0010760
5: 0.0030106, 0.0044281, 0.0030106, 0.0044281, -0.0010182, 0.0010182
6: -0.0103552, -0.0047308, -0.0103552, -0.0047308, -0.0040401, 0.0040401
7: 0.0038863, 0.0115462, 0.0038863, 0.0115462, -0.0055022, 0.0055022
8: 0.9919515, 0.9973474, 0.9919515, 0.9973474, -0.0038759, 0.0038759
9: -0.0134793, -0.0085813, -0.0134793, -0.0085813, -0.0035183, 0.0035183

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021349, upper bound: 0.0020364
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0020932
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062836, 0.0087035, 0.0063152, 0.0089138, -0.0019918, 0.0017366
1: 0.0022301, 0.0025797, 0.0022347, 0.0026101, -0.0002878, 0.0002509
2: 0.0095479, 0.0108858, 0.0094316, 0.0108684, -0.0009601, 0.0011012
3: -0.0048055, -0.0034218, -0.0049258, -0.0034399, -0.0009930, 0.0011389
4: -0.0003326, 0.0011653, -0.0003131, 0.0012955, -0.0012330, 0.0010750
5: 0.0030106, 0.0044281, 0.0028873, 0.0044096, -0.0010173, 0.0011668
6: -0.0103552, -0.0047308, -0.0108442, -0.0048042, -0.0040364, 0.0046295
7: 0.0038863, 0.0115462, 0.0039862, 0.0122122, -0.0063050, 0.0054972
8: 0.9919515, 0.9973474, 0.9920218, 0.9978164, -0.0044414, 0.0038723
9: -0.0134793, -0.0085813, -0.0139052, -0.0086452, -0.0035150, 0.0040316

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021349, upper bound: 0.0020866
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0021475
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063152, 0.0089138, 0.0062836, 0.0087035, -0.0017366, 0.0019918
1: 0.0022347, 0.0026101, 0.0022301, 0.0025797, -0.0002509, 0.0002878
2: 0.0094316, 0.0108684, 0.0095479, 0.0108858, -0.0011012, 0.0009601
3: -0.0049258, -0.0034399, -0.0048055, -0.0034218, -0.0011389, 0.0009930
4: -0.0003131, 0.0012955, -0.0003326, 0.0011653, -0.0010750, 0.0012330
5: 0.0028873, 0.0044096, 0.0030106, 0.0044281, -0.0011668, 0.0010173
6: -0.0108442, -0.0048042, -0.0103552, -0.0047308, -0.0046295, 0.0040364
7: 0.0039862, 0.0122122, 0.0038863, 0.0115462, -0.0054972, 0.0063050
8: 0.9920218, 0.9978164, 0.9919515, 0.9973474, -0.0038723, 0.0044414
9: -0.0139052, -0.0086452, -0.0134793, -0.0085813, -0.0040316, 0.0035150

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020057
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020562
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063152, 0.0089138, 0.0063152, 0.0089138, -0.0018279, 0.0018279
1: 0.0022347, 0.0026101, 0.0022347, 0.0026101, -0.0002641, 0.0002641
2: 0.0094316, 0.0108684, 0.0094316, 0.0108684, -0.0010106, 0.0010106
3: -0.0049258, -0.0034399, -0.0049258, -0.0034399, -0.0010452, 0.0010452
4: -0.0003131, 0.0012955, -0.0003131, 0.0012955, -0.0011315, 0.0011315
5: 0.0028873, 0.0044096, 0.0028873, 0.0044096, -0.0010708, 0.0010708
6: -0.0108442, -0.0048042, -0.0108442, -0.0048042, -0.0042484, 0.0042484
7: 0.0039862, 0.0122122, 0.0039862, 0.0122122, -0.0057860, 0.0057860
8: 0.9920218, 0.9978164, 0.9920218, 0.9978164, -0.0040758, 0.0040758
9: -0.0139052, -0.0086452, -0.0139052, -0.0086452, -0.0036997, 0.0036997

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020057
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020562
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062836, 0.0087035, 0.0061497, 0.0086350, -0.0017329, 0.0019255
1: 0.0022301, 0.0025797, 0.0022107, 0.0025698, -0.0002504, 0.0002782
2: 0.0095479, 0.0108858, 0.0095858, 0.0109599, -0.0010646, 0.0009581
3: -0.0048055, -0.0034218, -0.0047664, -0.0033452, -0.0011010, 0.0009909
4: -0.0003326, 0.0011653, -0.0004155, 0.0011229, -0.0010727, 0.0011919
5: 0.0030106, 0.0044281, 0.0030507, 0.0045066, -0.0011280, 0.0010151
6: -0.0103552, -0.0047308, -0.0101962, -0.0044195, -0.0044755, 0.0040278
7: 0.0038863, 0.0115462, 0.0034622, 0.0113296, -0.0054855, 0.0060952
8: 0.9919515, 0.9973474, 0.9916527, 0.9971946, -0.0038641, 0.0042936
9: -0.0134793, -0.0085813, -0.0133408, -0.0083102, -0.0038974, 0.0035076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0020119
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0020645
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062836, 0.0087035, 0.0061877, 0.0088396, -0.0019728, 0.0019180
1: 0.0022301, 0.0025797, 0.0022162, 0.0025994, -0.0002850, 0.0002771
2: 0.0095479, 0.0108858, 0.0094727, 0.0109388, -0.0010604, 0.0010907
3: -0.0048055, -0.0034218, -0.0048833, -0.0033670, -0.0010967, 0.0011280
4: -0.0003326, 0.0011653, -0.0003920, 0.0012495, -0.0012212, 0.0011873
5: 0.0030106, 0.0044281, 0.0029309, 0.0044843, -0.0011235, 0.0011556
6: -0.0103552, -0.0047308, -0.0106716, -0.0045080, -0.0044579, 0.0045853
7: 0.0038863, 0.0115462, 0.0035827, 0.0119770, -0.0062447, 0.0060712
8: 0.9919515, 0.9973474, 0.9917376, 0.9976507, -0.0043989, 0.0042767
9: -0.0134793, -0.0085813, -0.0137548, -0.0083873, -0.0038821, 0.0039931

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0020590
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0021161
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063152, 0.0089138, 0.0061497, 0.0086350, -0.0017313, 0.0021791
1: 0.0022347, 0.0026101, 0.0022107, 0.0025698, -0.0002501, 0.0003148
2: 0.0094316, 0.0108684, 0.0095858, 0.0109599, -0.0012048, 0.0009572
3: -0.0049258, -0.0034399, -0.0047664, -0.0033452, -0.0012461, 0.0009900
4: -0.0003131, 0.0012955, -0.0004155, 0.0011229, -0.0010717, 0.0013489
5: 0.0028873, 0.0044096, 0.0030507, 0.0045066, -0.0012765, 0.0010142
6: -0.0108442, -0.0048042, -0.0101962, -0.0044195, -0.0050649, 0.0040241
7: 0.0039862, 0.0122122, 0.0034622, 0.0113296, -0.0054804, 0.0068980
8: 0.9920218, 0.9978164, 0.9916527, 0.9971946, -0.0038605, 0.0048591
9: -0.0139052, -0.0086452, -0.0133408, -0.0083102, -0.0044108, 0.0035043

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0019844
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0020287
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063152, 0.0089138, 0.0061877, 0.0088396, -0.0018252, 0.0020207
1: 0.0022347, 0.0026101, 0.0022162, 0.0025994, -0.0002637, 0.0002919
2: 0.0094316, 0.0108684, 0.0094727, 0.0109388, -0.0011172, 0.0010091
3: -0.0049258, -0.0034399, -0.0048833, -0.0033670, -0.0011555, 0.0010436
4: -0.0003131, 0.0012955, -0.0003920, 0.0012495, -0.0011298, 0.0012509
5: 0.0028873, 0.0044096, 0.0029309, 0.0044843, -0.0011837, 0.0010692
6: -0.0108442, -0.0048042, -0.0106716, -0.0045080, -0.0046968, 0.0042422
7: 0.0039862, 0.0122122, 0.0035827, 0.0119770, -0.0057775, 0.0063966
8: 0.9920218, 0.9978164, 0.9917376, 0.9976507, -0.0040698, 0.0045059
9: -0.0139052, -0.0086452, -0.0137548, -0.0083873, -0.0040901, 0.0036943

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0019844
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0020287
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061497, 0.0086350, 0.0062836, 0.0087035, -0.0019255, 0.0017329
1: 0.0022107, 0.0025698, 0.0022301, 0.0025797, -0.0002782, 0.0002504
2: 0.0095858, 0.0109599, 0.0095479, 0.0108858, -0.0009581, 0.0010646
3: -0.0047664, -0.0033452, -0.0048055, -0.0034218, -0.0009909, 0.0011010
4: -0.0004155, 0.0011229, -0.0003326, 0.0011653, -0.0011919, 0.0010727
5: 0.0030507, 0.0045066, 0.0030106, 0.0044281, -0.0010151, 0.0011280
6: -0.0101962, -0.0044195, -0.0103552, -0.0047308, -0.0040278, 0.0044755
7: 0.0034622, 0.0113296, 0.0038863, 0.0115462, -0.0060952, 0.0054855
8: 0.9916527, 0.9971946, 0.9919515, 0.9973474, -0.0042936, 0.0038641
9: -0.0133408, -0.0083102, -0.0134793, -0.0085813, -0.0035076, 0.0038974

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0020552
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0021046
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061497, 0.0086350, 0.0061497, 0.0086350, -0.0017665, 0.0017665
1: 0.0022107, 0.0025698, 0.0022107, 0.0025698, -0.0002552, 0.0002552
2: 0.0095858, 0.0109599, 0.0095858, 0.0109599, -0.0009767, 0.0009767
3: -0.0047664, -0.0033452, -0.0047664, -0.0033452, -0.0010101, 0.0010101
4: -0.0004155, 0.0011229, -0.0004155, 0.0011229, -0.0010935, 0.0010935
5: 0.0030507, 0.0045066, 0.0030507, 0.0045066, -0.0010348, 0.0010348
6: -0.0101962, -0.0044195, -0.0101962, -0.0044195, -0.0041059, 0.0041059
7: 0.0034622, 0.0113296, 0.0034622, 0.0113296, -0.0055919, 0.0055919
8: 0.9916527, 0.9971946, 0.9916527, 0.9971946, -0.0039391, 0.0039391
9: -0.0133408, -0.0083102, -0.0133408, -0.0083102, -0.0035756, 0.0035756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0020552
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0021046
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061877, 0.0088396, 0.0062836, 0.0087035, -0.0019180, 0.0019728
1: 0.0022162, 0.0025994, 0.0022301, 0.0025797, -0.0002771, 0.0002850
2: 0.0094727, 0.0109388, 0.0095479, 0.0108858, -0.0010907, 0.0010604
3: -0.0048833, -0.0033670, -0.0048055, -0.0034218, -0.0011280, 0.0010967
4: -0.0003920, 0.0012495, -0.0003326, 0.0011653, -0.0011873, 0.0012212
5: 0.0029309, 0.0044843, 0.0030106, 0.0044281, -0.0011556, 0.0011235
6: -0.0106716, -0.0045080, -0.0103552, -0.0047308, -0.0045853, 0.0044579
7: 0.0035827, 0.0119770, 0.0038863, 0.0115462, -0.0060712, 0.0062447
8: 0.9917376, 0.9976507, 0.9919515, 0.9973474, -0.0042767, 0.0043989
9: -0.0137548, -0.0083873, -0.0134793, -0.0085813, -0.0039931, 0.0038821

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019837
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061877, 0.0088396, 0.0061497, 0.0086350, -0.0017669, 0.0020193
1: 0.0022162, 0.0025994, 0.0022107, 0.0025698, -0.0002553, 0.0002917
2: 0.0094727, 0.0109388, 0.0095858, 0.0109599, -0.0011164, 0.0009769
3: -0.0048833, -0.0033670, -0.0047664, -0.0033452, -0.0011546, 0.0010103
4: -0.0003920, 0.0012495, -0.0004155, 0.0011229, -0.0010937, 0.0012500
5: 0.0029309, 0.0044843, 0.0030507, 0.0045066, -0.0011829, 0.0010350
6: -0.0106716, -0.0045080, -0.0101962, -0.0044195, -0.0046934, 0.0041067
7: 0.0035827, 0.0119770, 0.0034622, 0.0113296, -0.0055930, 0.0063920
8: 0.9917376, 0.9976507, 0.9916527, 0.9971946, -0.0039398, 0.0045026
9: -0.0137548, -0.0083873, -0.0133408, -0.0083102, -0.0040872, 0.0035763

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019837
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061497, 0.0086350, 0.0063152, 0.0089138, -0.0021791, 0.0017313
1: 0.0022107, 0.0025698, 0.0022347, 0.0026101, -0.0003148, 0.0002501
2: 0.0095858, 0.0109599, 0.0094316, 0.0108684, -0.0009572, 0.0012048
3: -0.0047664, -0.0033452, -0.0049258, -0.0034399, -0.0009900, 0.0012461
4: -0.0004155, 0.0011229, -0.0003131, 0.0012955, -0.0013489, 0.0010717
5: 0.0030507, 0.0045066, 0.0028873, 0.0044096, -0.0010142, 0.0012765
6: -0.0101962, -0.0044195, -0.0108442, -0.0048042, -0.0040241, 0.0050649
7: 0.0034622, 0.0113296, 0.0039862, 0.0122122, -0.0068980, 0.0054804
8: 0.9916527, 0.9971946, 0.9920218, 0.9978164, -0.0048591, 0.0038605
9: -0.0133408, -0.0083102, -0.0139052, -0.0086452, -0.0035043, 0.0044108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0020625
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0021108
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061497, 0.0086350, 0.0061877, 0.0088396, -0.0020193, 0.0017669
1: 0.0022107, 0.0025698, 0.0022162, 0.0025994, -0.0002917, 0.0002553
2: 0.0095858, 0.0109599, 0.0094727, 0.0109388, -0.0009769, 0.0011164
3: -0.0047664, -0.0033452, -0.0048833, -0.0033670, -0.0010103, 0.0011546
4: -0.0004155, 0.0011229, -0.0003920, 0.0012495, -0.0012500, 0.0010937
5: 0.0030507, 0.0045066, 0.0029309, 0.0044843, -0.0010350, 0.0011829
6: -0.0101962, -0.0044195, -0.0106716, -0.0045080, -0.0041067, 0.0046934
7: 0.0034622, 0.0113296, 0.0035827, 0.0119770, -0.0063920, 0.0055930
8: 0.9916527, 0.9971946, 0.9917376, 0.9976507, -0.0045026, 0.0039398
9: -0.0133408, -0.0083102, -0.0137548, -0.0083873, -0.0035763, 0.0040872

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0020625
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0021108
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061877, 0.0088396, 0.0063152, 0.0089138, -0.0020207, 0.0018252
1: 0.0022162, 0.0025994, 0.0022347, 0.0026101, -0.0002919, 0.0002637
2: 0.0094727, 0.0109388, 0.0094316, 0.0108684, -0.0010091, 0.0011172
3: -0.0048833, -0.0033670, -0.0049258, -0.0034399, -0.0010436, 0.0011555
4: -0.0003920, 0.0012495, -0.0003131, 0.0012955, -0.0012509, 0.0011298
5: 0.0029309, 0.0044843, 0.0028873, 0.0044096, -0.0010692, 0.0011837
6: -0.0106716, -0.0045080, -0.0108442, -0.0048042, -0.0042422, 0.0046968
7: 0.0035827, 0.0119770, 0.0039862, 0.0122122, -0.0063966, 0.0057775
8: 0.9917376, 0.9976507, 0.9920218, 0.9978164, -0.0045059, 0.0040698
9: -0.0137548, -0.0083873, -0.0139052, -0.0086452, -0.0036943, 0.0040901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019837
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061877, 0.0088396, 0.0061877, 0.0088396, -0.0018605, 0.0018605
1: 0.0022162, 0.0025994, 0.0022162, 0.0025994, -0.0002688, 0.0002688
2: 0.0094727, 0.0109388, 0.0094727, 0.0109388, -0.0010286, 0.0010286
3: -0.0048833, -0.0033670, -0.0048833, -0.0033670, -0.0010639, 0.0010639
4: -0.0003920, 0.0012495, -0.0003920, 0.0012495, -0.0011517, 0.0011517
5: 0.0029309, 0.0044843, 0.0029309, 0.0044843, -0.0010899, 0.0010899
6: -0.0106716, -0.0045080, -0.0106716, -0.0045080, -0.0043243, 0.0043243
7: 0.0035827, 0.0119770, 0.0035827, 0.0119770, -0.0058893, 0.0058893
8: 0.9917376, 0.9976507, 0.9917376, 0.9976507, -0.0041486, 0.0041486
9: -0.0137548, -0.0083873, -0.0137548, -0.0083873, -0.0037658, 0.0037658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019836
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259
time: 0.79 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 3.31 seconds
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021349, upper bound: 0.0020364
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0020932
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021349, upper bound: 0.0020866
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021350, upper bound: 0.0021475
IS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020057
IS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020562
IS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020057
IS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0022369, upper bound: 0.0020562
IS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0020119
IS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0020645
IS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0020590
IS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020949, upper bound: 0.0021161
IS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0019844
IS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0020287
IS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0019844
IS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021920, upper bound: 0.0020287
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0020552
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0021046
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0020552
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021627, upper bound: 0.0021046
IS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019837
IS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259
IS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019837
IS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259
IS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0020625
IS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0021108
IS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0020625
IS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0020888, upper bound: 0.0021108
IS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019837
IS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259
IS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0019836
IS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.31
Output dim: 8, lower bound: -0.0021874, upper bound: 0.0020259

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0062883, 0.0086844, -0.0016779, 0.0016663
1: 0.0022345, 0.0025704, 0.0022308, 0.0025769, -0.0002424, 0.0002407
2: 0.0095835, 0.0108688, 0.0095585, 0.0108832, -0.0009212, 0.0009277
3: -0.0047687, -0.0034394, -0.0047946, -0.0034245, -0.0009528, 0.0009594
4: -0.0003136, 0.0011255, -0.0003297, 0.0011535, -0.0010386, 0.0010315
5: 0.0030483, 0.0044101, 0.0030218, 0.0044254, -0.0009761, 0.0009829
6: -0.0102056, -0.0048024, -0.0103108, -0.0047418, -0.0038729, 0.0038999
7: 0.0039837, 0.0113425, 0.0039012, 0.0114858, -0.0053113, 0.0052746
8: 0.9920201, 0.9972038, 0.9919620, 0.9973047, -0.0037414, 0.0037155
9: -0.0133490, -0.0086437, -0.0134407, -0.0085909, -0.0033727, 0.0033962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0020521
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0020521
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0062847, 0.0086959, -0.0017276, 0.0016730
1: 0.0022308, 0.0025750, 0.0022303, 0.0025786, -0.0002496, 0.0002417
2: 0.0095658, 0.0108831, 0.0095521, 0.0108852, -0.0009250, 0.0009551
3: -0.0047871, -0.0034247, -0.0048012, -0.0034225, -0.0009567, 0.0009878
4: -0.0003295, 0.0011453, -0.0003319, 0.0011606, -0.0010694, 0.0010356
5: 0.0030295, 0.0044252, 0.0030150, 0.0044275, -0.0009801, 0.0010120
6: -0.0102803, -0.0047424, -0.0103376, -0.0047334, -0.0038886, 0.0040153
7: 0.0039021, 0.0114441, 0.0038898, 0.0115222, -0.0054685, 0.0052960
8: 0.9919626, 0.9972753, 0.9919540, 0.9973303, -0.0038522, 0.0037306
9: -0.0134140, -0.0085915, -0.0134640, -0.0085836, -0.0033864, 0.0034967

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0021081
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0021081
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0063204, 0.0088935, -0.0019312, 0.0016640
1: 0.0022345, 0.0025704, 0.0022354, 0.0026072, -0.0002790, 0.0002404
2: 0.0095835, 0.0108688, 0.0094429, 0.0108655, -0.0009200, 0.0010677
3: -0.0047687, -0.0034394, -0.0049142, -0.0034429, -0.0009515, 0.0011043
4: -0.0003136, 0.0011255, -0.0003098, 0.0012830, -0.0011954, 0.0010300
5: 0.0030483, 0.0044101, 0.0028992, 0.0044066, -0.0009748, 0.0011313
6: -0.0102056, -0.0048024, -0.0107970, -0.0048164, -0.0038676, 0.0044886
7: 0.0039837, 0.0113425, 0.0040028, 0.0121479, -0.0061131, 0.0052673
8: 0.9920201, 0.9972038, 0.9920335, 0.9977711, -0.0043062, 0.0037104
9: -0.0133490, -0.0086437, -0.0138641, -0.0086558, -0.0033680, 0.0039089

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0020866
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0020866
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0063164, 0.0089072, -0.0019816, 0.0016719
1: 0.0022308, 0.0025750, 0.0022348, 0.0026091, -0.0002863, 0.0002415
2: 0.0095658, 0.0108831, 0.0094353, 0.0108677, -0.0009243, 0.0010956
3: -0.0047871, -0.0034247, -0.0049220, -0.0034406, -0.0009560, 0.0011331
4: -0.0003295, 0.0011453, -0.0003123, 0.0012914, -0.0012266, 0.0010349
5: 0.0030295, 0.0044252, 0.0028912, 0.0044089, -0.0009794, 0.0011608
6: -0.0102803, -0.0047424, -0.0108288, -0.0048070, -0.0038859, 0.0046057
7: 0.0039021, 0.0114441, 0.0039900, 0.0121911, -0.0062726, 0.0052923
8: 0.9919626, 0.9972753, 0.9920246, 0.9978016, -0.0044186, 0.0037280
9: -0.0134140, -0.0085915, -0.0138917, -0.0086477, -0.0033840, 0.0040109

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0021475
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0021475
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0062883, 0.0086844, -0.0016731, 0.0019175
1: 0.0022409, 0.0026003, 0.0022308, 0.0025769, -0.0002417, 0.0002770
2: 0.0094692, 0.0108446, 0.0095585, 0.0108832, -0.0010601, 0.0009250
3: -0.0048870, -0.0034644, -0.0047946, -0.0034245, -0.0010965, 0.0009567
4: -0.0002865, 0.0012535, -0.0003297, 0.0011535, -0.0010357, 0.0011870
5: 0.0029272, 0.0043845, 0.0030218, 0.0044254, -0.0011233, 0.0009801
6: -0.0106862, -0.0049040, -0.0103108, -0.0047418, -0.0044568, 0.0038887
7: 0.0041221, 0.0119970, 0.0039012, 0.0114858, -0.0052960, 0.0060698
8: 0.9921175, 0.9976648, 0.9919620, 0.9973047, -0.0037306, 0.0042757
9: -0.0137675, -0.0087321, -0.0134407, -0.0085909, -0.0038812, 0.0033864

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
time: 0.63 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0062847, 0.0086959, -0.0017254, 0.0019264
1: 0.0022354, 0.0026056, 0.0022303, 0.0025786, -0.0002493, 0.0002783
2: 0.0094488, 0.0108654, 0.0095521, 0.0108852, -0.0010650, 0.0009539
3: -0.0049081, -0.0034429, -0.0048012, -0.0034225, -0.0011015, 0.0009866
4: -0.0003098, 0.0012764, -0.0003319, 0.0011606, -0.0010680, 0.0011925
5: 0.0029055, 0.0044065, 0.0030150, 0.0044275, -0.0011285, 0.0010107
6: -0.0107722, -0.0048166, -0.0103376, -0.0047334, -0.0044774, 0.0040102
7: 0.0040031, 0.0121141, 0.0038898, 0.0115222, -0.0054616, 0.0060979
8: 0.9920337, 0.9977473, 0.9919540, 0.9973303, -0.0038472, 0.0042955
9: -0.0138425, -0.0086560, -0.0134640, -0.0085836, -0.0038991, 0.0034923

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0063204, 0.0088935, -0.0017655, 0.0017554
1: 0.0022409, 0.0026003, 0.0022354, 0.0026072, -0.0002551, 0.0002536
2: 0.0094692, 0.0108446, 0.0094429, 0.0108655, -0.0009705, 0.0009761
3: -0.0048870, -0.0034644, -0.0049142, -0.0034429, -0.0010038, 0.0010095
4: -0.0002865, 0.0012535, -0.0003098, 0.0012830, -0.0010929, 0.0010866
5: 0.0029272, 0.0043845, 0.0028992, 0.0044066, -0.0010283, 0.0010342
6: -0.0106862, -0.0049040, -0.0107970, -0.0048164, -0.0040800, 0.0041035
7: 0.0041221, 0.0119970, 0.0040028, 0.0121479, -0.0055886, 0.0055566
8: 0.9921175, 0.9976648, 0.9920335, 0.9977711, -0.0039367, 0.0039142
9: -0.0137675, -0.0087321, -0.0138641, -0.0086558, -0.0035531, 0.0035735

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0063164, 0.0089072, -0.0018173, 0.0017647
1: 0.0022354, 0.0026056, 0.0022348, 0.0026091, -0.0002626, 0.0002550
2: 0.0094488, 0.0108654, 0.0094353, 0.0108677, -0.0009757, 0.0010048
3: -0.0049081, -0.0034429, -0.0049220, -0.0034406, -0.0010091, 0.0010392
4: -0.0003098, 0.0012764, -0.0003123, 0.0012914, -0.0011250, 0.0010924
5: 0.0029055, 0.0044065, 0.0028912, 0.0044089, -0.0010338, 0.0010646
6: -0.0107722, -0.0048166, -0.0108288, -0.0048070, -0.0041017, 0.0042240
7: 0.0040031, 0.0121141, 0.0039900, 0.0121911, -0.0057527, 0.0055862
8: 0.9920337, 0.9977473, 0.9920246, 0.9978016, -0.0040523, 0.0039350
9: -0.0138425, -0.0086560, -0.0138917, -0.0086477, -0.0035720, 0.0036784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.88 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0061547, 0.0086158, -0.0016756, 0.0018533
1: 0.0022345, 0.0025704, 0.0022115, 0.0025670, -0.0002421, 0.0002677
2: 0.0095835, 0.0108688, 0.0095964, 0.0109571, -0.0010246, 0.0009264
3: -0.0047687, -0.0034394, -0.0047554, -0.0033481, -0.0010597, 0.0009581
4: -0.0003136, 0.0011255, -0.0004124, 0.0011111, -0.0010373, 0.0011472
5: 0.0030483, 0.0044101, 0.0030619, 0.0045036, -0.0010857, 0.0009816
6: -0.0102056, -0.0048024, -0.0101515, -0.0044312, -0.0043075, 0.0038947
7: 0.0039837, 0.0113425, 0.0034782, 0.0112688, -0.0053042, 0.0058665
8: 0.9920201, 0.9972038, 0.9916640, 0.9971519, -0.0037364, 0.0041325
9: -0.0133490, -0.0086437, -0.0133019, -0.0083204, -0.0037512, 0.0033916

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021399, upper bound: 0.0020324
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021399, upper bound: 0.0020324
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0061508, 0.0086272, -0.0017206, 0.0018724
1: 0.0022308, 0.0025750, 0.0022109, 0.0025687, -0.0002486, 0.0002705
2: 0.0095658, 0.0108831, 0.0095901, 0.0109592, -0.0010352, 0.0009513
3: -0.0047871, -0.0034247, -0.0047619, -0.0033459, -0.0010706, 0.0009838
4: -0.0003295, 0.0011453, -0.0004148, 0.0011181, -0.0010651, 0.0011590
5: 0.0030295, 0.0044252, 0.0030552, 0.0045059, -0.0010968, 0.0010079
6: -0.0102803, -0.0047424, -0.0101781, -0.0044222, -0.0043519, 0.0039991
7: 0.0039021, 0.0114441, 0.0034659, 0.0113050, -0.0054464, 0.0059270
8: 0.9919626, 0.9972753, 0.9916553, 0.9971773, -0.0038366, 0.0041751
9: -0.0134140, -0.0085915, -0.0133250, -0.0083125, -0.0037899, 0.0034826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021398, upper bound: 0.0020871
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021398, upper bound: 0.0020871
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0061931, 0.0088210, -0.0019156, 0.0018454
1: 0.0022345, 0.0025704, 0.0022170, 0.0025967, -0.0002767, 0.0002666
2: 0.0095835, 0.0108688, 0.0094830, 0.0109359, -0.0010203, 0.0010591
3: -0.0047687, -0.0034394, -0.0048727, -0.0033701, -0.0010552, 0.0010953
4: -0.0003136, 0.0011255, -0.0003887, 0.0012381, -0.0011858, 0.0011423
5: 0.0030483, 0.0044101, 0.0029417, 0.0044812, -0.0010810, 0.0011221
6: -0.0102056, -0.0048024, -0.0106285, -0.0045204, -0.0042891, 0.0044523
7: 0.0039837, 0.0113425, 0.0035997, 0.0119183, -0.0060637, 0.0058414
8: 0.9920201, 0.9972038, 0.9917496, 0.9976094, -0.0042714, 0.0041148
9: -0.0133490, -0.0086437, -0.0137172, -0.0083981, -0.0037352, 0.0038773

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0020590
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0020590
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0061890, 0.0088318, -0.0019608, 0.0018590
1: 0.0022308, 0.0025750, 0.0022164, 0.0025982, -0.0002833, 0.0002686
2: 0.0095658, 0.0108831, 0.0094770, 0.0109381, -0.0010278, 0.0010841
3: -0.0047871, -0.0034247, -0.0048789, -0.0033677, -0.0010630, 0.0011212
4: -0.0003295, 0.0011453, -0.0003912, 0.0012447, -0.0012138, 0.0011507
5: 0.0030295, 0.0044252, 0.0029354, 0.0044835, -0.0010890, 0.0011486
6: -0.0102803, -0.0047424, -0.0106535, -0.0045109, -0.0043207, 0.0045574
7: 0.0039021, 0.0114441, 0.0035868, 0.0119525, -0.0062068, 0.0058845
8: 0.9919626, 0.9972753, 0.9917405, 0.9976334, -0.0043722, 0.0041451
9: -0.0134140, -0.0085915, -0.0137391, -0.0083898, -0.0037627, 0.0039688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0021161
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0021161
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0061547, 0.0086158, -0.0016708, 0.0021045
1: 0.0022409, 0.0026003, 0.0022115, 0.0025670, -0.0002414, 0.0003040
2: 0.0094692, 0.0108446, 0.0095964, 0.0109571, -0.0011635, 0.0009238
3: -0.0048870, -0.0034644, -0.0047554, -0.0033481, -0.0012034, 0.0009554
4: -0.0002865, 0.0012535, -0.0004124, 0.0011111, -0.0010343, 0.0013027
5: 0.0029272, 0.0043845, 0.0030619, 0.0045036, -0.0012328, 0.0009788
6: -0.0106862, -0.0049040, -0.0101515, -0.0044312, -0.0048915, 0.0038835
7: 0.0041221, 0.0119970, 0.0034782, 0.0112688, -0.0052890, 0.0066617
8: 0.9921175, 0.9976648, 0.9916640, 0.9971519, -0.0037256, 0.0046927
9: -0.0137675, -0.0087321, -0.0133019, -0.0083204, -0.0042597, 0.0033819

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0061508, 0.0086272, -0.0017184, 0.0021257
1: 0.0022354, 0.0026056, 0.0022109, 0.0025687, -0.0002483, 0.0003071
2: 0.0094488, 0.0108654, 0.0095901, 0.0109592, -0.0011752, 0.0009501
3: -0.0049081, -0.0034429, -0.0047619, -0.0033459, -0.0012155, 0.0009826
4: -0.0003098, 0.0012764, -0.0004148, 0.0011181, -0.0010637, 0.0013158
5: 0.0029055, 0.0044065, 0.0030552, 0.0045059, -0.0012452, 0.0010066
6: -0.0107722, -0.0048166, -0.0101781, -0.0044222, -0.0049407, 0.0039940
7: 0.0040031, 0.0121141, 0.0034659, 0.0113050, -0.0054395, 0.0067289
8: 0.9920337, 0.9977473, 0.9916553, 0.9971773, -0.0038317, 0.0047399
9: -0.0138425, -0.0086560, -0.0133250, -0.0083125, -0.0043026, 0.0034782

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0061931, 0.0088210, -0.0017661, 0.0019480
1: 0.0022409, 0.0026003, 0.0022170, 0.0025967, -0.0002551, 0.0002814
2: 0.0094692, 0.0108446, 0.0094830, 0.0109359, -0.0010770, 0.0009764
3: -0.0048870, -0.0034644, -0.0048727, -0.0033701, -0.0011139, 0.0010099
4: -0.0002865, 0.0012535, -0.0003887, 0.0012381, -0.0010932, 0.0012058
5: 0.0029272, 0.0043845, 0.0029417, 0.0044812, -0.0011411, 0.0010346
6: -0.0106862, -0.0049040, -0.0106285, -0.0045204, -0.0045276, 0.0041049
7: 0.0041221, 0.0119970, 0.0035997, 0.0119183, -0.0055905, 0.0061662
8: 0.9921175, 0.9976648, 0.9917496, 0.9976094, -0.0039381, 0.0043436
9: -0.0137675, -0.0087321, -0.0137172, -0.0083981, -0.0039428, 0.0035747

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0061890, 0.0088318, -0.0018125, 0.0019691
1: 0.0022354, 0.0026056, 0.0022164, 0.0025982, -0.0002619, 0.0002845
2: 0.0094488, 0.0108654, 0.0094770, 0.0109381, -0.0010887, 0.0010021
3: -0.0049081, -0.0034429, -0.0048789, -0.0033677, -0.0011259, 0.0010364
4: -0.0003098, 0.0012764, -0.0003912, 0.0012447, -0.0011220, 0.0012189
5: 0.0029055, 0.0044065, 0.0029354, 0.0044835, -0.0011535, 0.0010618
6: -0.0107722, -0.0048166, -0.0106535, -0.0045109, -0.0045767, 0.0042128
7: 0.0040031, 0.0121141, 0.0035868, 0.0119525, -0.0057375, 0.0062331
8: 0.9920337, 0.9977473, 0.9917405, 0.9976334, -0.0040416, 0.0043907
9: -0.0138425, -0.0086560, -0.0137391, -0.0083898, -0.0039856, 0.0036687

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0062883, 0.0086844, -0.0018865, 0.0016699
1: 0.0022144, 0.0025606, 0.0022308, 0.0025769, -0.0002726, 0.0002413
2: 0.0096211, 0.0109460, 0.0095585, 0.0108832, -0.0009232, 0.0010430
3: -0.0047298, -0.0033595, -0.0047946, -0.0034245, -0.0009549, 0.0010787
4: -0.0004001, 0.0010834, -0.0003297, 0.0011535, -0.0011678, 0.0010337
5: 0.0030881, 0.0044919, 0.0030218, 0.0044254, -0.0009782, 0.0011051
6: -0.0100476, -0.0044776, -0.0103108, -0.0047418, -0.0038813, 0.0043848
7: 0.0035415, 0.0111272, 0.0039012, 0.0114858, -0.0059718, 0.0052860
8: 0.9917086, 0.9970521, 0.9919620, 0.9973047, -0.0042066, 0.0037236
9: -0.0132114, -0.0083608, -0.0134407, -0.0085909, -0.0033800, 0.0038185

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0020552
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0020552
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0062847, 0.0086959, -0.0019144, 0.0016731
1: 0.0022115, 0.0025647, 0.0022303, 0.0025786, -0.0002766, 0.0002417
2: 0.0096053, 0.0109570, 0.0095521, 0.0108852, -0.0009250, 0.0010584
3: -0.0047463, -0.0033482, -0.0048012, -0.0034225, -0.0009567, 0.0010947
4: -0.0004123, 0.0011011, -0.0003319, 0.0011606, -0.0011851, 0.0010357
5: 0.0030713, 0.0045035, 0.0030150, 0.0044275, -0.0009801, 0.0011215
6: -0.0101143, -0.0044317, -0.0103376, -0.0047334, -0.0038887, 0.0044496
7: 0.0034788, 0.0112181, 0.0038898, 0.0115222, -0.0060600, 0.0052961
8: 0.9916644, 0.9971162, 0.9919540, 0.9973303, -0.0042688, 0.0037307
9: -0.0132695, -0.0083208, -0.0134640, -0.0085836, -0.0033865, 0.0038750

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0021046
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0021046
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0061547, 0.0086158, -0.0017105, 0.0016963
1: 0.0022144, 0.0025606, 0.0022115, 0.0025670, -0.0002471, 0.0002451
2: 0.0096211, 0.0109460, 0.0095964, 0.0109571, -0.0009378, 0.0009457
3: -0.0047298, -0.0033595, -0.0047554, -0.0033481, -0.0009699, 0.0009781
4: -0.0004001, 0.0010834, -0.0004124, 0.0011111, -0.0010588, 0.0010500
5: 0.0030881, 0.0044919, 0.0030619, 0.0045036, -0.0009937, 0.0010020
6: -0.0100476, -0.0044776, -0.0101515, -0.0044312, -0.0039426, 0.0039757
7: 0.0035415, 0.0111272, 0.0034782, 0.0112688, -0.0054145, 0.0053695
8: 0.9917086, 0.9970521, 0.9916640, 0.9971519, -0.0038141, 0.0037824
9: -0.0132114, -0.0083608, -0.0133019, -0.0083204, -0.0034334, 0.0034622

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0020552
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0020552
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0061508, 0.0086272, -0.0017559, 0.0017046
1: 0.0022115, 0.0025647, 0.0022109, 0.0025687, -0.0002537, 0.0002463
2: 0.0096053, 0.0109570, 0.0095901, 0.0109592, -0.0009424, 0.0009708
3: -0.0047463, -0.0033482, -0.0047619, -0.0033459, -0.0009747, 0.0010040
4: -0.0004123, 0.0011011, -0.0004148, 0.0011181, -0.0010869, 0.0010552
5: 0.0030713, 0.0045035, 0.0030552, 0.0045059, -0.0009985, 0.0010286
6: -0.0101143, -0.0044317, -0.0101781, -0.0044222, -0.0039619, 0.0040811
7: 0.0034788, 0.0112181, 0.0034659, 0.0113050, -0.0055581, 0.0053958
8: 0.9916644, 0.9971162, 0.9916553, 0.9971773, -0.0039152, 0.0038009
9: -0.0132695, -0.0083208, -0.0133250, -0.0083125, -0.0034502, 0.0035540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0021046
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0021046
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0062883, 0.0086844, -0.0018772, 0.0019118
1: 0.0022200, 0.0025906, 0.0022308, 0.0025769, -0.0002712, 0.0002762
2: 0.0095063, 0.0109246, 0.0095585, 0.0108832, -0.0010570, 0.0010378
3: -0.0048486, -0.0033817, -0.0047946, -0.0034245, -0.0010932, 0.0010734
4: -0.0003761, 0.0012119, -0.0003297, 0.0011535, -0.0011620, 0.0011834
5: 0.0029664, 0.0044693, 0.0030218, 0.0044254, -0.0011199, 0.0010996
6: -0.0105304, -0.0045676, -0.0103108, -0.0047418, -0.0044435, 0.0043631
7: 0.0036640, 0.0117847, 0.0039012, 0.0114858, -0.0059421, 0.0060517
8: 0.9917949, 0.9975153, 0.9919620, 0.9973047, -0.0041858, 0.0042630
9: -0.0136318, -0.0084392, -0.0134407, -0.0085909, -0.0038696, 0.0037996

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019836
time: 0.83 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019836
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0062847, 0.0086959, -0.0019064, 0.0019143
1: 0.0022171, 0.0025942, 0.0022303, 0.0025786, -0.0002754, 0.0002766
2: 0.0094926, 0.0109357, 0.0095521, 0.0108852, -0.0010584, 0.0010540
3: -0.0048628, -0.0033702, -0.0048012, -0.0034225, -0.0010946, 0.0010901
4: -0.0003885, 0.0012273, -0.0003319, 0.0011606, -0.0011801, 0.0011850
5: 0.0029520, 0.0044810, 0.0030150, 0.0044275, -0.0011214, 0.0011168
6: -0.0105878, -0.0045210, -0.0103376, -0.0047334, -0.0044493, 0.0044310
7: 0.0036006, 0.0118630, 0.0038898, 0.0115222, -0.0060347, 0.0060596
8: 0.9917502, 0.9975703, 0.9919540, 0.9973303, -0.0042510, 0.0042685
9: -0.0136819, -0.0083986, -0.0134640, -0.0085836, -0.0038747, 0.0038587

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0061547, 0.0086158, -0.0017060, 0.0019463
1: 0.0022200, 0.0025906, 0.0022115, 0.0025670, -0.0002465, 0.0002812
2: 0.0095063, 0.0109246, 0.0095964, 0.0109571, -0.0010761, 0.0009432
3: -0.0048486, -0.0033817, -0.0047554, -0.0033481, -0.0011129, 0.0009755
4: -0.0003761, 0.0012119, -0.0004124, 0.0011111, -0.0010561, 0.0012048
5: 0.0029664, 0.0044693, 0.0030619, 0.0045036, -0.0011401, 0.0009994
6: -0.0105304, -0.0045676, -0.0101515, -0.0044312, -0.0045237, 0.0039652
7: 0.0036640, 0.0117847, 0.0034782, 0.0112688, -0.0054003, 0.0061609
8: 0.9917949, 0.9975153, 0.9916640, 0.9971519, -0.0038041, 0.0043399
9: -0.0136318, -0.0084392, -0.0133019, -0.0083204, -0.0039395, 0.0034531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0061508, 0.0086272, -0.0017553, 0.0019574
1: 0.0022171, 0.0025942, 0.0022109, 0.0025687, -0.0002536, 0.0002828
2: 0.0094926, 0.0109357, 0.0095901, 0.0109592, -0.0010822, 0.0009705
3: -0.0048628, -0.0033702, -0.0047619, -0.0033459, -0.0011193, 0.0010037
4: -0.0003885, 0.0012273, -0.0004148, 0.0011181, -0.0010866, 0.0012117
5: 0.0029520, 0.0044810, 0.0030552, 0.0045059, -0.0011466, 0.0010283
6: -0.0105878, -0.0045210, -0.0101781, -0.0044222, -0.0045496, 0.0040798
7: 0.0036006, 0.0118630, 0.0034659, 0.0113050, -0.0055563, 0.0061961
8: 0.9917502, 0.9975703, 0.9916553, 0.9971773, -0.0039140, 0.0043647
9: -0.0136819, -0.0083986, -0.0133250, -0.0083125, -0.0039620, 0.0035529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0063204, 0.0088935, -0.0021398, 0.0016676
1: 0.0022144, 0.0025606, 0.0022354, 0.0026072, -0.0003091, 0.0002409
2: 0.0096211, 0.0109460, 0.0094429, 0.0108655, -0.0009220, 0.0011831
3: -0.0047298, -0.0033595, -0.0049142, -0.0034429, -0.0009535, 0.0012236
4: -0.0004001, 0.0010834, -0.0003098, 0.0012830, -0.0013246, 0.0010323
5: 0.0030881, 0.0044919, 0.0028992, 0.0044066, -0.0009769, 0.0012535
6: -0.0100476, -0.0044776, -0.0107970, -0.0048164, -0.0038759, 0.0049736
7: 0.0035415, 0.0111272, 0.0040028, 0.0121479, -0.0067736, 0.0052787
8: 0.9917086, 0.9970521, 0.9920335, 0.9977711, -0.0047715, 0.0037184
9: -0.0132114, -0.0083608, -0.0138641, -0.0086558, -0.0033753, 0.0043312

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0020625
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0020625
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0063164, 0.0089072, -0.0021684, 0.0016719
1: 0.0022115, 0.0025647, 0.0022348, 0.0026091, -0.0003133, 0.0002415
2: 0.0096053, 0.0109570, 0.0094353, 0.0108677, -0.0009244, 0.0011989
3: -0.0047463, -0.0033482, -0.0049220, -0.0034406, -0.0009560, 0.0012399
4: -0.0004123, 0.0011011, -0.0003123, 0.0012914, -0.0013423, 0.0010350
5: 0.0030713, 0.0045035, 0.0028912, 0.0044089, -0.0009794, 0.0012703
6: -0.0101143, -0.0044317, -0.0108288, -0.0048070, -0.0038860, 0.0050401
7: 0.0034788, 0.0112181, 0.0039900, 0.0121911, -0.0068641, 0.0052924
8: 0.9916644, 0.9971162, 0.9920246, 0.9978016, -0.0048352, 0.0037281
9: -0.0132695, -0.0083208, -0.0138917, -0.0086477, -0.0033841, 0.0043891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.86 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0021108
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0021108
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0061931, 0.0088210, -0.0019634, 0.0016958
1: 0.0022144, 0.0025606, 0.0022170, 0.0025967, -0.0002836, 0.0002450
2: 0.0096211, 0.0109460, 0.0094830, 0.0109359, -0.0009376, 0.0010855
3: -0.0047298, -0.0033595, -0.0048727, -0.0033701, -0.0009697, 0.0011227
4: -0.0004001, 0.0010834, -0.0003887, 0.0012381, -0.0012154, 0.0010497
5: 0.0030881, 0.0044919, 0.0029417, 0.0044812, -0.0009934, 0.0011501
6: -0.0100476, -0.0044776, -0.0106285, -0.0045204, -0.0039415, 0.0045634
7: 0.0035415, 0.0111272, 0.0035997, 0.0119183, -0.0062150, 0.0053680
8: 0.9917086, 0.9970521, 0.9917496, 0.9976094, -0.0043779, 0.0037813
9: -0.0132114, -0.0083608, -0.0137172, -0.0083981, -0.0034324, 0.0039740

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0020625
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0020625
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0061890, 0.0088318, -0.0020088, 0.0017051
1: 0.0022115, 0.0025647, 0.0022164, 0.0025982, -0.0002902, 0.0002463
2: 0.0096053, 0.0109570, 0.0094770, 0.0109381, -0.0009427, 0.0011106
3: -0.0047463, -0.0033482, -0.0048789, -0.0033677, -0.0009750, 0.0011487
4: -0.0004123, 0.0011011, -0.0003912, 0.0012447, -0.0012435, 0.0010555
5: 0.0030713, 0.0045035, 0.0029354, 0.0044835, -0.0009988, 0.0011768
6: -0.0101143, -0.0044317, -0.0106535, -0.0045109, -0.0039630, 0.0046691
7: 0.0034788, 0.0112181, 0.0035868, 0.0119525, -0.0063589, 0.0053973
8: 0.9916644, 0.9971162, 0.9917405, 0.9976334, -0.0044794, 0.0038020
9: -0.0132695, -0.0083208, -0.0137391, -0.0083898, -0.0034512, 0.0040661

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0021108
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0021108
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0063204, 0.0088935, -0.0019796, 0.0017632
1: 0.0022200, 0.0025906, 0.0022354, 0.0026072, -0.0002860, 0.0002547
2: 0.0095063, 0.0109246, 0.0094429, 0.0108655, -0.0009748, 0.0010945
3: -0.0048486, -0.0033817, -0.0049142, -0.0034429, -0.0010082, 0.0011320
4: -0.0003761, 0.0012119, -0.0003098, 0.0012830, -0.0012254, 0.0010914
5: 0.0029664, 0.0044693, 0.0028992, 0.0044066, -0.0010329, 0.0011597
6: -0.0105304, -0.0045676, -0.0107970, -0.0048164, -0.0040980, 0.0046012
7: 0.0036640, 0.0117847, 0.0040028, 0.0121479, -0.0062665, 0.0055812
8: 0.9917949, 0.9975153, 0.9920335, 0.9977711, -0.0044142, 0.0039315
9: -0.0136318, -0.0084392, -0.0138641, -0.0086558, -0.0035688, 0.0040070

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019837
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019837
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0063164, 0.0089072, -0.0020096, 0.0017677
1: 0.0022171, 0.0025942, 0.0022348, 0.0026091, -0.0002903, 0.0002554
2: 0.0094926, 0.0109357, 0.0094353, 0.0108677, -0.0009773, 0.0011111
3: -0.0048628, -0.0033702, -0.0049220, -0.0034406, -0.0010108, 0.0011491
4: -0.0003885, 0.0012273, -0.0003123, 0.0012914, -0.0012440, 0.0010942
5: 0.0029520, 0.0044810, 0.0028912, 0.0044089, -0.0010355, 0.0011772
6: -0.0105878, -0.0045210, -0.0108288, -0.0048070, -0.0041086, 0.0046709
7: 0.0036006, 0.0118630, 0.0039900, 0.0121911, -0.0063614, 0.0055955
8: 0.9917502, 0.9975703, 0.9920246, 0.9978016, -0.0044811, 0.0039416
9: -0.0136819, -0.0083986, -0.0138917, -0.0086477, -0.0035779, 0.0040677

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0061931, 0.0088210, -0.0018028, 0.0017893
1: 0.0022200, 0.0025906, 0.0022170, 0.0025967, -0.0002605, 0.0002585
2: 0.0095063, 0.0109246, 0.0094830, 0.0109359, -0.0009892, 0.0009967
3: -0.0048486, -0.0033817, -0.0048727, -0.0033701, -0.0010231, 0.0010309
4: -0.0003761, 0.0012119, -0.0003887, 0.0012381, -0.0011160, 0.0011076
5: 0.0029664, 0.0044693, 0.0029417, 0.0044812, -0.0010482, 0.0010561
6: -0.0105304, -0.0045676, -0.0106285, -0.0045204, -0.0041588, 0.0041903
7: 0.0036640, 0.0117847, 0.0035997, 0.0119183, -0.0057068, 0.0056639
8: 0.9917949, 0.9975153, 0.9917496, 0.9976094, -0.0040200, 0.0039898
9: -0.0136318, -0.0084392, -0.0137172, -0.0083981, -0.0036216, 0.0036491

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0061890, 0.0088318, -0.0018492, 0.0017976
1: 0.0022171, 0.0025942, 0.0022164, 0.0025982, -0.0002672, 0.0002597
2: 0.0094926, 0.0109357, 0.0094770, 0.0109381, -0.0009938, 0.0010224
3: -0.0048628, -0.0033702, -0.0048789, -0.0033677, -0.0010279, 0.0010574
4: -0.0003885, 0.0012273, -0.0003912, 0.0012447, -0.0011447, 0.0011127
5: 0.0029520, 0.0044810, 0.0029354, 0.0044835, -0.0010530, 0.0010832
6: -0.0105878, -0.0045210, -0.0106535, -0.0045109, -0.0041780, 0.0042980
7: 0.0036006, 0.0118630, 0.0035868, 0.0119525, -0.0058535, 0.0056901
8: 0.9917502, 0.9975703, 0.9917405, 0.9976334, -0.0041233, 0.0040082
9: -0.0136819, -0.0083986, -0.0137391, -0.0083898, -0.0036384, 0.0037429

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259
time: 0.78 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.30 seconds
IS_A2_B2_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0020521
IS_A2_B2_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0020521
IS_A2_B2_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0021081
IS_A2_B2_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021481, upper bound: 0.0021081
IS_A2_B2_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0020866
IS_A2_B2_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0020866
IS_A2_B2_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0021475
IS_A2_B2_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020907, upper bound: 0.0021475
IS_A2_B2_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
IS_A2_B2_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
IS_A2_B2_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
IS_A2_B2_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
IS_A2_B2_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
IS_A2_B2_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020057
IS_A2_B2_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
IS_A2_B2_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021873, upper bound: 0.0020562
IS_A2_B2_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021399, upper bound: 0.0020324
IS_A2_B2_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021399, upper bound: 0.0020324
IS_A2_B2_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021398, upper bound: 0.0020871
IS_A2_B2_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021398, upper bound: 0.0020871
IS_A2_B2_A1_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0020590
IS_A2_B2_A1_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0020590
IS_A2_B2_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0021161
IS_A2_B2_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020619, upper bound: 0.0021161
IS_A2_B2_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
IS_A2_B2_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
IS_A2_B2_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
IS_A2_B2_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
IS_A2_B2_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
IS_A2_B2_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0019844
IS_A2_B2_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
IS_A2_B2_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021533, upper bound: 0.0020287
IS_A2_B2_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0020552
IS_A2_B2_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0020552
IS_A2_B2_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0021046
IS_A2_B2_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021320, upper bound: 0.0021046
IS_A2_B2_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0020552
IS_A2_B2_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0020552
IS_A2_B2_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0021046
IS_A2_B2_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021237, upper bound: 0.0021046
IS_A2_B2_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019836
IS_A2_B2_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019836
IS_A2_B2_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
IS_A2_B2_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
IS_A2_B2_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
IS_A2_B2_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
IS_A2_B2_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259
IS_A2_B2_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259
IS_A2_B2_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0020625
IS_A2_B2_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0020625
IS_A2_B2_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0021108
IS_A2_B2_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0021108
IS_A2_B2_A2_B2_A1_B2_A1_B1, status: Status.VERIFIED, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0020625
IS_A2_B2_A2_B2_A1_B2_A1_B2, status: Status.VERIFIED, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0020625
IS_A2_B2_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0021108
IS_A2_B2_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0020540, upper bound: 0.0021108
IS_A2_B2_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019837
IS_A2_B2_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0019837
IS_A2_B2_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
IS_A2_B2_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021606, upper bound: 0.0020259
IS_A2_B2_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
IS_A2_B2_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0019837
IS_A2_B2_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259
IS_A2_B2_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.30
Output dim: 8, lower bound: -0.0021455, upper bound: 0.0020259

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0063144, 0.0086391, -0.0016311, 0.0016311
1: 0.0022345, 0.0025704, 0.0022345, 0.0025704, -0.0002356, 0.0002356
2: 0.0095835, 0.0108688, 0.0095835, 0.0108688, -0.0009018, 0.0009018
3: -0.0047687, -0.0034394, -0.0047687, -0.0034394, -0.0009327, 0.0009327
4: -0.0003136, 0.0011255, -0.0003136, 0.0011255, -0.0010097, 0.0010097
5: 0.0030483, 0.0044101, 0.0030483, 0.0044101, -0.0009555, 0.0009555
6: -0.0102056, -0.0048024, -0.0102056, -0.0048024, -0.0037911, 0.0037911
7: 0.0039837, 0.0113425, 0.0039837, 0.0113425, -0.0051632, 0.0051632
8: 0.9920201, 0.9972038, 0.9920201, 0.9972038, -0.0036370, 0.0036370
9: -0.0133490, -0.0086437, -0.0133490, -0.0086437, -0.0033015, 0.0033015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0019884
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020950, upper bound: 0.0019884
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0062886, 0.0086712, -0.0016704, 0.0016663
1: 0.0022345, 0.0025704, 0.0022308, 0.0025750, -0.0002413, 0.0002407
2: 0.0095835, 0.0108688, 0.0095658, 0.0108831, -0.0009212, 0.0009235
3: -0.0047687, -0.0034394, -0.0047871, -0.0034247, -0.0009528, 0.0009552
4: -0.0003136, 0.0011255, -0.0003295, 0.0011453, -0.0010340, 0.0010315
5: 0.0030483, 0.0044101, 0.0030295, 0.0044252, -0.0009761, 0.0009785
6: -0.0102056, -0.0048024, -0.0102803, -0.0047424, -0.0038729, 0.0038825
7: 0.0039837, 0.0113425, 0.0039021, 0.0114441, -0.0052877, 0.0052746
8: 0.9920201, 0.9972038, 0.9919626, 0.9972753, -0.0037247, 0.0037155
9: -0.0133490, -0.0086437, -0.0134140, -0.0085915, -0.0033727, 0.0033811

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0019883
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020950, upper bound: 0.0019883
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0063144, 0.0086391, -0.0016663, 0.0016704
1: 0.0022308, 0.0025750, 0.0022345, 0.0025704, -0.0002407, 0.0002413
2: 0.0095658, 0.0108831, 0.0095835, 0.0108688, -0.0009235, 0.0009212
3: -0.0047871, -0.0034247, -0.0047687, -0.0034394, -0.0009552, 0.0009528
4: -0.0003295, 0.0011453, -0.0003136, 0.0011255, -0.0010315, 0.0010340
5: 0.0030295, 0.0044252, 0.0030483, 0.0044101, -0.0009785, 0.0009761
6: -0.0102803, -0.0047424, -0.0102056, -0.0048024, -0.0038825, 0.0038729
7: 0.0039021, 0.0114441, 0.0039837, 0.0113425, -0.0052746, 0.0052877
8: 0.9919626, 0.9972753, 0.9920201, 0.9972038, -0.0037155, 0.0037247
9: -0.0134140, -0.0085915, -0.0133490, -0.0086437, -0.0033811, 0.0033727

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0020420
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020883, upper bound: 0.0020421
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0062886, 0.0086712, -0.0016698, 0.0016698
1: 0.0022308, 0.0025750, 0.0022308, 0.0025750, -0.0002412, 0.0002412
2: 0.0095658, 0.0108831, 0.0095658, 0.0108831, -0.0009232, 0.0009232
3: -0.0047871, -0.0034247, -0.0047871, -0.0034247, -0.0009548, 0.0009548
4: -0.0003295, 0.0011453, -0.0003295, 0.0011453, -0.0010336, 0.0010336
5: 0.0030295, 0.0044252, 0.0030295, 0.0044252, -0.0009782, 0.0009782
6: -0.0102803, -0.0047424, -0.0102803, -0.0047424, -0.0038810, 0.0038810
7: 0.0039021, 0.0114441, 0.0039021, 0.0114441, -0.0052856, 0.0052856
8: 0.9919626, 0.9972753, 0.9919626, 0.9972753, -0.0037233, 0.0037233
9: -0.0134140, -0.0085915, -0.0134140, -0.0085915, -0.0033798, 0.0033798

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0020421
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020883, upper bound: 0.0020421
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0063581, 0.0088459, -0.0018823, 0.0016263
1: 0.0022345, 0.0025704, 0.0022409, 0.0026003, -0.0002719, 0.0002350
2: 0.0095835, 0.0108688, 0.0094692, 0.0108446, -0.0008991, 0.0010407
3: -0.0047687, -0.0034394, -0.0048870, -0.0034644, -0.0009299, 0.0010763
4: -0.0003136, 0.0011255, -0.0002865, 0.0012535, -0.0011652, 0.0010067
5: 0.0030483, 0.0044101, 0.0029272, 0.0043845, -0.0009527, 0.0011027
6: -0.0102056, -0.0048024, -0.0106862, -0.0049040, -0.0037799, 0.0043750
7: 0.0039837, 0.0113425, 0.0041221, 0.0119970, -0.0059584, 0.0051479
8: 0.9920201, 0.9972038, 0.9921175, 0.9976648, -0.0041972, 0.0036263
9: -0.0133490, -0.0086437, -0.0137675, -0.0087321, -0.0032917, 0.0038100

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020193
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020408, upper bound: 0.0020192
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0063205, 0.0088829, -0.0019239, 0.0016641
1: 0.0022345, 0.0025704, 0.0022354, 0.0026056, -0.0002779, 0.0002404
2: 0.0095835, 0.0108688, 0.0094488, 0.0108654, -0.0009200, 0.0010637
3: -0.0047687, -0.0034394, -0.0049081, -0.0034429, -0.0009515, 0.0011001
4: -0.0003136, 0.0011255, -0.0003098, 0.0012764, -0.0011909, 0.0010301
5: 0.0030483, 0.0044101, 0.0029055, 0.0044065, -0.0009748, 0.0011270
6: -0.0102056, -0.0048024, -0.0107722, -0.0048166, -0.0038678, 0.0044716
7: 0.0039837, 0.0113425, 0.0040031, 0.0121141, -0.0060899, 0.0052676
8: 0.9920201, 0.9972038, 0.9920337, 0.9977473, -0.0042899, 0.0037106
9: -0.0133490, -0.0086437, -0.0138425, -0.0086560, -0.0033682, 0.0038941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020193
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020408, upper bound: 0.0020192
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0063581, 0.0088459, -0.0019175, 0.0016656
1: 0.0022308, 0.0025750, 0.0022409, 0.0026003, -0.0002770, 0.0002406
2: 0.0095658, 0.0108831, 0.0094692, 0.0108446, -0.0009209, 0.0010601
3: -0.0047871, -0.0034247, -0.0048870, -0.0034644, -0.0009524, 0.0010964
4: -0.0003295, 0.0011453, -0.0002865, 0.0012535, -0.0011870, 0.0010310
5: 0.0030295, 0.0044252, 0.0029272, 0.0043845, -0.0009757, 0.0011233
6: -0.0102803, -0.0047424, -0.0106862, -0.0049040, -0.0038713, 0.0044568
7: 0.0039021, 0.0114441, 0.0041221, 0.0119970, -0.0060698, 0.0052724
8: 0.9919626, 0.9972753, 0.9921175, 0.9976648, -0.0042757, 0.0037140
9: -0.0134140, -0.0085915, -0.0137675, -0.0087321, -0.0033713, 0.0038812

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020276, upper bound: 0.0020789
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020278, upper bound: 0.0020789
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0063205, 0.0088829, -0.0019231, 0.0016683
1: 0.0022308, 0.0025750, 0.0022354, 0.0026056, -0.0002778, 0.0002410
2: 0.0095658, 0.0108831, 0.0094488, 0.0108654, -0.0009224, 0.0010632
3: -0.0047871, -0.0034247, -0.0049081, -0.0034429, -0.0009539, 0.0010997
4: -0.0003295, 0.0011453, -0.0003098, 0.0012764, -0.0011904, 0.0010327
5: 0.0030295, 0.0044252, 0.0029055, 0.0044065, -0.0009773, 0.0011266
6: -0.0102803, -0.0047424, -0.0107722, -0.0048166, -0.0038776, 0.0044698
7: 0.0039021, 0.0114441, 0.0040031, 0.0121141, -0.0060875, 0.0052809
8: 0.9919626, 0.9972753, 0.9920337, 0.9977473, -0.0042882, 0.0037200
9: -0.0134140, -0.0085915, -0.0138425, -0.0086560, -0.0033767, 0.0038925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020276, upper bound: 0.0020789
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020278, upper bound: 0.0020789
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0063144, 0.0086391, -0.0016263, 0.0018823
1: 0.0022409, 0.0026003, 0.0022345, 0.0025704, -0.0002350, 0.0002719
2: 0.0094692, 0.0108446, 0.0095835, 0.0108688, -0.0010407, 0.0008991
3: -0.0048870, -0.0034644, -0.0047687, -0.0034394, -0.0010763, 0.0009299
4: -0.0002865, 0.0012535, -0.0003136, 0.0011255, -0.0010067, 0.0011652
5: 0.0029272, 0.0043845, 0.0030483, 0.0044101, -0.0011027, 0.0009527
6: -0.0106862, -0.0049040, -0.0102056, -0.0048024, -0.0043750, 0.0037799
7: 0.0041221, 0.0119970, 0.0039837, 0.0113425, -0.0051479, 0.0059584
8: 0.9921175, 0.9976648, 0.9920201, 0.9972038, -0.0036263, 0.0041972
9: -0.0137675, -0.0087321, -0.0133490, -0.0086437, -0.0038100, 0.0032917

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0062886, 0.0086712, -0.0016656, 0.0019175
1: 0.0022409, 0.0026003, 0.0022308, 0.0025750, -0.0002406, 0.0002770
2: 0.0094692, 0.0108446, 0.0095658, 0.0108831, -0.0010601, 0.0009209
3: -0.0048870, -0.0034644, -0.0047871, -0.0034247, -0.0010964, 0.0009524
4: -0.0002865, 0.0012535, -0.0003295, 0.0011453, -0.0010310, 0.0011870
5: 0.0029272, 0.0043845, 0.0030295, 0.0044252, -0.0011233, 0.0009757
6: -0.0106862, -0.0049040, -0.0102803, -0.0047424, -0.0044568, 0.0038713
7: 0.0041221, 0.0119970, 0.0039021, 0.0114441, -0.0052724, 0.0060698
8: 0.9921175, 0.9976648, 0.9919626, 0.9972753, -0.0037140, 0.0042757
9: -0.0137675, -0.0087321, -0.0134140, -0.0085915, -0.0038812, 0.0033713

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0063144, 0.0086391, -0.0016641, 0.0019239
1: 0.0022354, 0.0026056, 0.0022345, 0.0025704, -0.0002404, 0.0002779
2: 0.0094488, 0.0108654, 0.0095835, 0.0108688, -0.0010637, 0.0009200
3: -0.0049081, -0.0034429, -0.0047687, -0.0034394, -0.0011001, 0.0009515
4: -0.0003098, 0.0012764, -0.0003136, 0.0011255, -0.0010301, 0.0011909
5: 0.0029055, 0.0044065, 0.0030483, 0.0044101, -0.0011270, 0.0009748
6: -0.0107722, -0.0048166, -0.0102056, -0.0048024, -0.0044716, 0.0038678
7: 0.0040031, 0.0121141, 0.0039837, 0.0113425, -0.0052676, 0.0060899
8: 0.9920337, 0.9977473, 0.9920201, 0.9972038, -0.0037106, 0.0042899
9: -0.0138425, -0.0086560, -0.0133490, -0.0086437, -0.0038941, 0.0033682

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0062886, 0.0086712, -0.0016683, 0.0019231
1: 0.0022354, 0.0026056, 0.0022308, 0.0025750, -0.0002410, 0.0002778
2: 0.0094488, 0.0108654, 0.0095658, 0.0108831, -0.0010632, 0.0009224
3: -0.0049081, -0.0034429, -0.0047871, -0.0034247, -0.0010997, 0.0009539
4: -0.0003098, 0.0012764, -0.0003295, 0.0011453, -0.0010327, 0.0011904
5: 0.0029055, 0.0044065, 0.0030295, 0.0044252, -0.0011266, 0.0009773
6: -0.0107722, -0.0048166, -0.0102803, -0.0047424, -0.0044698, 0.0038776
7: 0.0040031, 0.0121141, 0.0039021, 0.0114441, -0.0052809, 0.0060875
8: 0.9920337, 0.9977473, 0.9919626, 0.9972753, -0.0037200, 0.0042882
9: -0.0138425, -0.0086560, -0.0134140, -0.0085915, -0.0038925, 0.0033767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0063581, 0.0088459, -0.0017182, 0.0017182
1: 0.0022409, 0.0026003, 0.0022409, 0.0026003, -0.0002482, 0.0002482
2: 0.0094692, 0.0108446, 0.0094692, 0.0108446, -0.0009499, 0.0009499
3: -0.0048870, -0.0034644, -0.0048870, -0.0034644, -0.0009825, 0.0009825
4: -0.0002865, 0.0012535, -0.0002865, 0.0012535, -0.0010636, 0.0010636
5: 0.0029272, 0.0043845, 0.0029272, 0.0043845, -0.0010065, 0.0010065
6: -0.0106862, -0.0049040, -0.0106862, -0.0049040, -0.0039935, 0.0039935
7: 0.0041221, 0.0119970, 0.0041221, 0.0119970, -0.0054388, 0.0054388
8: 0.9921175, 0.9976648, 0.9921175, 0.9976648, -0.0038312, 0.0038312
9: -0.0137675, -0.0087321, -0.0137675, -0.0087321, -0.0034777, 0.0034777

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
time: 0.81 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0063205, 0.0088829, -0.0017585, 0.0017554
1: 0.0022409, 0.0026003, 0.0022354, 0.0026056, -0.0002541, 0.0002536
2: 0.0094692, 0.0108446, 0.0094488, 0.0108654, -0.0009705, 0.0009722
3: -0.0048870, -0.0034644, -0.0049081, -0.0034429, -0.0010038, 0.0010055
4: -0.0002865, 0.0012535, -0.0003098, 0.0012764, -0.0010886, 0.0010866
5: 0.0029272, 0.0043845, 0.0029055, 0.0044065, -0.0010283, 0.0010301
6: -0.0106862, -0.0049040, -0.0107722, -0.0048166, -0.0040801, 0.0040873
7: 0.0041221, 0.0119970, 0.0040031, 0.0121141, -0.0055666, 0.0055567
8: 0.9921175, 0.9976648, 0.9920337, 0.9977473, -0.0039212, 0.0039143
9: -0.0137675, -0.0087321, -0.0138425, -0.0086560, -0.0035531, 0.0035594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0063581, 0.0088459, -0.0017554, 0.0017585
1: 0.0022354, 0.0026056, 0.0022409, 0.0026003, -0.0002536, 0.0002541
2: 0.0094488, 0.0108654, 0.0094692, 0.0108446, -0.0009722, 0.0009705
3: -0.0049081, -0.0034429, -0.0048870, -0.0034644, -0.0010055, 0.0010038
4: -0.0003098, 0.0012764, -0.0002865, 0.0012535, -0.0010866, 0.0010886
5: 0.0029055, 0.0044065, 0.0029272, 0.0043845, -0.0010301, 0.0010283
6: -0.0107722, -0.0048166, -0.0106862, -0.0049040, -0.0040873, 0.0040801
7: 0.0040031, 0.0121141, 0.0041221, 0.0119970, -0.0055567, 0.0055666
8: 0.9920337, 0.9977473, 0.9921175, 0.9976648, -0.0039143, 0.0039212
9: -0.0138425, -0.0086560, -0.0137675, -0.0087321, -0.0035594, 0.0035531

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0063205, 0.0088829, -0.0017613, 0.0017613
1: 0.0022354, 0.0026056, 0.0022354, 0.0026056, -0.0002545, 0.0002545
2: 0.0094488, 0.0108654, 0.0094488, 0.0108654, -0.0009738, 0.0009738
3: -0.0049081, -0.0034429, -0.0049081, -0.0034429, -0.0010071, 0.0010071
4: -0.0003098, 0.0012764, -0.0003098, 0.0012764, -0.0010903, 0.0010903
5: 0.0029055, 0.0044065, 0.0029055, 0.0044065, -0.0010317, 0.0010317
6: -0.0107722, -0.0048166, -0.0107722, -0.0048166, -0.0040937, 0.0040937
7: 0.0040031, 0.0121141, 0.0040031, 0.0121141, -0.0055752, 0.0055752
8: 0.9920337, 0.9977473, 0.9920337, 0.9977473, -0.0039273, 0.0039273
9: -0.0138425, -0.0086560, -0.0138425, -0.0086560, -0.0035649, 0.0035649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0061747, 0.0085711, -0.0016347, 0.0018398
1: 0.0022345, 0.0025704, 0.0022144, 0.0025606, -0.0002362, 0.0002658
2: 0.0095835, 0.0108688, 0.0096211, 0.0109460, -0.0010172, 0.0009038
3: -0.0047687, -0.0034394, -0.0047298, -0.0033595, -0.0010520, 0.0009347
4: -0.0003136, 0.0011255, -0.0004001, 0.0010834, -0.0010119, 0.0011388
5: 0.0030483, 0.0044101, 0.0030881, 0.0044919, -0.0010777, 0.0009576
6: -0.0102056, -0.0048024, -0.0100476, -0.0044776, -0.0042761, 0.0037995
7: 0.0039837, 0.0113425, 0.0035415, 0.0111272, -0.0051746, 0.0058237
8: 0.9920201, 0.9972038, 0.9917086, 0.9970521, -0.0036451, 0.0041023
9: -0.0133490, -0.0086437, -0.0132114, -0.0083608, -0.0037238, 0.0033088

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0019614
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020723, upper bound: 0.0019614
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063144, 0.0086391, 0.0061549, 0.0085998, -0.0016572, 0.0018531
1: 0.0022345, 0.0025704, 0.0022115, 0.0025647, -0.0002394, 0.0002677
2: 0.0095835, 0.0108688, 0.0096053, 0.0109570, -0.0010246, 0.0009162
3: -0.0047687, -0.0034394, -0.0047463, -0.0033482, -0.0010596, 0.0009476
4: -0.0003136, 0.0011255, -0.0004123, 0.0011011, -0.0010258, 0.0011471
5: 0.0030483, 0.0044101, 0.0030713, 0.0045035, -0.0010856, 0.0009708
6: -0.0102056, -0.0048024, -0.0101143, -0.0044317, -0.0043072, 0.0038518
7: 0.0039837, 0.0113425, 0.0034788, 0.0112181, -0.0052458, 0.0058661
8: 0.9920201, 0.9972038, 0.9916644, 0.9971162, -0.0036953, 0.0041322
9: -0.0133490, -0.0086437, -0.0132695, -0.0083208, -0.0037509, 0.0033543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0019614
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020723, upper bound: 0.0019614
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0061747, 0.0085711, -0.0016699, 0.0018791
1: 0.0022308, 0.0025750, 0.0022144, 0.0025606, -0.0002413, 0.0002715
2: 0.0095658, 0.0108831, 0.0096211, 0.0109460, -0.0010389, 0.0009232
3: -0.0047871, -0.0034247, -0.0047298, -0.0033595, -0.0010745, 0.0009549
4: -0.0003295, 0.0011453, -0.0004001, 0.0010834, -0.0010337, 0.0011632
5: 0.0030295, 0.0044252, 0.0030881, 0.0044919, -0.0011008, 0.0009782
6: -0.0102803, -0.0047424, -0.0100476, -0.0044776, -0.0043675, 0.0038813
7: 0.0039021, 0.0114441, 0.0035415, 0.0111272, -0.0052860, 0.0059482
8: 0.9919626, 0.9972753, 0.9917086, 0.9970521, -0.0037236, 0.0041900
9: -0.0134140, -0.0085915, -0.0132114, -0.0083608, -0.0038034, 0.0033800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020670, upper bound: 0.0020133
time: 0.81 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020682, upper bound: 0.0020131
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0061549, 0.0085998, -0.0016698, 0.0018687
1: 0.0022308, 0.0025750, 0.0022115, 0.0025647, -0.0002412, 0.0002700
2: 0.0095658, 0.0108831, 0.0096053, 0.0109570, -0.0010332, 0.0009232
3: -0.0047871, -0.0034247, -0.0047463, -0.0033482, -0.0010685, 0.0009548
4: -0.0003295, 0.0011453, -0.0004123, 0.0011011, -0.0010337, 0.0011568
5: 0.0030295, 0.0044252, 0.0030713, 0.0045035, -0.0010947, 0.0009782
6: -0.0102803, -0.0047424, -0.0101143, -0.0044317, -0.0043434, 0.0038812
7: 0.0039021, 0.0114441, 0.0034788, 0.0112181, -0.0052858, 0.0059153
8: 0.9919626, 0.9972753, 0.9916644, 0.9971162, -0.0037234, 0.0041669
9: -0.0134140, -0.0085915, -0.0132695, -0.0083208, -0.0037824, 0.0033799

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020670, upper bound: 0.0020133
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020682, upper bound: 0.0020132
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0062134, 0.0087788, -0.0019118, 0.0018697
1: 0.0022308, 0.0025750, 0.0022200, 0.0025906, -0.0002762, 0.0002701
2: 0.0095658, 0.0108831, 0.0095063, 0.0109246, -0.0010337, 0.0010570
3: -0.0047871, -0.0034247, -0.0048486, -0.0033817, -0.0010691, 0.0010932
4: -0.0003295, 0.0011453, -0.0003761, 0.0012119, -0.0011834, 0.0011574
5: 0.0030295, 0.0044252, 0.0029664, 0.0044693, -0.0010953, 0.0011199
6: -0.0102803, -0.0047424, -0.0105304, -0.0045676, -0.0043457, 0.0044435
7: 0.0039021, 0.0114441, 0.0036640, 0.0117847, -0.0060517, 0.0059185
8: 0.9919626, 0.9972753, 0.9917949, 0.9975153, -0.0042629, 0.0041691
9: -0.0134140, -0.0085915, -0.0136318, -0.0084392, -0.0037845, 0.0038696

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019836, upper bound: 0.0020219
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019767, upper bound: 0.0020177
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0062886, 0.0086712, 0.0061933, 0.0088035, -0.0019110, 0.0018548
1: 0.0022308, 0.0025750, 0.0022171, 0.0025942, -0.0002761, 0.0002680
2: 0.0095658, 0.0108831, 0.0094926, 0.0109357, -0.0010255, 0.0010566
3: -0.0047871, -0.0034247, -0.0048628, -0.0033702, -0.0010606, 0.0010927
4: -0.0003295, 0.0011453, -0.0003885, 0.0012273, -0.0011830, 0.0011482
5: 0.0030295, 0.0044252, 0.0029520, 0.0044810, -0.0010865, 0.0011195
6: -0.0102803, -0.0047424, -0.0105878, -0.0045210, -0.0043111, 0.0044417
7: 0.0039021, 0.0114441, 0.0036006, 0.0118630, -0.0060493, 0.0058713
8: 0.9919626, 0.9972753, 0.9917502, 0.9975703, -0.0042612, 0.0041359
9: -0.0134140, -0.0085915, -0.0136819, -0.0083986, -0.0037543, 0.0038681

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019836, upper bound: 0.0020219
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019767, upper bound: 0.0020177
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0061747, 0.0085711, -0.0016299, 0.0020910
1: 0.0022409, 0.0026003, 0.0022144, 0.0025606, -0.0002355, 0.0003021
2: 0.0094692, 0.0108446, 0.0096211, 0.0109460, -0.0011561, 0.0009011
3: -0.0048870, -0.0034644, -0.0047298, -0.0033595, -0.0011956, 0.0009320
4: -0.0002865, 0.0012535, -0.0004001, 0.0010834, -0.0010089, 0.0012944
5: 0.0029272, 0.0043845, 0.0030881, 0.0044919, -0.0012249, 0.0009548
6: -0.0106862, -0.0049040, -0.0100476, -0.0044776, -0.0048600, 0.0037883
7: 0.0041221, 0.0119970, 0.0035415, 0.0111272, -0.0051594, 0.0066189
8: 0.9921175, 0.9976648, 0.9917086, 0.9970521, -0.0036344, 0.0046625
9: -0.0137675, -0.0087321, -0.0132114, -0.0083608, -0.0042323, 0.0032990

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020592, upper bound: 0.0019063
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020590, upper bound: 0.0019060
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0061549, 0.0085998, -0.0016524, 0.0021044
1: 0.0022409, 0.0026003, 0.0022115, 0.0025647, -0.0002387, 0.0003040
2: 0.0094692, 0.0108446, 0.0096053, 0.0109570, -0.0011635, 0.0009136
3: -0.0048870, -0.0034644, -0.0047463, -0.0033482, -0.0012033, 0.0009449
4: -0.0002865, 0.0012535, -0.0004123, 0.0011011, -0.0010229, 0.0013026
5: 0.0029272, 0.0043845, 0.0030713, 0.0045035, -0.0012327, 0.0009680
6: -0.0106862, -0.0049040, -0.0101143, -0.0044317, -0.0048911, 0.0038406
7: 0.0041221, 0.0119970, 0.0034788, 0.0112181, -0.0052306, 0.0066613
8: 0.9921175, 0.9976648, 0.9916644, 0.9971162, -0.0036845, 0.0046924
9: -0.0137675, -0.0087321, -0.0132695, -0.0083208, -0.0042594, 0.0033446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020592, upper bound: 0.0019063
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020590, upper bound: 0.0019060
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0061747, 0.0085711, -0.0016677, 0.0021325
1: 0.0022354, 0.0026056, 0.0022144, 0.0025606, -0.0002409, 0.0003081
2: 0.0094488, 0.0108654, 0.0096211, 0.0109460, -0.0011790, 0.0009220
3: -0.0049081, -0.0034429, -0.0047298, -0.0033595, -0.0012194, 0.0009536
4: -0.0003098, 0.0012764, -0.0004001, 0.0010834, -0.0010323, 0.0013201
5: 0.0029055, 0.0044065, 0.0030881, 0.0044919, -0.0012492, 0.0009769
6: -0.0107722, -0.0048166, -0.0100476, -0.0044776, -0.0049566, 0.0038762
7: 0.0040031, 0.0121141, 0.0035415, 0.0111272, -0.0052790, 0.0067505
8: 0.9920337, 0.9977473, 0.9917086, 0.9970521, -0.0037187, 0.0047552
9: -0.0138425, -0.0086560, -0.0132114, -0.0083608, -0.0043164, 0.0033756

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020513, upper bound: 0.0019462
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0019461
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0061549, 0.0085998, -0.0016683, 0.0021220
1: 0.0022354, 0.0026056, 0.0022115, 0.0025647, -0.0002410, 0.0003066
2: 0.0094488, 0.0108654, 0.0096053, 0.0109570, -0.0011732, 0.0009224
3: -0.0049081, -0.0034429, -0.0047463, -0.0033482, -0.0012134, 0.0009540
4: -0.0003098, 0.0012764, -0.0004123, 0.0011011, -0.0010327, 0.0013136
5: 0.0029055, 0.0044065, 0.0030713, 0.0045035, -0.0012431, 0.0009773
6: -0.0107722, -0.0048166, -0.0101143, -0.0044317, -0.0049322, 0.0038777
7: 0.0040031, 0.0121141, 0.0034788, 0.0112181, -0.0052810, 0.0067172
8: 0.9920337, 0.9977473, 0.9916644, 0.9971162, -0.0037201, 0.0047318
9: -0.0138425, -0.0086560, -0.0132695, -0.0083208, -0.0042952, 0.0033768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020513, upper bound: 0.0019462
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0019461
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0062134, 0.0087788, -0.0017259, 0.0019323
1: 0.0022409, 0.0026003, 0.0022200, 0.0025906, -0.0002493, 0.0002792
2: 0.0094692, 0.0108446, 0.0095063, 0.0109246, -0.0010683, 0.0009542
3: -0.0048870, -0.0034644, -0.0048486, -0.0033817, -0.0011049, 0.0009869
4: -0.0002865, 0.0012535, -0.0003761, 0.0012119, -0.0010684, 0.0011961
5: 0.0029272, 0.0043845, 0.0029664, 0.0044693, -0.0011320, 0.0010111
6: -0.0106862, -0.0049040, -0.0105304, -0.0045676, -0.0044913, 0.0040116
7: 0.0041221, 0.0119970, 0.0036640, 0.0117847, -0.0054634, 0.0061167
8: 0.9921175, 0.9976648, 0.9917949, 0.9975153, -0.0038485, 0.0043087
9: -0.0137675, -0.0087321, -0.0136318, -0.0084392, -0.0039112, 0.0034934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0019063
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020529, upper bound: 0.0019060
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0063581, 0.0088459, 0.0061933, 0.0088035, -0.0017463, 0.0019477
1: 0.0022409, 0.0026003, 0.0022171, 0.0025942, -0.0002523, 0.0002814
2: 0.0094692, 0.0108446, 0.0094926, 0.0109357, -0.0010768, 0.0009655
3: -0.0048870, -0.0034644, -0.0048628, -0.0033702, -0.0011137, 0.0009986
4: -0.0002865, 0.0012535, -0.0003885, 0.0012273, -0.0010810, 0.0012057
5: 0.0029272, 0.0043845, 0.0029520, 0.0044810, -0.0011410, 0.0010230
6: -0.0106862, -0.0049040, -0.0105878, -0.0045210, -0.0045270, 0.0040589
7: 0.0041221, 0.0119970, 0.0036006, 0.0118630, -0.0055279, 0.0061654
8: 0.9921175, 0.9976648, 0.9917502, 0.9975703, -0.0038940, 0.0043431
9: -0.0137675, -0.0087321, -0.0136819, -0.0083986, -0.0039423, 0.0035347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0019063
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020529, upper bound: 0.0019060
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0062134, 0.0087788, -0.0017632, 0.0019727
1: 0.0022354, 0.0026056, 0.0022200, 0.0025906, -0.0002547, 0.0002850
2: 0.0094488, 0.0108654, 0.0095063, 0.0109246, -0.0010906, 0.0009748
3: -0.0049081, -0.0034429, -0.0048486, -0.0033817, -0.0011280, 0.0010082
4: -0.0003098, 0.0012764, -0.0003761, 0.0012119, -0.0010914, 0.0012211
5: 0.0029055, 0.0044065, 0.0029664, 0.0044693, -0.0011556, 0.0010329
6: -0.0107722, -0.0048166, -0.0105304, -0.0045676, -0.0045850, 0.0040981
7: 0.0040031, 0.0121141, 0.0036640, 0.0117847, -0.0055813, 0.0062444
8: 0.9920337, 0.9977473, 0.9917949, 0.9975153, -0.0039316, 0.0043987
9: -0.0138425, -0.0086560, -0.0136318, -0.0084392, -0.0039928, 0.0035688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020480, upper bound: 0.0019462
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020475, upper bound: 0.0019461
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0063205, 0.0088829, 0.0061933, 0.0088035, -0.0017642, 0.0019650
1: 0.0022354, 0.0026056, 0.0022171, 0.0025942, -0.0002549, 0.0002839
2: 0.0094488, 0.0108654, 0.0094926, 0.0109357, -0.0010864, 0.0009754
3: -0.0049081, -0.0034429, -0.0048628, -0.0033702, -0.0011236, 0.0010088
4: -0.0003098, 0.0012764, -0.0003885, 0.0012273, -0.0010921, 0.0012164
5: 0.0029055, 0.0044065, 0.0029520, 0.0044810, -0.0011511, 0.0010335
6: -0.0107722, -0.0048166, -0.0105878, -0.0045210, -0.0045673, 0.0041005
7: 0.0040031, 0.0121141, 0.0036006, 0.0118630, -0.0055845, 0.0062202
8: 0.9920337, 0.9977473, 0.9917502, 0.9975703, -0.0039339, 0.0043816
9: -0.0138425, -0.0086560, -0.0136819, -0.0083986, -0.0039774, 0.0035709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.93 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020480, upper bound: 0.0019462
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020475, upper bound: 0.0019461
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0063144, 0.0086391, -0.0018398, 0.0016347
1: 0.0022144, 0.0025606, 0.0022345, 0.0025704, -0.0002658, 0.0002362
2: 0.0096211, 0.0109460, 0.0095835, 0.0108688, -0.0009038, 0.0010172
3: -0.0047298, -0.0033595, -0.0047687, -0.0034394, -0.0009347, 0.0010520
4: -0.0004001, 0.0010834, -0.0003136, 0.0011255, -0.0011388, 0.0010119
5: 0.0030881, 0.0044919, 0.0030483, 0.0044101, -0.0009576, 0.0010777
6: -0.0100476, -0.0044776, -0.0102056, -0.0048024, -0.0037995, 0.0042761
7: 0.0035415, 0.0111272, 0.0039837, 0.0113425, -0.0058237, 0.0051746
8: 0.9917086, 0.9970521, 0.9920201, 0.9972038, -0.0041023, 0.0036451
9: -0.0132114, -0.0083608, -0.0133490, -0.0086437, -0.0033088, 0.0037238

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020693, upper bound: 0.0019794
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020730, upper bound: 0.0019793
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0062886, 0.0086712, -0.0018791, 0.0016699
1: 0.0022144, 0.0025606, 0.0022308, 0.0025750, -0.0002715, 0.0002413
2: 0.0096211, 0.0109460, 0.0095658, 0.0108831, -0.0009232, 0.0010389
3: -0.0047298, -0.0033595, -0.0047871, -0.0034247, -0.0009549, 0.0010745
4: -0.0004001, 0.0010834, -0.0003295, 0.0011453, -0.0011632, 0.0010337
5: 0.0030881, 0.0044919, 0.0030295, 0.0044252, -0.0009782, 0.0011008
6: -0.0100476, -0.0044776, -0.0102803, -0.0047424, -0.0038813, 0.0043675
7: 0.0035415, 0.0111272, 0.0039021, 0.0114441, -0.0059482, 0.0052860
8: 0.9917086, 0.9970521, 0.9919626, 0.9972753, -0.0041900, 0.0037236
9: -0.0132114, -0.0083608, -0.0134140, -0.0085915, -0.0033800, 0.0038034

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.85 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020693, upper bound: 0.0019794
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020730, upper bound: 0.0019793
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0063144, 0.0086391, -0.0018531, 0.0016572
1: 0.0022115, 0.0025647, 0.0022345, 0.0025704, -0.0002677, 0.0002394
2: 0.0096053, 0.0109570, 0.0095835, 0.0108688, -0.0009162, 0.0010246
3: -0.0047463, -0.0033482, -0.0047687, -0.0034394, -0.0009476, 0.0010596
4: -0.0004123, 0.0011011, -0.0003136, 0.0011255, -0.0011471, 0.0010258
5: 0.0030713, 0.0045035, 0.0030483, 0.0044101, -0.0009708, 0.0010856
6: -0.0101143, -0.0044317, -0.0102056, -0.0048024, -0.0038518, 0.0043072
7: 0.0034788, 0.0112181, 0.0039837, 0.0113425, -0.0058661, 0.0052458
8: 0.9916644, 0.9971162, 0.9920201, 0.9972038, -0.0041322, 0.0036953
9: -0.0132695, -0.0083208, -0.0133490, -0.0086437, -0.0033543, 0.0037509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020612, upper bound: 0.0020236
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020660, upper bound: 0.0020236
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0062886, 0.0086712, -0.0018687, 0.0016698
1: 0.0022115, 0.0025647, 0.0022308, 0.0025750, -0.0002700, 0.0002412
2: 0.0096053, 0.0109570, 0.0095658, 0.0108831, -0.0009232, 0.0010332
3: -0.0047463, -0.0033482, -0.0047871, -0.0034247, -0.0009548, 0.0010685
4: -0.0004123, 0.0011011, -0.0003295, 0.0011453, -0.0011568, 0.0010337
5: 0.0030713, 0.0045035, 0.0030295, 0.0044252, -0.0009782, 0.0010947
6: -0.0101143, -0.0044317, -0.0102803, -0.0047424, -0.0038812, 0.0043434
7: 0.0034788, 0.0112181, 0.0039021, 0.0114441, -0.0059153, 0.0052858
8: 0.9916644, 0.9971162, 0.9919626, 0.9972753, -0.0041669, 0.0037234
9: -0.0132695, -0.0083208, -0.0134140, -0.0085915, -0.0033799, 0.0037824

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020612, upper bound: 0.0020236
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020660, upper bound: 0.0020236
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0061747, 0.0085711, -0.0016654, 0.0016654
1: 0.0022144, 0.0025606, 0.0022144, 0.0025606, -0.0002406, 0.0002406
2: 0.0096211, 0.0109460, 0.0096211, 0.0109460, -0.0009207, 0.0009207
3: -0.0047298, -0.0033595, -0.0047298, -0.0033595, -0.0009523, 0.0009523
4: -0.0004001, 0.0010834, -0.0004001, 0.0010834, -0.0010309, 0.0010309
5: 0.0030881, 0.0044919, 0.0030881, 0.0044919, -0.0009756, 0.0009756
6: -0.0100476, -0.0044776, -0.0100476, -0.0044776, -0.0038707, 0.0038707
7: 0.0035415, 0.0111272, 0.0035415, 0.0111272, -0.0052716, 0.0052716
8: 0.9917086, 0.9970521, 0.9917086, 0.9970521, -0.0037134, 0.0037134
9: -0.0132114, -0.0083608, -0.0132114, -0.0083608, -0.0033708, 0.0033708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020587, upper bound: 0.0019794
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020615, upper bound: 0.0019793
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0061549, 0.0085998, -0.0017022, 0.0016962
1: 0.0022144, 0.0025606, 0.0022115, 0.0025647, -0.0002459, 0.0002450
2: 0.0096211, 0.0109460, 0.0096053, 0.0109570, -0.0009378, 0.0009411
3: -0.0047298, -0.0033595, -0.0047463, -0.0033482, -0.0009699, 0.0009734
4: -0.0004001, 0.0010834, -0.0004123, 0.0011011, -0.0010537, 0.0010500
5: 0.0030881, 0.0044919, 0.0030713, 0.0045035, -0.0009936, 0.0009972
6: -0.0100476, -0.0044776, -0.0101143, -0.0044317, -0.0039423, 0.0039565
7: 0.0035415, 0.0111272, 0.0034788, 0.0112181, -0.0053884, 0.0053691
8: 0.9917086, 0.9970521, 0.9916644, 0.9971162, -0.0037957, 0.0037821
9: -0.0132114, -0.0083608, -0.0132695, -0.0083208, -0.0034332, 0.0034455

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020587, upper bound: 0.0019794
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020615, upper bound: 0.0019793
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0061747, 0.0085711, -0.0016962, 0.0017022
1: 0.0022115, 0.0025647, 0.0022144, 0.0025606, -0.0002450, 0.0002459
2: 0.0096053, 0.0109570, 0.0096211, 0.0109460, -0.0009411, 0.0009378
3: -0.0047463, -0.0033482, -0.0047298, -0.0033595, -0.0009734, 0.0009699
4: -0.0004123, 0.0011011, -0.0004001, 0.0010834, -0.0010499, 0.0010537
5: 0.0030713, 0.0045035, 0.0030881, 0.0044919, -0.0009972, 0.0009936
6: -0.0101143, -0.0044317, -0.0100476, -0.0044776, -0.0039565, 0.0039423
7: 0.0034788, 0.0112181, 0.0035415, 0.0111272, -0.0053691, 0.0053884
8: 0.9916644, 0.9971162, 0.9917086, 0.9970521, -0.0037821, 0.0037957
9: -0.0132695, -0.0083208, -0.0132114, -0.0083608, -0.0034455, 0.0034332

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0020236
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020555, upper bound: 0.0020236
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0061549, 0.0085998, -0.0017012, 0.0017012
1: 0.0022115, 0.0025647, 0.0022115, 0.0025647, -0.0002458, 0.0002458
2: 0.0096053, 0.0109570, 0.0096053, 0.0109570, -0.0009405, 0.0009405
3: -0.0047463, -0.0033482, -0.0047463, -0.0033482, -0.0009727, 0.0009727
4: -0.0004123, 0.0011011, -0.0004123, 0.0011011, -0.0010530, 0.0010530
5: 0.0030713, 0.0045035, 0.0030713, 0.0045035, -0.0009965, 0.0009965
6: -0.0101143, -0.0044317, -0.0101143, -0.0044317, -0.0039539, 0.0039539
7: 0.0034788, 0.0112181, 0.0034788, 0.0112181, -0.0053849, 0.0053849
8: 0.9916644, 0.9971162, 0.9916644, 0.9971162, -0.0037933, 0.0037933
9: -0.0132695, -0.0083208, -0.0132695, -0.0083208, -0.0034433, 0.0034433

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0020236
time: 0.79 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020555, upper bound: 0.0020236
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0063144, 0.0086391, -0.0018304, 0.0018766
1: 0.0022200, 0.0025906, 0.0022345, 0.0025704, -0.0002644, 0.0002711
2: 0.0095063, 0.0109246, 0.0095835, 0.0108688, -0.0010375, 0.0010120
3: -0.0048486, -0.0033817, -0.0047687, -0.0034394, -0.0010731, 0.0010466
4: -0.0003761, 0.0012119, -0.0003136, 0.0011255, -0.0011330, 0.0011616
5: 0.0029664, 0.0044693, 0.0030483, 0.0044101, -0.0010993, 0.0010722
6: -0.0105304, -0.0045676, -0.0102056, -0.0048024, -0.0043617, 0.0042543
7: 0.0036640, 0.0117847, 0.0039837, 0.0113425, -0.0057940, 0.0059403
8: 0.9917949, 0.9975153, 0.9920201, 0.9972038, -0.0040814, 0.0041845
9: -0.0136318, -0.0084392, -0.0133490, -0.0086437, -0.0037984, 0.0037049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020714, upper bound: 0.0018976
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020733, upper bound: 0.0018976
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0062886, 0.0086712, -0.0018697, 0.0019118
1: 0.0022200, 0.0025906, 0.0022308, 0.0025750, -0.0002701, 0.0002762
2: 0.0095063, 0.0109246, 0.0095658, 0.0108831, -0.0010570, 0.0010337
3: -0.0048486, -0.0033817, -0.0047871, -0.0034247, -0.0010932, 0.0010691
4: -0.0003761, 0.0012119, -0.0003295, 0.0011453, -0.0011574, 0.0011834
5: 0.0029664, 0.0044693, 0.0030295, 0.0044252, -0.0011199, 0.0010953
6: -0.0105304, -0.0045676, -0.0102803, -0.0047424, -0.0044435, 0.0043457
7: 0.0036640, 0.0117847, 0.0039021, 0.0114441, -0.0059185, 0.0060517
8: 0.9917949, 0.9975153, 0.9919626, 0.9972753, -0.0041691, 0.0042629
9: -0.0136318, -0.0084392, -0.0134140, -0.0085915, -0.0038696, 0.0037845

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020714, upper bound: 0.0018976
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020733, upper bound: 0.0018976
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0063144, 0.0086391, -0.0018451, 0.0018979
1: 0.0022171, 0.0025942, 0.0022345, 0.0025704, -0.0002666, 0.0002742
2: 0.0094926, 0.0109357, 0.0095835, 0.0108688, -0.0010493, 0.0010201
3: -0.0048628, -0.0033702, -0.0047687, -0.0034394, -0.0010852, 0.0010551
4: -0.0003885, 0.0012273, -0.0003136, 0.0011255, -0.0011422, 0.0011748
5: 0.0029520, 0.0044810, 0.0030483, 0.0044101, -0.0011118, 0.0010809
6: -0.0105878, -0.0045210, -0.0102056, -0.0048024, -0.0044112, 0.0042886
7: 0.0036006, 0.0118630, 0.0039837, 0.0113425, -0.0058407, 0.0060076
8: 0.9917502, 0.9975703, 0.9920201, 0.9972038, -0.0041143, 0.0042319
9: -0.0136819, -0.0083986, -0.0133490, -0.0086437, -0.0038415, 0.0037347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020617, upper bound: 0.0019312
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0019312
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0062886, 0.0086712, -0.0018548, 0.0019110
1: 0.0022171, 0.0025942, 0.0022308, 0.0025750, -0.0002680, 0.0002761
2: 0.0094926, 0.0109357, 0.0095658, 0.0108831, -0.0010566, 0.0010255
3: -0.0048628, -0.0033702, -0.0047871, -0.0034247, -0.0010927, 0.0010606
4: -0.0003885, 0.0012273, -0.0003295, 0.0011453, -0.0011482, 0.0011830
5: 0.0029520, 0.0044810, 0.0030295, 0.0044252, -0.0011195, 0.0010865
6: -0.0105878, -0.0045210, -0.0102803, -0.0047424, -0.0044417, 0.0043111
7: 0.0036006, 0.0118630, 0.0039021, 0.0114441, -0.0058713, 0.0060493
8: 0.9917502, 0.9975703, 0.9919626, 0.9972753, -0.0041359, 0.0042612
9: -0.0136819, -0.0083986, -0.0134140, -0.0085915, -0.0038681, 0.0037543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020617, upper bound: 0.0019312
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0019312
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0061747, 0.0085711, -0.0016609, 0.0019154
1: 0.0022200, 0.0025906, 0.0022144, 0.0025606, -0.0002399, 0.0002767
2: 0.0095063, 0.0109246, 0.0096211, 0.0109460, -0.0010590, 0.0009182
3: -0.0048486, -0.0033817, -0.0047298, -0.0033595, -0.0010952, 0.0009497
4: -0.0003761, 0.0012119, -0.0004001, 0.0010834, -0.0010281, 0.0011857
5: 0.0029664, 0.0044693, 0.0030881, 0.0044919, -0.0011220, 0.0009729
6: -0.0105304, -0.0045676, -0.0100476, -0.0044776, -0.0044519, 0.0038603
7: 0.0036640, 0.0117847, 0.0035415, 0.0111272, -0.0052574, 0.0060631
8: 0.9917949, 0.9975153, 0.9917086, 0.9970521, -0.0037034, 0.0042709
9: -0.0136318, -0.0084392, -0.0132114, -0.0083608, -0.0038769, 0.0033617

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0018976
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020547, upper bound: 0.0018976
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0061549, 0.0085998, -0.0016977, 0.0019462
1: 0.0022200, 0.0025906, 0.0022115, 0.0025647, -0.0002453, 0.0002812
2: 0.0095063, 0.0109246, 0.0096053, 0.0109570, -0.0010760, 0.0009386
3: -0.0048486, -0.0033817, -0.0047463, -0.0033482, -0.0011128, 0.0009708
4: -0.0003761, 0.0012119, -0.0004123, 0.0011011, -0.0010509, 0.0012047
5: 0.0029664, 0.0044693, 0.0030713, 0.0045035, -0.0011401, 0.0009945
6: -0.0105304, -0.0045676, -0.0101143, -0.0044317, -0.0045235, 0.0039460
7: 0.0036640, 0.0117847, 0.0034788, 0.0112181, -0.0053741, 0.0061606
8: 0.9917949, 0.9975153, 0.9916644, 0.9971162, -0.0037857, 0.0043396
9: -0.0136318, -0.0084392, -0.0132695, -0.0083208, -0.0039392, 0.0034364

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0018976
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020547, upper bound: 0.0018976
time: 0.88 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0061747, 0.0085711, -0.0016956, 0.0019553
1: 0.0022171, 0.0025942, 0.0022144, 0.0025606, -0.0002450, 0.0002825
2: 0.0094926, 0.0109357, 0.0096211, 0.0109460, -0.0010810, 0.0009375
3: -0.0048628, -0.0033702, -0.0047298, -0.0033595, -0.0011180, 0.0009696
4: -0.0003885, 0.0012273, -0.0004001, 0.0010834, -0.0010496, 0.0012103
5: 0.0029520, 0.0044810, 0.0030881, 0.0044919, -0.0011454, 0.0009933
6: -0.0105878, -0.0045210, -0.0100476, -0.0044776, -0.0045446, 0.0039410
7: 0.0036006, 0.0118630, 0.0035415, 0.0111272, -0.0053674, 0.0061894
8: 0.9917502, 0.9975703, 0.9917086, 0.9970521, -0.0037809, 0.0043599
9: -0.0136819, -0.0083986, -0.0132114, -0.0083608, -0.0039576, 0.0034320

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020483, upper bound: 0.0019312
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020494, upper bound: 0.0019312
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0061549, 0.0085998, -0.0017010, 0.0019540
1: 0.0022171, 0.0025942, 0.0022115, 0.0025647, -0.0002458, 0.0002823
2: 0.0094926, 0.0109357, 0.0096053, 0.0109570, -0.0010803, 0.0009405
3: -0.0048628, -0.0033702, -0.0047463, -0.0033482, -0.0011173, 0.0009727
4: -0.0003885, 0.0012273, -0.0004123, 0.0011011, -0.0010530, 0.0012095
5: 0.0029520, 0.0044810, 0.0030713, 0.0045035, -0.0011446, 0.0009965
6: -0.0105878, -0.0045210, -0.0101143, -0.0044317, -0.0045416, 0.0039537
7: 0.0036006, 0.0118630, 0.0034788, 0.0112181, -0.0053846, 0.0061853
8: 0.9917502, 0.9975703, 0.9916644, 0.9971162, -0.0037930, 0.0043570
9: -0.0136819, -0.0083986, -0.0132695, -0.0083208, -0.0039550, 0.0034430

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020483, upper bound: 0.0019312
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020494, upper bound: 0.0019312
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0063581, 0.0088459, -0.0020910, 0.0016299
1: 0.0022144, 0.0025606, 0.0022409, 0.0026003, -0.0003021, 0.0002355
2: 0.0096211, 0.0109460, 0.0094692, 0.0108446, -0.0009011, 0.0011561
3: -0.0047298, -0.0033595, -0.0048870, -0.0034644, -0.0009320, 0.0011956
4: -0.0004001, 0.0010834, -0.0002865, 0.0012535, -0.0012944, 0.0010089
5: 0.0030881, 0.0044919, 0.0029272, 0.0043845, -0.0009548, 0.0012249
6: -0.0100476, -0.0044776, -0.0106862, -0.0049040, -0.0037883, 0.0048600
7: 0.0035415, 0.0111272, 0.0041221, 0.0119970, -0.0066189, 0.0051594
8: 0.9917086, 0.9970521, 0.9921175, 0.9976648, -0.0046625, 0.0036344
9: -0.0132114, -0.0083608, -0.0137675, -0.0087321, -0.0032990, 0.0042323

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020102, upper bound: 0.0019648
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020105, upper bound: 0.0019598
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0061747, 0.0085711, 0.0063205, 0.0088829, -0.0021325, 0.0016677
1: 0.0022144, 0.0025606, 0.0022354, 0.0026056, -0.0003081, 0.0002409
2: 0.0096211, 0.0109460, 0.0094488, 0.0108654, -0.0009220, 0.0011790
3: -0.0047298, -0.0033595, -0.0049081, -0.0034429, -0.0009536, 0.0012194
4: -0.0004001, 0.0010834, -0.0003098, 0.0012764, -0.0013201, 0.0010323
5: 0.0030881, 0.0044919, 0.0029055, 0.0044065, -0.0009769, 0.0012492
6: -0.0100476, -0.0044776, -0.0107722, -0.0048166, -0.0038762, 0.0049566
7: 0.0035415, 0.0111272, 0.0040031, 0.0121141, -0.0067505, 0.0052790
8: 0.9917086, 0.9970521, 0.9920337, 0.9977473, -0.0047552, 0.0037187
9: -0.0132114, -0.0083608, -0.0138425, -0.0086560, -0.0033756, 0.0043164

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020102, upper bound: 0.0019648
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020105, upper bound: 0.0019598
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0063581, 0.0088459, -0.0021044, 0.0016524
1: 0.0022115, 0.0025647, 0.0022409, 0.0026003, -0.0003040, 0.0002387
2: 0.0096053, 0.0109570, 0.0094692, 0.0108446, -0.0009136, 0.0011635
3: -0.0047463, -0.0033482, -0.0048870, -0.0034644, -0.0009449, 0.0012033
4: -0.0004123, 0.0011011, -0.0002865, 0.0012535, -0.0013026, 0.0010229
5: 0.0030713, 0.0045035, 0.0029272, 0.0043845, -0.0009680, 0.0012327
6: -0.0101143, -0.0044317, -0.0106862, -0.0049040, -0.0038406, 0.0048911
7: 0.0034788, 0.0112181, 0.0041221, 0.0119970, -0.0066613, 0.0052306
8: 0.9916644, 0.9971162, 0.9921175, 0.9976648, -0.0046924, 0.0036845
9: -0.0132695, -0.0083208, -0.0137675, -0.0087321, -0.0033446, 0.0042594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019989, upper bound: 0.0020071
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019988, upper bound: 0.0020066
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0063205, 0.0088829, -0.0021220, 0.0016683
1: 0.0022115, 0.0025647, 0.0022354, 0.0026056, -0.0003066, 0.0002410
2: 0.0096053, 0.0109570, 0.0094488, 0.0108654, -0.0009224, 0.0011732
3: -0.0047463, -0.0033482, -0.0049081, -0.0034429, -0.0009540, 0.0012134
4: -0.0004123, 0.0011011, -0.0003098, 0.0012764, -0.0013136, 0.0010327
5: 0.0030713, 0.0045035, 0.0029055, 0.0044065, -0.0009773, 0.0012431
6: -0.0101143, -0.0044317, -0.0107722, -0.0048166, -0.0038777, 0.0049322
7: 0.0034788, 0.0112181, 0.0040031, 0.0121141, -0.0067172, 0.0052810
8: 0.9916644, 0.9971162, 0.9920337, 0.9977473, -0.0047318, 0.0037201
9: -0.0132695, -0.0083208, -0.0138425, -0.0086560, -0.0033768, 0.0042952

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019989, upper bound: 0.0020071
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019988, upper bound: 0.0020066
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0062134, 0.0087788, -0.0019462, 0.0016977
1: 0.0022115, 0.0025647, 0.0022200, 0.0025906, -0.0002812, 0.0002453
2: 0.0096053, 0.0109570, 0.0095063, 0.0109246, -0.0009386, 0.0010760
3: -0.0047463, -0.0033482, -0.0048486, -0.0033817, -0.0009708, 0.0011128
4: -0.0004123, 0.0011011, -0.0003761, 0.0012119, -0.0012047, 0.0010509
5: 0.0030713, 0.0045035, 0.0029664, 0.0044693, -0.0009945, 0.0011401
6: -0.0101143, -0.0044317, -0.0105304, -0.0045676, -0.0039460, 0.0045235
7: 0.0034788, 0.0112181, 0.0036640, 0.0117847, -0.0061606, 0.0053741
8: 0.9916644, 0.9971162, 0.9917949, 0.9975153, -0.0043396, 0.0037857
9: -0.0132695, -0.0083208, -0.0136318, -0.0084392, -0.0034364, 0.0039392

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019790, upper bound: 0.0020071
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019746, upper bound: 0.0020066
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061549, 0.0085998, 0.0061933, 0.0088035, -0.0019540, 0.0017010
1: 0.0022115, 0.0025647, 0.0022171, 0.0025942, -0.0002823, 0.0002458
2: 0.0096053, 0.0109570, 0.0094926, 0.0109357, -0.0009405, 0.0010803
3: -0.0047463, -0.0033482, -0.0048628, -0.0033702, -0.0009727, 0.0011173
4: -0.0004123, 0.0011011, -0.0003885, 0.0012273, -0.0012095, 0.0010530
5: 0.0030713, 0.0045035, 0.0029520, 0.0044810, -0.0009965, 0.0011446
6: -0.0101143, -0.0044317, -0.0105878, -0.0045210, -0.0039537, 0.0045416
7: 0.0034788, 0.0112181, 0.0036006, 0.0118630, -0.0061853, 0.0053846
8: 0.9916644, 0.9971162, 0.9917502, 0.9975703, -0.0043570, 0.0037930
9: -0.0132695, -0.0083208, -0.0136819, -0.0083986, -0.0034430, 0.0039550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019790, upper bound: 0.0020071
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019746, upper bound: 0.0020066
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0063581, 0.0088459, -0.0019323, 0.0017259
1: 0.0022200, 0.0025906, 0.0022409, 0.0026003, -0.0002792, 0.0002493
2: 0.0095063, 0.0109246, 0.0094692, 0.0108446, -0.0009542, 0.0010683
3: -0.0048486, -0.0033817, -0.0048870, -0.0034644, -0.0009869, 0.0011049
4: -0.0003761, 0.0012119, -0.0002865, 0.0012535, -0.0011961, 0.0010684
5: 0.0029664, 0.0044693, 0.0029272, 0.0043845, -0.0010111, 0.0011320
6: -0.0105304, -0.0045676, -0.0106862, -0.0049040, -0.0040116, 0.0044913
7: 0.0036640, 0.0117847, 0.0041221, 0.0119970, -0.0061167, 0.0054634
8: 0.9917949, 0.9975153, 0.9921175, 0.9976648, -0.0043087, 0.0038485
9: -0.0136318, -0.0084392, -0.0137675, -0.0087321, -0.0034934, 0.0039112

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020711, upper bound: 0.0018976
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020727, upper bound: 0.0018976
time: 0.68 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0063205, 0.0088829, -0.0019727, 0.0017632
1: 0.0022200, 0.0025906, 0.0022354, 0.0026056, -0.0002850, 0.0002547
2: 0.0095063, 0.0109246, 0.0094488, 0.0108654, -0.0009748, 0.0010906
3: -0.0048486, -0.0033817, -0.0049081, -0.0034429, -0.0010082, 0.0011280
4: -0.0003761, 0.0012119, -0.0003098, 0.0012764, -0.0012211, 0.0010914
5: 0.0029664, 0.0044693, 0.0029055, 0.0044065, -0.0010329, 0.0011556
6: -0.0105304, -0.0045676, -0.0107722, -0.0048166, -0.0040981, 0.0045850
7: 0.0036640, 0.0117847, 0.0040031, 0.0121141, -0.0062444, 0.0055813
8: 0.9917949, 0.9975153, 0.9920337, 0.9977473, -0.0043987, 0.0039316
9: -0.0136318, -0.0084392, -0.0138425, -0.0086560, -0.0035688, 0.0039928

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020711, upper bound: 0.0018976
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020727, upper bound: 0.0018976
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0063581, 0.0088459, -0.0019477, 0.0017463
1: 0.0022171, 0.0025942, 0.0022409, 0.0026003, -0.0002814, 0.0002523
2: 0.0094926, 0.0109357, 0.0094692, 0.0108446, -0.0009655, 0.0010768
3: -0.0048628, -0.0033702, -0.0048870, -0.0034644, -0.0009986, 0.0011137
4: -0.0003885, 0.0012273, -0.0002865, 0.0012535, -0.0012057, 0.0010810
5: 0.0029520, 0.0044810, 0.0029272, 0.0043845, -0.0010230, 0.0011410
6: -0.0105878, -0.0045210, -0.0106862, -0.0049040, -0.0040589, 0.0045270
7: 0.0036006, 0.0118630, 0.0041221, 0.0119970, -0.0061654, 0.0055279
8: 0.9917502, 0.9975703, 0.9921175, 0.9976648, -0.0043431, 0.0038940
9: -0.0136819, -0.0083986, -0.0137675, -0.0087321, -0.0035347, 0.0039423

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020604, upper bound: 0.0019312
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020623, upper bound: 0.0019312
time: 0.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0063205, 0.0088829, -0.0019650, 0.0017642
1: 0.0022171, 0.0025942, 0.0022354, 0.0026056, -0.0002839, 0.0002549
2: 0.0094926, 0.0109357, 0.0094488, 0.0108654, -0.0009754, 0.0010864
3: -0.0048628, -0.0033702, -0.0049081, -0.0034429, -0.0010088, 0.0011236
4: -0.0003885, 0.0012273, -0.0003098, 0.0012764, -0.0012164, 0.0010921
5: 0.0029520, 0.0044810, 0.0029055, 0.0044065, -0.0010335, 0.0011511
6: -0.0105878, -0.0045210, -0.0107722, -0.0048166, -0.0041005, 0.0045673
7: 0.0036006, 0.0118630, 0.0040031, 0.0121141, -0.0062202, 0.0055845
8: 0.9917502, 0.9975703, 0.9920337, 0.9977473, -0.0043816, 0.0039339
9: -0.0136819, -0.0083986, -0.0138425, -0.0086560, -0.0035709, 0.0039774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020604, upper bound: 0.0019312
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020623, upper bound: 0.0019312
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0062134, 0.0087788, -0.0017563, 0.0017563
1: 0.0022200, 0.0025906, 0.0022200, 0.0025906, -0.0002537, 0.0002537
2: 0.0095063, 0.0109246, 0.0095063, 0.0109246, -0.0009710, 0.0009710
3: -0.0048486, -0.0033817, -0.0048486, -0.0033817, -0.0010043, 0.0010043
4: -0.0003761, 0.0012119, -0.0003761, 0.0012119, -0.0010872, 0.0010872
5: 0.0029664, 0.0044693, 0.0029664, 0.0044693, -0.0010289, 0.0010289
6: -0.0105304, -0.0045676, -0.0105304, -0.0045676, -0.0040822, 0.0040822
7: 0.0036640, 0.0117847, 0.0036640, 0.0117847, -0.0055596, 0.0055596
8: 0.9917949, 0.9975153, 0.9917949, 0.9975153, -0.0039163, 0.0039163
9: -0.0136318, -0.0084392, -0.0136318, -0.0084392, -0.0035550, 0.0035550

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020525, upper bound: 0.0018976
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020498, upper bound: 0.0018976
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0062134, 0.0087788, 0.0061933, 0.0088035, -0.0017943, 0.0017890
1: 0.0022200, 0.0025906, 0.0022171, 0.0025942, -0.0002592, 0.0002585
2: 0.0095063, 0.0109246, 0.0094926, 0.0109357, -0.0009891, 0.0009920
3: -0.0048486, -0.0033817, -0.0048628, -0.0033702, -0.0010230, 0.0010260
4: -0.0003761, 0.0012119, -0.0003885, 0.0012273, -0.0011107, 0.0011074
5: 0.0029664, 0.0044693, 0.0029520, 0.0044810, -0.0010480, 0.0010511
6: -0.0105304, -0.0045676, -0.0105878, -0.0045210, -0.0041581, 0.0041705
7: 0.0036640, 0.0117847, 0.0036006, 0.0118630, -0.0056798, 0.0056630
8: 0.9917949, 0.9975153, 0.9917502, 0.9975703, -0.0040010, 0.0039891
9: -0.0136318, -0.0084392, -0.0136819, -0.0083986, -0.0036211, 0.0036318

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020525, upper bound: 0.0018976
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020498, upper bound: 0.0018976
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0062134, 0.0087788, -0.0017890, 0.0017943
1: 0.0022171, 0.0025942, 0.0022200, 0.0025906, -0.0002585, 0.0002592
2: 0.0094926, 0.0109357, 0.0095063, 0.0109246, -0.0009920, 0.0009891
3: -0.0048628, -0.0033702, -0.0048486, -0.0033817, -0.0010260, 0.0010230
4: -0.0003885, 0.0012273, -0.0003761, 0.0012119, -0.0011074, 0.0011107
5: 0.0029520, 0.0044810, 0.0029664, 0.0044693, -0.0010511, 0.0010480
6: -0.0105878, -0.0045210, -0.0105304, -0.0045676, -0.0041705, 0.0041581
7: 0.0036006, 0.0118630, 0.0036640, 0.0117847, -0.0056630, 0.0056798
8: 0.9917502, 0.9975703, 0.9917949, 0.9975153, -0.0039891, 0.0040010
9: -0.0136819, -0.0083986, -0.0136318, -0.0084392, -0.0036318, 0.0036211

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020449, upper bound: 0.0019312
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020436, upper bound: 0.0019312
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0061933, 0.0088035, 0.0061933, 0.0088035, -0.0017937, 0.0017937
1: 0.0022171, 0.0025942, 0.0022171, 0.0025942, -0.0002591, 0.0002591
2: 0.0094926, 0.0109357, 0.0094926, 0.0109357, -0.0009917, 0.0009917
3: -0.0048628, -0.0033702, -0.0048628, -0.0033702, -0.0010257, 0.0010257
4: -0.0003885, 0.0012273, -0.0003885, 0.0012273, -0.0011103, 0.0011103
5: 0.0029520, 0.0044810, 0.0029520, 0.0044810, -0.0010508, 0.0010508
6: -0.0105878, -0.0045210, -0.0105878, -0.0045210, -0.0041691, 0.0041691
7: 0.0036006, 0.0118630, 0.0036006, 0.0118630, -0.0056780, 0.0056780
8: 0.9917502, 0.9975703, 0.9917502, 0.9975703, -0.0039997, 0.0039997
9: -0.0136819, -0.0083986, -0.0136819, -0.0083986, -0.0036306, 0.0036306

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020449, upper bound: 0.0019312
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0020436, upper bound: 0.0019312
time: 0.68 seconds

## Summary of splitting at layer (split count: 8)
- Time for IS candidates: 3.40 seconds
IS_A2_B2_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0019884
IS_A2_B2_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020950, upper bound: 0.0019884
IS_A2_B2_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0019883
IS_A2_B2_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020950, upper bound: 0.0019883
IS_A2_B2_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0020420
IS_A2_B2_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020883, upper bound: 0.0020421
IS_A2_B2_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0020421
IS_A2_B2_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020883, upper bound: 0.0020421
IS_A2_B2_A1_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020193
IS_A2_B2_A1_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020408, upper bound: 0.0020192
IS_A2_B2_A1_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020402, upper bound: 0.0020193
IS_A2_B2_A1_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020408, upper bound: 0.0020192
IS_A2_B2_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020276, upper bound: 0.0020789
IS_A2_B2_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020278, upper bound: 0.0020789
IS_A2_B2_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020276, upper bound: 0.0020789
IS_A2_B2_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020278, upper bound: 0.0020789
IS_A2_B2_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
IS_A2_B2_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
IS_A2_B2_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
IS_A2_B2_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
IS_A2_B2_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021255, upper bound: 0.0019419
IS_A2_B2_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
IS_A2_B2_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
IS_A2_B2_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
IS_A2_B2_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0021201, upper bound: 0.0019896
IS_A2_B2_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0019614
IS_A2_B2_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020723, upper bound: 0.0019614
IS_A2_B2_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020725, upper bound: 0.0019614
IS_A2_B2_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020723, upper bound: 0.0019614
IS_A2_B2_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020670, upper bound: 0.0020133
IS_A2_B2_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020682, upper bound: 0.0020131
IS_A2_B2_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020670, upper bound: 0.0020133
IS_A2_B2_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020682, upper bound: 0.0020132
IS_A2_B2_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019836, upper bound: 0.0020219
IS_A2_B2_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019767, upper bound: 0.0020177
IS_A2_B2_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019836, upper bound: 0.0020219
IS_A2_B2_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019767, upper bound: 0.0020177
IS_A2_B2_A1_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020592, upper bound: 0.0019063
IS_A2_B2_A1_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020590, upper bound: 0.0019060
IS_A2_B2_A1_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020592, upper bound: 0.0019063
IS_A2_B2_A1_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020590, upper bound: 0.0019060
IS_A2_B2_A1_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020513, upper bound: 0.0019462
IS_A2_B2_A1_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0019461
IS_A2_B2_A1_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020513, upper bound: 0.0019462
IS_A2_B2_A1_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0019461
IS_A2_B2_A1_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0019063
IS_A2_B2_A1_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020529, upper bound: 0.0019060
IS_A2_B2_A1_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0019063
IS_A2_B2_A1_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020529, upper bound: 0.0019060
IS_A2_B2_A1_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020480, upper bound: 0.0019462
IS_A2_B2_A1_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020475, upper bound: 0.0019461
IS_A2_B2_A1_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020480, upper bound: 0.0019462
IS_A2_B2_A1_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020475, upper bound: 0.0019461
IS_A2_B2_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020693, upper bound: 0.0019794
IS_A2_B2_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020730, upper bound: 0.0019793
IS_A2_B2_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020693, upper bound: 0.0019794
IS_A2_B2_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020730, upper bound: 0.0019793
IS_A2_B2_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020612, upper bound: 0.0020236
IS_A2_B2_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020660, upper bound: 0.0020236
IS_A2_B2_A2_B1_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020612, upper bound: 0.0020236
IS_A2_B2_A2_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020660, upper bound: 0.0020236
IS_A2_B2_A2_B1_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020587, upper bound: 0.0019794
IS_A2_B2_A2_B1_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020615, upper bound: 0.0019793
IS_A2_B2_A2_B1_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020587, upper bound: 0.0019794
IS_A2_B2_A2_B1_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020615, upper bound: 0.0019793
IS_A2_B2_A2_B1_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0020236
IS_A2_B2_A2_B1_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020555, upper bound: 0.0020236
IS_A2_B2_A2_B1_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020521, upper bound: 0.0020236
IS_A2_B2_A2_B1_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020555, upper bound: 0.0020236
IS_A2_B2_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020714, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020733, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020714, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020733, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020617, upper bound: 0.0019312
IS_A2_B2_A2_B1_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0019312
IS_A2_B2_A2_B1_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020617, upper bound: 0.0019312
IS_A2_B2_A2_B1_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020635, upper bound: 0.0019312
IS_A2_B2_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020547, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020549, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020547, upper bound: 0.0018976
IS_A2_B2_A2_B1_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020483, upper bound: 0.0019312
IS_A2_B2_A2_B1_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020494, upper bound: 0.0019312
IS_A2_B2_A2_B1_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020483, upper bound: 0.0019312
IS_A2_B2_A2_B1_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020494, upper bound: 0.0019312
IS_A2_B2_A2_B2_A1_B1_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020102, upper bound: 0.0019648
IS_A2_B2_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020105, upper bound: 0.0019598
IS_A2_B2_A2_B2_A1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020102, upper bound: 0.0019648
IS_A2_B2_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020105, upper bound: 0.0019598
IS_A2_B2_A2_B2_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019989, upper bound: 0.0020071
IS_A2_B2_A2_B2_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019988, upper bound: 0.0020066
IS_A2_B2_A2_B2_A1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019989, upper bound: 0.0020071
IS_A2_B2_A2_B2_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019988, upper bound: 0.0020066
IS_A2_B2_A2_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019790, upper bound: 0.0020071
IS_A2_B2_A2_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019746, upper bound: 0.0020066
IS_A2_B2_A2_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019790, upper bound: 0.0020071
IS_A2_B2_A2_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0019746, upper bound: 0.0020066
IS_A2_B2_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020711, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020727, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020711, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020727, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020604, upper bound: 0.0019312
IS_A2_B2_A2_B2_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020623, upper bound: 0.0019312
IS_A2_B2_A2_B2_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020604, upper bound: 0.0019312
IS_A2_B2_A2_B2_A2_B1_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020623, upper bound: 0.0019312
IS_A2_B2_A2_B2_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020525, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020498, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020525, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B2_A1_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020498, upper bound: 0.0018976
IS_A2_B2_A2_B2_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020449, upper bound: 0.0019312
IS_A2_B2_A2_B2_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020436, upper bound: 0.0019312
IS_A2_B2_A2_B2_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020449, upper bound: 0.0019312
IS_A2_B2_A2_B2_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 9, time: 3.40
Output dim: 8, lower bound: -0.0020436, upper bound: 0.0019312

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063406, 0.0086374, 0.0063144, 0.0086391, -0.0016043, 0.0016296
1: 0.0022383, 0.0025702, 0.0022345, 0.0025704, -0.0002318, 0.0002354
2: 0.0095845, 0.0108543, 0.0095835, 0.0108688, -0.0009010, 0.0008870
3: -0.0047678, -0.0034544, -0.0047687, -0.0034394, -0.0009318, 0.0009173
4: -0.0002973, 0.0011244, -0.0003136, 0.0011255, -0.0009931, 0.0010087
5: 0.0030493, 0.0043947, 0.0030483, 0.0044101, -0.0009546, 0.0009398
6: -0.0102017, -0.0048633, -0.0102056, -0.0048024, -0.0037876, 0.0037288
7: 0.0040667, 0.0113372, 0.0039837, 0.0113425, -0.0050783, 0.0051584
8: 0.9920785, 0.9972000, 0.9920201, 0.9972038, -0.0035772, 0.0036337
9: -0.0133456, -0.0086967, -0.0133490, -0.0086437, -0.0032984, 0.0032472

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0020042
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020902, upper bound: 0.0020042
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063670, 0.0086784, 0.0063322, 0.0086380, -0.0015937, 0.0016650
1: 0.0022421, 0.0025761, 0.0022371, 0.0025702, -0.0002302, 0.0002405
2: 0.0095618, 0.0108397, 0.0095841, 0.0108589, -0.0009205, 0.0008811
3: -0.0047912, -0.0034695, -0.0047681, -0.0034496, -0.0009521, 0.0009113
4: -0.0002810, 0.0011498, -0.0003025, 0.0011248, -0.0009865, 0.0010307
5: 0.0030252, 0.0043793, 0.0030489, 0.0043996, -0.0009754, 0.0009336
6: -0.0102970, -0.0049245, -0.0102031, -0.0048438, -0.0038699, 0.0037042
7: 0.0041501, 0.0114670, 0.0040402, 0.0113390, -0.0050447, 0.0052705
8: 0.9921373, 0.9972915, 0.9920598, 0.9972013, -0.0035536, 0.0037127
9: -0.0134286, -0.0087500, -0.0133468, -0.0086798, -0.0033701, 0.0032257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019177, upper bound: 0.0018245
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018548, upper bound: 0.0017678
time: 0.66 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063406, 0.0086374, 0.0062886, 0.0086712, -0.0016436, 0.0016648
1: 0.0022383, 0.0025702, 0.0022308, 0.0025750, -0.0002375, 0.0002405
2: 0.0095845, 0.0108543, 0.0095658, 0.0108831, -0.0009204, 0.0009087
3: -0.0047678, -0.0034544, -0.0047871, -0.0034247, -0.0009519, 0.0009398
4: -0.0002973, 0.0011244, -0.0003295, 0.0011453, -0.0010174, 0.0010305
5: 0.0030493, 0.0043947, 0.0030295, 0.0044252, -0.0009752, 0.0009628
6: -0.0102017, -0.0048633, -0.0102803, -0.0047424, -0.0038694, 0.0038202
7: 0.0040667, 0.0113372, 0.0039021, 0.0114441, -0.0052027, 0.0052698
8: 0.9920785, 0.9972000, 0.9919626, 0.9972753, -0.0036649, 0.0037122
9: -0.0133456, -0.0086967, -0.0134140, -0.0085915, -0.0033697, 0.0033268

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021275, upper bound: 0.0019876
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021275, upper bound: 0.0019884
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063670, 0.0086784, 0.0063067, 0.0086701, -0.0016330, 0.0017003
1: 0.0022421, 0.0025761, 0.0022334, 0.0025749, -0.0002359, 0.0002456
2: 0.0095618, 0.0108397, 0.0095664, 0.0108731, -0.0009400, 0.0009028
3: -0.0047912, -0.0034695, -0.0047864, -0.0034350, -0.0009722, 0.0009338
4: -0.0002810, 0.0011498, -0.0003184, 0.0011446, -0.0010109, 0.0010525
5: 0.0030252, 0.0043793, 0.0030301, 0.0044146, -0.0009960, 0.0009566
6: -0.0102970, -0.0049245, -0.0102777, -0.0047844, -0.0039520, 0.0037956
7: 0.0041501, 0.0114670, 0.0039592, 0.0114406, -0.0051692, 0.0053822
8: 0.9921373, 0.9972915, 0.9920028, 0.9972728, -0.0036413, 0.0037913
9: -0.0134286, -0.0087500, -0.0134118, -0.0086280, -0.0034415, 0.0033053

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019585, upper bound: 0.0018123
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019360, upper bound: 0.0017678
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063145, 0.0086696, 0.0063144, 0.0086391, -0.0016392, 0.0016691
1: 0.0022346, 0.0025748, 0.0022345, 0.0025704, -0.0002368, 0.0002411
2: 0.0095667, 0.0108687, 0.0095835, 0.0108688, -0.0009228, 0.0009063
3: -0.0047862, -0.0034395, -0.0047687, -0.0034394, -0.0009544, 0.0009373
4: -0.0003135, 0.0011444, -0.0003136, 0.0011255, -0.0010147, 0.0010332
5: 0.0030304, 0.0044100, 0.0030483, 0.0044101, -0.0009777, 0.0009602
6: -0.0102766, -0.0048027, -0.0102056, -0.0048024, -0.0038794, 0.0038099
7: 0.0039841, 0.0114391, 0.0039837, 0.0113425, -0.0051888, 0.0052833
8: 0.9920204, 0.9972718, 0.9920201, 0.9972038, -0.0036551, 0.0037217
9: -0.0134108, -0.0086439, -0.0133490, -0.0086437, -0.0033783, 0.0033179

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.89 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0020421
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020826, upper bound: 0.0020421
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063429, 0.0087087, 0.0063322, 0.0086380, -0.0016289, 0.0017007
1: 0.0022387, 0.0025805, 0.0022371, 0.0025702, -0.0002353, 0.0002457
2: 0.0095450, 0.0108530, 0.0095841, 0.0108589, -0.0009403, 0.0009006
3: -0.0048085, -0.0034557, -0.0047681, -0.0034496, -0.0009725, 0.0009314
4: -0.0002959, 0.0011686, -0.0003025, 0.0011248, -0.0010083, 0.0010528
5: 0.0030075, 0.0043934, 0.0030489, 0.0043996, -0.0009963, 0.0009542
6: -0.0103675, -0.0048687, -0.0102031, -0.0048438, -0.0039530, 0.0037861
7: 0.0040740, 0.0115629, 0.0040402, 0.0113390, -0.0051563, 0.0053836
8: 0.9920837, 0.9973591, 0.9920598, 0.9972013, -0.0036322, 0.0037923
9: -0.0134900, -0.0087014, -0.0133468, -0.0086798, -0.0034424, 0.0032971

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.74 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019177, upper bound: 0.0019045
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018548, upper bound: 0.0018494
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063145, 0.0086696, 0.0062886, 0.0086712, -0.0016422, 0.0016684
1: 0.0022346, 0.0025748, 0.0022308, 0.0025750, -0.0002372, 0.0002410
2: 0.0095667, 0.0108687, 0.0095658, 0.0108831, -0.0009224, 0.0009079
3: -0.0047862, -0.0034395, -0.0047871, -0.0034247, -0.0009540, 0.0009390
4: -0.0003135, 0.0011444, -0.0003295, 0.0011453, -0.0010165, 0.0010328
5: 0.0030304, 0.0044100, 0.0030295, 0.0044252, -0.0009773, 0.0009620
6: -0.0102766, -0.0048027, -0.0102803, -0.0047424, -0.0038778, 0.0038169
7: 0.0039841, 0.0114391, 0.0039021, 0.0114441, -0.0051983, 0.0052812
8: 0.9920204, 0.9972718, 0.9919626, 0.9972753, -0.0036618, 0.0037202
9: -0.0134108, -0.0086439, -0.0134140, -0.0085915, -0.0033770, 0.0033239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.71 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020829, upper bound: 0.0020421
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020829, upper bound: 0.0020420
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063429, 0.0087087, 0.0063067, 0.0086701, -0.0016320, 0.0017018
1: 0.0022387, 0.0025805, 0.0022334, 0.0025749, -0.0002358, 0.0002459
2: 0.0095450, 0.0108530, 0.0095664, 0.0108731, -0.0009409, 0.0009023
3: -0.0048085, -0.0034557, -0.0047864, -0.0034350, -0.0009731, 0.0009332
4: -0.0002959, 0.0011686, -0.0003184, 0.0011446, -0.0010102, 0.0010534
5: 0.0030075, 0.0043934, 0.0030301, 0.0044146, -0.0009969, 0.0009560
6: -0.0103675, -0.0048687, -0.0102777, -0.0047844, -0.0039554, 0.0037931
7: 0.0040740, 0.0115629, 0.0039592, 0.0114406, -0.0051659, 0.0053870
8: 0.9920837, 0.9973591, 0.9920028, 0.9972728, -0.0036390, 0.0037947
9: -0.0134900, -0.0087014, -0.0134118, -0.0086280, -0.0034446, 0.0033032

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019576, upper bound: 0.0019126
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0019269, upper bound: 0.0018785
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063145, 0.0086696, 0.0063581, 0.0088459, -0.0018904, 0.0016642
1: 0.0022346, 0.0025748, 0.0022409, 0.0026003, -0.0002731, 0.0002404
2: 0.0095667, 0.0108687, 0.0094692, 0.0108446, -0.0009201, 0.0010452
3: -0.0047862, -0.0034395, -0.0048870, -0.0034644, -0.0009516, 0.0010810
4: -0.0003135, 0.0011444, -0.0002865, 0.0012535, -0.0011702, 0.0010302
5: 0.0030304, 0.0044100, 0.0029272, 0.0043845, -0.0009749, 0.0011074
6: -0.0102766, -0.0048027, -0.0106862, -0.0049040, -0.0038682, 0.0043939
7: 0.0039841, 0.0114391, 0.0041221, 0.0119970, -0.0059841, 0.0052681
8: 0.9920204, 0.9972718, 0.9921175, 0.9976648, -0.0042153, 0.0037110
9: -0.0134108, -0.0086439, -0.0137675, -0.0087321, -0.0033686, 0.0038264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020267, upper bound: 0.0020785
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020267, upper bound: 0.0020789
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063429, 0.0087087, 0.0063753, 0.0088447, -0.0018801, 0.0016957
1: 0.0022387, 0.0025805, 0.0022434, 0.0026001, -0.0002716, 0.0002450
2: 0.0095450, 0.0108530, 0.0094698, 0.0108351, -0.0009375, 0.0010394
3: -0.0048085, -0.0034557, -0.0048863, -0.0034743, -0.0009696, 0.0010750
4: -0.0002959, 0.0011686, -0.0002758, 0.0012528, -0.0011638, 0.0010496
5: 0.0030075, 0.0043934, 0.0029278, 0.0043744, -0.0009933, 0.0011013
6: -0.0103675, -0.0048687, -0.0106836, -0.0049440, -0.0039412, 0.0043698
7: 0.0040740, 0.0115629, 0.0041766, 0.0119934, -0.0059512, 0.0053676
8: 0.9920837, 0.9973591, 0.9921559, 0.9976622, -0.0041922, 0.0037810
9: -0.0134900, -0.0087014, -0.0137652, -0.0087670, -0.0034322, 0.0038054

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016803, upper bound: 0.0018002
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015101, upper bound: 0.0016676
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063145, 0.0086696, 0.0063205, 0.0088829, -0.0018955, 0.0016669
1: 0.0022346, 0.0025748, 0.0022354, 0.0026056, -0.0002738, 0.0002408
2: 0.0095667, 0.0108687, 0.0094488, 0.0108654, -0.0009216, 0.0010480
3: -0.0047862, -0.0034395, -0.0049081, -0.0034429, -0.0009531, 0.0010839
4: -0.0003135, 0.0011444, -0.0003098, 0.0012764, -0.0011734, 0.0010318
5: 0.0030304, 0.0044100, 0.0029055, 0.0044065, -0.0009765, 0.0011104
6: -0.0102766, -0.0048027, -0.0107722, -0.0048166, -0.0038743, 0.0044057
7: 0.0039841, 0.0114391, 0.0040031, 0.0121141, -0.0060002, 0.0052765
8: 0.9920204, 0.9972718, 0.9920337, 0.9977473, -0.0042267, 0.0037169
9: -0.0134108, -0.0086439, -0.0138425, -0.0086560, -0.0033739, 0.0038367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020295, upper bound: 0.0020785
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020295, upper bound: 0.0020789
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063429, 0.0087087, 0.0063378, 0.0088817, -0.0018852, 0.0017000
1: 0.0022387, 0.0025805, 0.0022379, 0.0026055, -0.0002724, 0.0002456
2: 0.0095450, 0.0108530, 0.0094494, 0.0108559, -0.0009399, 0.0010423
3: -0.0048085, -0.0034557, -0.0049075, -0.0034528, -0.0009721, 0.0010780
4: -0.0002959, 0.0011686, -0.0002991, 0.0012757, -0.0011670, 0.0010524
5: 0.0030075, 0.0043934, 0.0029061, 0.0043964, -0.0009959, 0.0011043
6: -0.0103675, -0.0048687, -0.0107696, -0.0048567, -0.0039514, 0.0043817
7: 0.0040740, 0.0115629, 0.0040577, 0.0121105, -0.0059675, 0.0053814
8: 0.9920837, 0.9973591, 0.9920722, 0.9977447, -0.0042037, 0.0037908
9: -0.0134900, -0.0087014, -0.0138402, -0.0086910, -0.0034410, 0.0038158

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017630, upper bound: 0.0018268
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016793, upper bound: 0.0017362
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063849, 0.0088442, 0.0063144, 0.0086391, -0.0015991, 0.0018808
1: 0.0022447, 0.0026000, 0.0022345, 0.0025704, -0.0002310, 0.0002717
2: 0.0094701, 0.0108298, 0.0095835, 0.0108688, -0.0010398, 0.0008841
3: -0.0048860, -0.0034798, -0.0047687, -0.0034394, -0.0010755, 0.0009144
4: -0.0002699, 0.0012524, -0.0003136, 0.0011255, -0.0009899, 0.0011642
5: 0.0029281, 0.0043688, 0.0030483, 0.0044101, -0.0011018, 0.0009367
6: -0.0106824, -0.0049663, -0.0102056, -0.0048024, -0.0043715, 0.0037167
7: 0.0042070, 0.0119918, 0.0039837, 0.0113425, -0.0050619, 0.0059536
8: 0.9921774, 0.9976611, 0.9920201, 0.9972038, -0.0035657, 0.0041938
9: -0.0137642, -0.0087864, -0.0133490, -0.0086437, -0.0038069, 0.0032367

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019545
time: 0.80 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019545
time: 0.83 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064098, 0.0088856, 0.0063322, 0.0086380, -0.0015894, 0.0019134
1: 0.0022483, 0.0026060, 0.0022371, 0.0025702, -0.0002296, 0.0002764
2: 0.0094472, 0.0108161, 0.0095841, 0.0108589, -0.0010579, 0.0008788
3: -0.0049097, -0.0034940, -0.0047681, -0.0034496, -0.0010941, 0.0009089
4: -0.0002545, 0.0012781, -0.0003025, 0.0011248, -0.0009839, 0.0011845
5: 0.0029039, 0.0043542, 0.0030489, 0.0043996, -0.0011209, 0.0009311
6: -0.0107786, -0.0050241, -0.0102031, -0.0048438, -0.0044474, 0.0036943
7: 0.0042856, 0.0121228, 0.0040402, 0.0113390, -0.0050313, 0.0060569
8: 0.9922327, 0.9977534, 0.9920598, 0.9972013, -0.0035442, 0.0042666
9: -0.0138480, -0.0088367, -0.0133468, -0.0086798, -0.0038730, 0.0032172

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017787, upper bound: 0.0015845
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016495, upper bound: 0.0014521
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063849, 0.0088442, 0.0062886, 0.0086712, -0.0016384, 0.0019160
1: 0.0022447, 0.0026000, 0.0022308, 0.0025750, -0.0002367, 0.0002768
2: 0.0094701, 0.0108298, 0.0095658, 0.0108831, -0.0010593, 0.0009058
3: -0.0048860, -0.0034798, -0.0047871, -0.0034247, -0.0010956, 0.0009369
4: -0.0002699, 0.0012524, -0.0003295, 0.0011453, -0.0010142, 0.0011860
5: 0.0029281, 0.0043688, 0.0030295, 0.0044252, -0.0011224, 0.0009598
6: -0.0106824, -0.0049663, -0.0102803, -0.0047424, -0.0044533, 0.0038081
7: 0.0042070, 0.0119918, 0.0039021, 0.0114441, -0.0051864, 0.0060650
8: 0.9921774, 0.9976611, 0.9919626, 0.9972753, -0.0036534, 0.0042723
9: -0.0137642, -0.0087864, -0.0134140, -0.0085915, -0.0038781, 0.0033163

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021629, upper bound: 0.0019419
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021629, upper bound: 0.0019419
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064098, 0.0088856, 0.0063067, 0.0086701, -0.0016288, 0.0019487
1: 0.0022483, 0.0026060, 0.0022334, 0.0025749, -0.0002353, 0.0002815
2: 0.0094472, 0.0108161, 0.0095664, 0.0108731, -0.0010774, 0.0009005
3: -0.0049097, -0.0034940, -0.0047864, -0.0034350, -0.0011143, 0.0009313
4: -0.0002545, 0.0012781, -0.0003184, 0.0011446, -0.0010082, 0.0012063
5: 0.0029039, 0.0043542, 0.0030301, 0.0044146, -0.0011416, 0.0009541
6: -0.0107786, -0.0050241, -0.0102777, -0.0047844, -0.0045294, 0.0037857
7: 0.0042856, 0.0121228, 0.0039592, 0.0114406, -0.0051558, 0.0061686
8: 0.9922327, 0.9977534, 0.9920028, 0.9972728, -0.0036319, 0.0043453
9: -0.0138480, -0.0088367, -0.0134118, -0.0086280, -0.0039444, 0.0032968

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018276, upper bound: 0.0015810
time: 0.78 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017550, upper bound: 0.0014521
time: 0.65 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063471, 0.0088813, 0.0063144, 0.0086391, -0.0016365, 0.0019224
1: 0.0022393, 0.0026054, 0.0022345, 0.0025704, -0.0002364, 0.0002777
2: 0.0094496, 0.0108507, 0.0095835, 0.0108688, -0.0010629, 0.0009048
3: -0.0049072, -0.0034581, -0.0047687, -0.0034394, -0.0010993, 0.0009357
4: -0.0002933, 0.0012754, -0.0003136, 0.0011255, -0.0010130, 0.0011900
5: 0.0029064, 0.0043909, 0.0030483, 0.0044101, -0.0011262, 0.0009586
6: -0.0107685, -0.0048784, -0.0102056, -0.0048024, -0.0044683, 0.0038036
7: 0.0040873, 0.0121090, 0.0039837, 0.0113425, -0.0051801, 0.0060854
8: 0.9920931, 0.9977437, 0.9920201, 0.9972038, -0.0036490, 0.0042867
9: -0.0138392, -0.0087099, -0.0133490, -0.0086437, -0.0038912, 0.0033123

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063722, 0.0089214, 0.0063322, 0.0086380, -0.0016271, 0.0019528
1: 0.0022429, 0.0026112, 0.0022371, 0.0025702, -0.0002351, 0.0002821
2: 0.0094274, 0.0108368, 0.0095841, 0.0108589, -0.0010796, 0.0008996
3: -0.0049302, -0.0034725, -0.0047681, -0.0034496, -0.0011166, 0.0009304
4: -0.0002778, 0.0013002, -0.0003025, 0.0011248, -0.0010072, 0.0012088
5: 0.0028829, 0.0043762, 0.0030489, 0.0043996, -0.0011439, 0.0009531
6: -0.0108618, -0.0049367, -0.0102031, -0.0048438, -0.0045387, 0.0037818
7: 0.0041667, 0.0122362, 0.0040402, 0.0113390, -0.0051505, 0.0061814
8: 0.9921490, 0.9978333, 0.9920598, 0.9972013, -0.0036281, 0.0043543
9: -0.0139205, -0.0087606, -0.0133468, -0.0086798, -0.0039525, 0.0032934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017787, upper bound: 0.0017016
time: 0.75 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016495, upper bound: 0.0015630
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063471, 0.0088813, 0.0062886, 0.0086712, -0.0016415, 0.0019216
1: 0.0022393, 0.0026054, 0.0022308, 0.0025750, -0.0002371, 0.0002776
2: 0.0094496, 0.0108507, 0.0095658, 0.0108831, -0.0010624, 0.0009075
3: -0.0049072, -0.0034581, -0.0047871, -0.0034247, -0.0010988, 0.0009386
4: -0.0002933, 0.0012754, -0.0003295, 0.0011453, -0.0010161, 0.0011895
5: 0.0029064, 0.0043909, 0.0030295, 0.0044252, -0.0011257, 0.0009616
6: -0.0107685, -0.0048784, -0.0102803, -0.0047424, -0.0044664, 0.0038153
7: 0.0040873, 0.0121090, 0.0039021, 0.0114441, -0.0051960, 0.0060828
8: 0.9920931, 0.9977437, 0.9919626, 0.9972753, -0.0036602, 0.0042849
9: -0.0138392, -0.0087099, -0.0134140, -0.0085915, -0.0038895, 0.0033225

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021125, upper bound: 0.0019896
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021125, upper bound: 0.0019896
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063722, 0.0089214, 0.0063067, 0.0086701, -0.0016306, 0.0019533
1: 0.0022429, 0.0026112, 0.0022334, 0.0025749, -0.0002356, 0.0002822
2: 0.0094274, 0.0108368, 0.0095664, 0.0108731, -0.0010799, 0.0009015
3: -0.0049302, -0.0034725, -0.0047864, -0.0034350, -0.0011169, 0.0009324
4: -0.0002778, 0.0013002, -0.0003184, 0.0011446, -0.0010094, 0.0012091
5: 0.0028829, 0.0043762, 0.0030301, 0.0044146, -0.0011442, 0.0009552
6: -0.0108618, -0.0049367, -0.0102777, -0.0047844, -0.0045400, 0.0037900
7: 0.0041667, 0.0122362, 0.0039592, 0.0114406, -0.0051616, 0.0061830
8: 0.9921490, 0.9978333, 0.9920028, 0.9972728, -0.0036360, 0.0043555
9: -0.0139205, -0.0087606, -0.0134118, -0.0086280, -0.0039536, 0.0033005

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018479, upper bound: 0.0017286
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017749, upper bound: 0.0016433
time: 0.70 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063849, 0.0088442, 0.0063581, 0.0088459, -0.0016911, 0.0017168
1: 0.0022447, 0.0026000, 0.0022409, 0.0026003, -0.0002443, 0.0002480
2: 0.0094701, 0.0108298, 0.0094692, 0.0108446, -0.0009492, 0.0009350
3: -0.0048860, -0.0034798, -0.0048870, -0.0034644, -0.0009817, 0.0009670
4: -0.0002699, 0.0012524, -0.0002865, 0.0012535, -0.0010468, 0.0010627
5: 0.0029281, 0.0043688, 0.0029272, 0.0043845, -0.0010057, 0.0009907
6: -0.0106824, -0.0049663, -0.0106862, -0.0049040, -0.0039902, 0.0039306
7: 0.0042070, 0.0119918, 0.0041221, 0.0119970, -0.0053532, 0.0054344
8: 0.9921774, 0.9976611, 0.9921175, 0.9976648, -0.0037709, 0.0038281
9: -0.0137642, -0.0087864, -0.0137675, -0.0087321, -0.0034749, 0.0034230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019545
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021184, upper bound: 0.0019545
time: 0.74 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0064098, 0.0088856, 0.0063753, 0.0088447, -0.0016818, 0.0017488
1: 0.0022483, 0.0026060, 0.0022434, 0.0026001, -0.0002430, 0.0002527
2: 0.0094472, 0.0108161, 0.0094698, 0.0108351, -0.0009669, 0.0009298
3: -0.0049097, -0.0034940, -0.0048863, -0.0034743, -0.0010000, 0.0009617
4: -0.0002545, 0.0012781, -0.0002758, 0.0012528, -0.0010411, 0.0010826
5: 0.0029039, 0.0043542, 0.0029278, 0.0043744, -0.0010245, 0.0009852
6: -0.0107786, -0.0050241, -0.0106836, -0.0049440, -0.0040648, 0.0039090
7: 0.0042856, 0.0121228, 0.0041766, 0.0119934, -0.0053237, 0.0055359
8: 0.9922327, 0.9977534, 0.9921559, 0.9976622, -0.0037501, 0.0038996
9: -0.0138480, -0.0088367, -0.0137652, -0.0087670, -0.0035398, 0.0034041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016822, upper bound: 0.0015745
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014954, upper bound: 0.0014338
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063849, 0.0088442, 0.0063205, 0.0088829, -0.0017315, 0.0017540
1: 0.0022447, 0.0026000, 0.0022354, 0.0026056, -0.0002501, 0.0002534
2: 0.0094701, 0.0108298, 0.0094488, 0.0108654, -0.0009697, 0.0009573
3: -0.0048860, -0.0034798, -0.0049081, -0.0034429, -0.0010030, 0.0009901
4: -0.0002699, 0.0012524, -0.0003098, 0.0012764, -0.0010718, 0.0010858
5: 0.0029281, 0.0043688, 0.0029055, 0.0044065, -0.0010275, 0.0010143
6: -0.0106824, -0.0049663, -0.0107722, -0.0048166, -0.0040768, 0.0040244
7: 0.0042070, 0.0119918, 0.0040031, 0.0121141, -0.0054809, 0.0055523
8: 0.9921774, 0.9976611, 0.9920337, 0.9977473, -0.0038609, 0.0039111
9: -0.0137642, -0.0087864, -0.0138425, -0.0086560, -0.0035503, 0.0035046

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021629, upper bound: 0.0019419
time: 0.77 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021629, upper bound: 0.0019419
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0064098, 0.0088856, 0.0063378, 0.0088817, -0.0017222, 0.0017862
1: 0.0022483, 0.0026060, 0.0022379, 0.0026055, -0.0002488, 0.0002581
2: 0.0094472, 0.0108161, 0.0094494, 0.0108559, -0.0009876, 0.0009521
3: -0.0049097, -0.0034940, -0.0049075, -0.0034528, -0.0010214, 0.0009848
4: -0.0002545, 0.0012781, -0.0002991, 0.0012757, -0.0010661, 0.0011057
5: 0.0029039, 0.0043542, 0.0029061, 0.0043964, -0.0010464, 0.0010088
6: -0.0107786, -0.0050241, -0.0107696, -0.0048567, -0.0041517, 0.0040028
7: 0.0042856, 0.0121228, 0.0040577, 0.0121105, -0.0054515, 0.0056542
8: 0.9922327, 0.9977534, 0.9920722, 0.9977447, -0.0038401, 0.0039829
9: -0.0138480, -0.0088367, -0.0138402, -0.0086910, -0.0036155, 0.0034858

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017405, upper bound: 0.0015706
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016233, upper bound: 0.0014338
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063471, 0.0088813, 0.0063581, 0.0088459, -0.0017284, 0.0017572
1: 0.0022393, 0.0026054, 0.0022409, 0.0026003, -0.0002497, 0.0002539
2: 0.0094496, 0.0108507, 0.0094692, 0.0108446, -0.0009715, 0.0009556
3: -0.0049072, -0.0034581, -0.0048870, -0.0034644, -0.0010048, 0.0009883
4: -0.0002933, 0.0012754, -0.0002865, 0.0012535, -0.0010699, 0.0010877
5: 0.0029064, 0.0043909, 0.0029272, 0.0043845, -0.0010293, 0.0010125
6: -0.0107685, -0.0048784, -0.0106862, -0.0049040, -0.0040841, 0.0040174
7: 0.0040873, 0.0121090, 0.0041221, 0.0119970, -0.0054713, 0.0055622
8: 0.9920931, 0.9977437, 0.9921175, 0.9976648, -0.0038541, 0.0039182
9: -0.0138392, -0.0087099, -0.0137675, -0.0087321, -0.0035566, 0.0034985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021109, upper bound: 0.0019896
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063722, 0.0089214, 0.0063753, 0.0088447, -0.0017191, 0.0017889
1: 0.0022429, 0.0026112, 0.0022434, 0.0026001, -0.0002484, 0.0002584
2: 0.0094274, 0.0108368, 0.0094698, 0.0108351, -0.0009891, 0.0009504
3: -0.0049302, -0.0034725, -0.0048863, -0.0034743, -0.0010229, 0.0009830
4: -0.0002778, 0.0013002, -0.0002758, 0.0012528, -0.0010641, 0.0011074
5: 0.0028829, 0.0043762, 0.0029278, 0.0043744, -0.0010480, 0.0010070
6: -0.0108618, -0.0049367, -0.0106836, -0.0049440, -0.0041580, 0.0039956
7: 0.0041667, 0.0122362, 0.0041766, 0.0119934, -0.0054416, 0.0056628
8: 0.9921490, 0.9978333, 0.9921559, 0.9976622, -0.0038332, 0.0039890
9: -0.0139205, -0.0087606, -0.0137652, -0.0087670, -0.0036209, 0.0034795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016825, upper bound: 0.0017009
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014954, upper bound: 0.0015472
time: 0.77 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063471, 0.0088813, 0.0063205, 0.0088829, -0.0017341, 0.0017599
1: 0.0022393, 0.0026054, 0.0022354, 0.0026056, -0.0002505, 0.0002543
2: 0.0094496, 0.0108507, 0.0094488, 0.0108654, -0.0009730, 0.0009587
3: -0.0049072, -0.0034581, -0.0049081, -0.0034429, -0.0010063, 0.0009916
4: -0.0002933, 0.0012754, -0.0003098, 0.0012764, -0.0010734, 0.0010894
5: 0.0029064, 0.0043909, 0.0029055, 0.0044065, -0.0010309, 0.0010158
6: -0.0107685, -0.0048784, -0.0107722, -0.0048166, -0.0040904, 0.0040304
7: 0.0040873, 0.0121090, 0.0040031, 0.0121141, -0.0054891, 0.0055708
8: 0.9920931, 0.9977437, 0.9920337, 0.9977473, -0.0038667, 0.0039242
9: -0.0138392, -0.0087099, -0.0138425, -0.0086560, -0.0035621, 0.0035099

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021125, upper bound: 0.0019896
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021125, upper bound: 0.0019896
time: 0.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063722, 0.0089214, 0.0063378, 0.0088817, -0.0017247, 0.0017923
1: 0.0022429, 0.0026112, 0.0022379, 0.0026055, -0.0002492, 0.0002589
2: 0.0094274, 0.0108368, 0.0094494, 0.0108559, -0.0009909, 0.0009536
3: -0.0049302, -0.0034725, -0.0049075, -0.0034528, -0.0010249, 0.0009862
4: -0.0002778, 0.0013002, -0.0002991, 0.0012757, -0.0010676, 0.0011095
5: 0.0028829, 0.0043762, 0.0029061, 0.0043964, -0.0010499, 0.0010103
6: -0.0108618, -0.0049367, -0.0107696, -0.0048567, -0.0041658, 0.0040088
7: 0.0041667, 0.0122362, 0.0040577, 0.0121105, -0.0054596, 0.0056735
8: 0.9921490, 0.9978333, 0.9920722, 0.9977447, -0.0038459, 0.0039965
9: -0.0139205, -0.0087606, -0.0138402, -0.0086910, -0.0036278, 0.0034910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017636, upper bound: 0.0017270
time: 0.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016598, upper bound: 0.0016240
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063406, 0.0086374, 0.0061747, 0.0085711, -0.0016079, 0.0018383
1: 0.0022383, 0.0025702, 0.0022144, 0.0025606, -0.0002323, 0.0002656
2: 0.0095845, 0.0108543, 0.0096211, 0.0109460, -0.0010163, 0.0008890
3: -0.0047678, -0.0034544, -0.0047298, -0.0033595, -0.0010511, 0.0009194
4: -0.0002973, 0.0011244, -0.0004001, 0.0010834, -0.0009953, 0.0011379
5: 0.0030493, 0.0043947, 0.0030881, 0.0044919, -0.0010769, 0.0009419
6: -0.0102017, -0.0048633, -0.0100476, -0.0044776, -0.0042726, 0.0037372
7: 0.0040667, 0.0113372, 0.0035415, 0.0111272, -0.0050897, 0.0058190
8: 0.9920785, 0.9972000, 0.9917086, 0.9970521, -0.0035853, 0.0040990
9: -0.0133456, -0.0086967, -0.0132114, -0.0083608, -0.0037208, 0.0032545

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020712, upper bound: 0.0019775
time: 0.74 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020712, upper bound: 0.0019776
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063670, 0.0086784, 0.0061933, 0.0085702, -0.0015974, 0.0018737
1: 0.0022421, 0.0025761, 0.0022170, 0.0025604, -0.0002308, 0.0002707
2: 0.0095618, 0.0108397, 0.0096216, 0.0109358, -0.0010359, 0.0008832
3: -0.0047912, -0.0034695, -0.0047293, -0.0033702, -0.0010714, 0.0009134
4: -0.0002810, 0.0011498, -0.0003885, 0.0010828, -0.0009888, 0.0011598
5: 0.0030252, 0.0043793, 0.0030887, 0.0044810, -0.0010976, 0.0009358
6: -0.0102970, -0.0049245, -0.0100454, -0.0045209, -0.0043550, 0.0037129
7: 0.0041501, 0.0114670, 0.0036003, 0.0111243, -0.0050566, 0.0059311
8: 0.9921373, 0.9972915, 0.9917500, 0.9970500, -0.0035620, 0.0041780
9: -0.0134286, -0.0087500, -0.0132095, -0.0083985, -0.0037925, 0.0032333

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017964, upper bound: 0.0017182
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0016240
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063406, 0.0086374, 0.0061549, 0.0085998, -0.0016304, 0.0018517
1: 0.0022383, 0.0025702, 0.0022115, 0.0025647, -0.0002355, 0.0002675
2: 0.0095845, 0.0108543, 0.0096053, 0.0109570, -0.0010237, 0.0009014
3: -0.0047678, -0.0034544, -0.0047463, -0.0033482, -0.0010588, 0.0009323
4: -0.0002973, 0.0011244, -0.0004123, 0.0011011, -0.0010092, 0.0011462
5: 0.0030493, 0.0043947, 0.0030713, 0.0045035, -0.0010847, 0.0009551
6: -0.0102017, -0.0048633, -0.0101143, -0.0044317, -0.0043037, 0.0037895
7: 0.0040667, 0.0113372, 0.0034788, 0.0112181, -0.0051609, 0.0058613
8: 0.9920785, 0.9972000, 0.9916644, 0.9971162, -0.0036355, 0.0041288
9: -0.0133456, -0.0086967, -0.0132695, -0.0083208, -0.0037479, 0.0033000

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021010, upper bound: 0.0019606
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021010, upper bound: 0.0019614
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063670, 0.0086784, 0.0061731, 0.0085988, -0.0016199, 0.0018864
1: 0.0022421, 0.0025761, 0.0022141, 0.0025646, -0.0002340, 0.0002725
2: 0.0095618, 0.0108397, 0.0096058, 0.0109469, -0.0010429, 0.0008956
3: -0.0047912, -0.0034695, -0.0047457, -0.0033587, -0.0010786, 0.0009263
4: -0.0002810, 0.0011498, -0.0004010, 0.0011005, -0.0010027, 0.0011677
5: 0.0030252, 0.0043793, 0.0030719, 0.0044928, -0.0011050, 0.0009489
6: -0.0102970, -0.0049245, -0.0101120, -0.0044741, -0.0043844, 0.0037651
7: 0.0041501, 0.0114670, 0.0035366, 0.0112150, -0.0051277, 0.0059712
8: 0.9921373, 0.9972915, 0.9917052, 0.9971139, -0.0036121, 0.0042062
9: -0.0134286, -0.0087500, -0.0132675, -0.0083577, -0.0038181, 0.0032788

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018327, upper bound: 0.0017104
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017629, upper bound: 0.0016239
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0063145, 0.0086696, 0.0061747, 0.0085711, -0.0016428, 0.0018777
1: 0.0022346, 0.0025748, 0.0022144, 0.0025606, -0.0002373, 0.0002713
2: 0.0095667, 0.0108687, 0.0096211, 0.0109460, -0.0010381, 0.0009083
3: -0.0047862, -0.0034395, -0.0047298, -0.0033595, -0.0010737, 0.0009394
4: -0.0003135, 0.0011444, -0.0004001, 0.0010834, -0.0010169, 0.0011623
5: 0.0030304, 0.0044100, 0.0030881, 0.0044919, -0.0011000, 0.0009624
6: -0.0102766, -0.0048027, -0.0100476, -0.0044776, -0.0043643, 0.0038183
7: 0.0039841, 0.0114391, 0.0035415, 0.0111272, -0.0052002, 0.0059439
8: 0.9920204, 0.9972718, 0.9917086, 0.9970521, -0.0036632, 0.0041870
9: -0.0134108, -0.0086439, -0.0132114, -0.0083608, -0.0038007, 0.0033252

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020663, upper bound: 0.0020131
time: 0.82 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020663, upper bound: 0.0020132
time: 0.76 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0063429, 0.0087087, 0.0061933, 0.0085702, -0.0016327, 0.0019094
1: 0.0022387, 0.0025805, 0.0022170, 0.0025604, -0.0002359, 0.0002759
2: 0.0095450, 0.0108530, 0.0096216, 0.0109358, -0.0010557, 0.0009027
3: -0.0048085, -0.0034557, -0.0047293, -0.0033702, -0.0010918, 0.0009336
4: -0.0002959, 0.0011686, -0.0003885, 0.0010828, -0.0010107, 0.0011820
5: 0.0030075, 0.0043934, 0.0030887, 0.0044810, -0.0011185, 0.0009564
6: -0.0103675, -0.0048687, -0.0100454, -0.0045209, -0.0044380, 0.0037948
7: 0.0040740, 0.0115629, 0.0036003, 0.0111243, -0.0051682, 0.0060442
8: 0.9920837, 0.9973591, 0.9917500, 0.9970500, -0.0036406, 0.0042577
9: -0.0134900, -0.0087014, -0.0132095, -0.0083985, -0.0038648, 0.0033047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017964, upper bound: 0.0018118
time: 0.68 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0016828, upper bound: 0.0017212
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0063145, 0.0086696, 0.0061549, 0.0085998, -0.0016422, 0.0018673
1: 0.0022346, 0.0025748, 0.0022115, 0.0025647, -0.0002373, 0.0002698
2: 0.0095667, 0.0108687, 0.0096053, 0.0109570, -0.0010324, 0.0009079
3: -0.0047862, -0.0034395, -0.0047463, -0.0033482, -0.0010678, 0.0009390
4: -0.0003135, 0.0011444, -0.0004123, 0.0011011, -0.0010166, 0.0011559
5: 0.0030304, 0.0044100, 0.0030713, 0.0045035, -0.0010939, 0.0009620
6: -0.0102766, -0.0048027, -0.0101143, -0.0044317, -0.0043402, 0.0038170
7: 0.0039841, 0.0114391, 0.0034788, 0.0112181, -0.0051984, 0.0059109
8: 0.9920204, 0.9972718, 0.9916644, 0.9971162, -0.0036619, 0.0041638
9: -0.0134108, -0.0086439, -0.0132695, -0.0083208, -0.0037796, 0.0033240

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 219

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020663, upper bound: 0.0020131
time: 0.72 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020663, upper bound: 0.0020132
time: 0.71 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0063429, 0.0087087, 0.0061731, 0.0085988, -0.0016321, 0.0019006
1: 0.0022387, 0.0025805, 0.0022141, 0.0025646, -0.0002358, 0.0002746
2: 0.0095450, 0.0108530, 0.0096058, 0.0109469, -0.0010508, 0.0009023
3: -0.0048085, -0.0034557, -0.0047457, -0.0033587, -0.0010868, 0.0009332
4: -0.0002959, 0.0011686, -0.0004010, 0.0011005, -0.0010103, 0.0011765
5: 0.0030075, 0.0043934, 0.0030719, 0.0044928, -0.0011134, 0.0009561
6: -0.0103675, -0.0048687, -0.0101120, -0.0044741, -0.0044175, 0.0037934
7: 0.0040740, 0.0115629, 0.0035366, 0.0112150, -0.0051663, 0.0060163
8: 0.9920837, 0.9973591, 0.9917052, 0.9971139, -0.0036393, 0.0042380
9: -0.0134900, -0.0087014, -0.0132675, -0.0083577, -0.0038470, 0.0033035

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018344, upper bound: 0.0018144
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017621, upper bound: 0.0017419
time: 0.80 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0061994, 0.0085696, 0.0063144, 0.0086391, -0.0018117, 0.0016334
1: 0.0022179, 0.0025604, 0.0022345, 0.0025704, -0.0002617, 0.0002360
2: 0.0096219, 0.0109324, 0.0095835, 0.0108688, -0.0009031, 0.0010017
3: -0.0047290, -0.0033737, -0.0047687, -0.0034394, -0.0009340, 0.0010360
4: -0.0003847, 0.0010825, -0.0003136, 0.0011255, -0.0011215, 0.0010111
5: 0.0030890, 0.0044774, 0.0030483, 0.0044101, -0.0009568, 0.0010613
6: -0.0100442, -0.0045352, -0.0102056, -0.0048024, -0.0037965, 0.0042110
7: 0.0036198, 0.0111226, 0.0039837, 0.0113425, -0.0057350, 0.0051705
8: 0.9917637, 0.9970489, 0.9920201, 0.9972038, -0.0040398, 0.0036422
9: -0.0132084, -0.0084109, -0.0133490, -0.0086437, -0.0033061, 0.0036671

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020693, upper bound: 0.0019902
time: 0.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020693, upper bound: 0.0019902
time: 0.75 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062289, 0.0086090, 0.0063322, 0.0086380, -0.0018034, 0.0016664
1: 0.0022222, 0.0025660, 0.0022371, 0.0025702, -0.0002605, 0.0002407
2: 0.0096002, 0.0109160, 0.0095841, 0.0108589, -0.0009213, 0.0009970
3: -0.0047515, -0.0033906, -0.0047681, -0.0034496, -0.0009528, 0.0010312
4: -0.0003665, 0.0011068, -0.0003025, 0.0011248, -0.0011163, 0.0010315
5: 0.0030659, 0.0044602, 0.0030489, 0.0043996, -0.0009762, 0.0010564
6: -0.0101356, -0.0046037, -0.0102031, -0.0048438, -0.0038731, 0.0041916
7: 0.0037132, 0.0112471, 0.0040402, 0.0113390, -0.0057085, 0.0052748
8: 0.9918296, 0.9971366, 0.9920598, 0.9972013, -0.0040212, 0.0037157
9: -0.0132880, -0.0084707, -0.0133468, -0.0086798, -0.0033729, 0.0036502

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018182, upper bound: 0.0016850
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017116, upper bound: 0.0015873
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0061994, 0.0085696, 0.0062886, 0.0086712, -0.0018511, 0.0016686
1: 0.0022179, 0.0025604, 0.0022308, 0.0025750, -0.0002674, 0.0002411
2: 0.0096219, 0.0109324, 0.0095658, 0.0108831, -0.0009225, 0.0010234
3: -0.0047290, -0.0033737, -0.0047871, -0.0034247, -0.0009541, 0.0010585
4: -0.0003847, 0.0010825, -0.0003295, 0.0011453, -0.0011458, 0.0010329
5: 0.0030890, 0.0044774, 0.0030295, 0.0044252, -0.0009775, 0.0010843
6: -0.0100442, -0.0045352, -0.0102803, -0.0047424, -0.0038783, 0.0043024
7: 0.0036198, 0.0111226, 0.0039021, 0.0114441, -0.0058595, 0.0052818
8: 0.9917637, 0.9970489, 0.9919626, 0.9972753, -0.0041275, 0.0037206
9: -0.0132084, -0.0084109, -0.0134140, -0.0085915, -0.0033774, 0.0037467

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.67 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021036, upper bound: 0.0019792
time: 0.73 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021036, upper bound: 0.0019793
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062289, 0.0086090, 0.0063067, 0.0086701, -0.0018427, 0.0017017
1: 0.0022222, 0.0025660, 0.0022334, 0.0025749, -0.0002662, 0.0002458
2: 0.0096002, 0.0109160, 0.0095664, 0.0108731, -0.0009408, 0.0010188
3: -0.0047515, -0.0033906, -0.0047864, -0.0034350, -0.0009730, 0.0010537
4: -0.0003665, 0.0011068, -0.0003184, 0.0011446, -0.0011407, 0.0010534
5: 0.0030659, 0.0044602, 0.0030301, 0.0044146, -0.0009968, 0.0010795
6: -0.0101356, -0.0046037, -0.0102777, -0.0047844, -0.0039551, 0.0042830
7: 0.0037132, 0.0112471, 0.0039592, 0.0114406, -0.0058330, 0.0053865
8: 0.9918296, 0.9971366, 0.9920028, 0.9972728, -0.0041089, 0.0037944
9: -0.0132880, -0.0084707, -0.0134118, -0.0086280, -0.0034443, 0.0037298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018657, upper bound: 0.0016791
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018146, upper bound: 0.0015873
time: 0.78 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062092, 0.0086363, 0.0063322, 0.0086380, -0.0018165, 0.0016868
1: 0.0022194, 0.0025700, 0.0022371, 0.0025702, -0.0002624, 0.0002437
2: 0.0095851, 0.0109269, 0.0095841, 0.0108589, -0.0009326, 0.0010043
3: -0.0047671, -0.0033793, -0.0047681, -0.0034496, -0.0009646, 0.0010387
4: -0.0003787, 0.0011237, -0.0003025, 0.0011248, -0.0011244, 0.0010442
5: 0.0030499, 0.0044717, 0.0030489, 0.0043996, -0.0009882, 0.0010641
6: -0.0101990, -0.0045580, -0.0102031, -0.0048438, -0.0039207, 0.0042220
7: 0.0036509, 0.0113335, 0.0040402, 0.0113390, -0.0057500, 0.0053397
8: 0.9917856, 0.9971974, 0.9920598, 0.9972013, -0.0040504, 0.0037614
9: -0.0133433, -0.0084308, -0.0133468, -0.0086798, -0.0034143, 0.0036767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018182, upper bound: 0.0017813
time: 0.67 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017101, upper bound: 0.0016765
time: 0.73 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062092, 0.0086363, 0.0063067, 0.0086701, -0.0018317, 0.0017002
1: 0.0022194, 0.0025700, 0.0022334, 0.0025749, -0.0002646, 0.0002456
2: 0.0095851, 0.0109269, 0.0095664, 0.0108731, -0.0009400, 0.0010127
3: -0.0047671, -0.0033793, -0.0047864, -0.0034350, -0.0009722, 0.0010474
4: -0.0003787, 0.0011237, -0.0003184, 0.0011446, -0.0011339, 0.0010525
5: 0.0030499, 0.0044717, 0.0030301, 0.0044146, -0.0009960, 0.0010730
6: -0.0101990, -0.0045580, -0.0102777, -0.0047844, -0.0039518, 0.0042575
7: 0.0036509, 0.0113335, 0.0039592, 0.0114406, -0.0057983, 0.0053820
8: 0.9917856, 0.9971974, 0.9920028, 0.9972728, -0.0040845, 0.0037912
9: -0.0133433, -0.0084308, -0.0134118, -0.0086280, -0.0034414, 0.0037076

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0018519, upper bound: 0.0017865
time: 0.66 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0017760, upper bound: 0.0017066
time: 0.69 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062392, 0.0087773, 0.0063144, 0.0086391, -0.0018010, 0.0018753
1: 0.0022237, 0.0025904, 0.0022345, 0.0025704, -0.0002602, 0.0002709
2: 0.0095071, 0.0109104, 0.0095835, 0.0108688, -0.0010368, 0.0009957
3: -0.0048477, -0.0033964, -0.0047687, -0.0034394, -0.0010723, 0.0010298
4: -0.0003601, 0.0012110, -0.0003136, 0.0011255, -0.0011148, 0.0011609
5: 0.0029673, 0.0044542, 0.0030483, 0.0044101, -0.0010986, 0.0010550
6: -0.0105268, -0.0046276, -0.0102056, -0.0048024, -0.0043588, 0.0041860
7: 0.0037456, 0.0117799, 0.0039837, 0.0113425, -0.0057010, 0.0059363
8: 0.9918523, 0.9975119, 0.9920201, 0.9972038, -0.0040159, 0.0041817
9: -0.0136287, -0.0084914, -0.0133490, -0.0086437, -0.0037958, 0.0036454

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020710, upper bound: 0.0019064
time: 0.76 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0020710, upper bound: 0.0019065
time: 0.79 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0062659, 0.0088185, 0.0063322, 0.0086380, -0.0017941, 0.0019102
1: 0.0022275, 0.0025963, 0.0022371, 0.0025702, -0.0002592, 0.0002760
2: 0.0094843, 0.0108956, 0.0095841, 0.0108589, -0.0010561, 0.0009919
3: -0.0048713, -0.0034117, -0.0047681, -0.0034496, -0.0010923, 0.0010259
4: -0.0003436, 0.0012365, -0.0003025, 0.0011248, -0.0011106, 0.0011825
5: 0.0029432, 0.0044385, 0.0030489, 0.0043996, -0.0011190, 0.0010510
6: -0.0106226, -0.0046896, -0.0102031, -0.0048438, -0.0044399, 0.0041700
7: 0.0038301, 0.0119104, 0.0040402, 0.0113390, -0.0056792, 0.0060467
8: 0.9919119, 0.9976038, 0.9920598, 0.9972013, -0.0040005, 0.0042594
9: -0.0137122, -0.0085454, -0.0133468, -0.0086798, -0.0038664, 0.0036314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0014633, upper bound: 0.0012417
time: 0.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0012437, upper bound: 0.0010494
time: 0.61 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0062392, 0.0087773, 0.0062886, 0.0086712, -0.0018403, 0.0019105
1: 0.0022237, 0.0025904, 0.0022308, 0.0025750, -0.0002659, 0.0002760
2: 0.0095071, 0.0109104, 0.0095658, 0.0108831, -0.0010563, 0.0010175
3: -0.0048477, -0.0033964, -0.0047871, -0.0034247, -0.0010925, 0.0010523
4: -0.0003601, 0.0012110, -0.0003295, 0.0011453, -0.0011392, 0.0011826
5: 0.0029673, 0.0044542, 0.0030295, 0.0044252, -0.0011192, 0.0010781
6: -0.0105268, -0.0046276, -0.0102803, -0.0047424, -0.0044406, 0.0042774
7: 0.0037456, 0.0117799, 0.0039021, 0.0114441, -0.0058255, 0.0060477
8: 0.9918523, 0.9975119, 0.9919626, 0.9972753, -0.0041036, 0.0042601
9: -0.0136287, -0.0084914, -0.0134140, -0.0085915, -0.0038671, 0.0037250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 126

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021014, upper bound: 0.0018976
time: 0.65 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0021014, upper bound: 0.0018976
time: 0.67 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0062659, 0.0088185, 0.0063067, 0.0086701, -0.0018334, 0.0019455
1: 0.0022275, 0.0025963, 0.0022334, 0.0025749, -0.0002649, 0.0002811
2: 0.0094843, 0.0108956, 0.0095664, 0.0108731, -0.0010756, 0.0010137
3: -0.0048713, -0.0034117, -0.0047864, -0.0034350, -0.0011125, 0.0010484
4: -0.0003436, 0.0012365, -0.0003184, 0.0011446, -0.0011349, 0.0012043
5: 0.0029432, 0.0044385, 0.0030301, 0.0044146, -0.0011397, 0.0010740
6: -0.0106226, -0.0046896, -0.0102777, -0.0047844, -0.0045219, 0.0042614
7: 0.0038301, 0.0119104, 0.0039592, 0.0114406, -0.0058037, 0.0061584
8: 0.9919119, 0.9976038, 0.9920028, 0.9972728, -0.0040882, 0.0043381
9: -0.0137122, -0.0085454, -0.0134118, -0.0086280, -0.0039378, 0.0037110

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 247

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0015053, upper bound: 0.0012409
time: 0.69 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0013767, upper bound: 0.0010494
time: 0.64 seconds

## BFS IS instance: IS_A2_B2_A2_B2_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0062392, 0.0087773, 0.0063581, 0.0088459, -0.0019042, 0.0017247
1: 0.0022237, 0.0025904, 0.0022409, 0.0026003, -0.0002751, 0.0002492
2: 0.0095071, 0.0109104, 0.0094692, 0.0108446, -0.0009535, 0.0010528
3: -0.0048477, -0.0033964, -0.0048870, -0.0034644, -0.0009862, 0.0010889
4: -0.0003601, 0.0012110, -0.0002865, 0.0012535, -0.0011788, 0.0010676
5: 0.0029673, 0.0044542, 0.0029272, 0.0043845, -0.0010103, 0.0011155
6: -0.0105268, -0.0046276, -0.0106862, -0.0049040, -0.0040086, 0.0044260
7: 0.0037456, 0.0117799, 0.0041221, 0.0119970, -0.0060278, 0.0054594
8: 0.9918523, 0.9975119, 0.9921175, 0.9976648, -0.0042461, 0.0038457
9: -0.0136287, -0.0084914, -0.0137675, -0.0087321, -0.0034909, 0.0038543

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=1, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.82 seconds

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.39 + 598.09 = 601.48 seconds
