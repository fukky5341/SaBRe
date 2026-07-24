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
Threshold: 0.00504846


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.0037152, 0.0100136, 0.0037152, 0.0100136, -0.0062984, 0.0062984)
1: (0.0018590, 0.0027690, 0.0018590, 0.0027690, -0.0009099, 0.0009099)
2: (0.0088236, 0.0123058, 0.0088236, 0.0123058, -0.0034822, 0.0034822)
3: (-0.0055547, -0.0019532, -0.0055547, -0.0019532, -0.0036015, 0.0036015)
4: (-0.0019225, 0.0019763, -0.0019225, 0.0019763, -0.0038988, 0.0038988)
5: (0.0022431, 0.0059327, 0.0022431, 0.0059327, -0.0036896, 0.0036896)
6: (-0.0134005, 0.0012387, -0.0134005, 0.0012387, -0.0146392, 0.0146392)
7: (-0.0042438, 0.0156936, -0.0042438, 0.0156936, -0.0199373, 0.0199373)
8: (0.9862245, 1.0002687, 0.9862245, 1.0002687, -0.0140442, 0.0140442)
9: (-0.0161313, -0.0033828, -0.0161313, -0.0033828, -0.0127485, 0.0127485)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.67 + 1.97 = 3.64 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0078698, upper bound: 0.0078697

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073747, upper bound: 0.0075769
time: 0.82 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075770, upper bound: 0.0075769
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.87 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 8, lower bound: -0.0073747, upper bound: 0.0075769
IS_A2, status: Status.UNKNOWN, split count: 1, time: 1.87
Output dim: 8, lower bound: -0.0075770, upper bound: 0.0075769

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: 0.0043802, 0.0100284, 0.0039479, 0.0100054, -0.0055447, 0.0058942
1: 0.0019551, 0.0027711, 0.0018927, 0.0027678, -0.0008011, 0.0008515
2: 0.0088154, 0.0119382, 0.0088281, 0.0121772, -0.0032587, 0.0030655
3: -0.0055631, -0.0023334, -0.0055500, -0.0020863, -0.0033703, 0.0031705
4: -0.0015109, 0.0019854, -0.0017784, 0.0019712, -0.0034323, 0.0036486
5: 0.0022345, 0.0055432, 0.0022479, 0.0057964, -0.0034528, 0.0032481
6: -0.0134347, -0.0003067, -0.0133814, 0.0006979, -0.0136997, 0.0128875
7: -0.0021390, 0.0157402, -0.0035072, 0.0156675, -0.0175517, 0.0186578
8: 0.9877071, 1.0003016, 0.9867434, 1.0002505, -0.0123638, 0.0131429
9: -0.0161610, -0.0047286, -0.0161146, -0.0038537, -0.0119303, 0.0112230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071185, upper bound: 0.0072039
time: 0.86 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071185, upper bound: 0.0073224
time: 0.86 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: 0.0039003, 0.0100043, 0.0037395, 0.0100125, -0.0055573, 0.0062648
1: 0.0018858, 0.0027676, 0.0018625, 0.0027688, -0.0008029, 0.0009051
2: 0.0088287, 0.0122035, 0.0088242, 0.0122924, -0.0034637, 0.0030725
3: -0.0055494, -0.0020590, -0.0055540, -0.0019671, -0.0035823, 0.0031777
4: -0.0018079, 0.0019706, -0.0019075, 0.0019756, -0.0034401, 0.0038780
5: 0.0022485, 0.0058243, 0.0022438, 0.0059185, -0.0036699, 0.0032555
6: -0.0133788, 0.0008087, -0.0133977, 0.0011824, -0.0145612, 0.0129168
7: -0.0036581, 0.0156641, -0.0041670, 0.0156898, -0.0175915, 0.0198311
8: 0.9866371, 1.0002480, 0.9862785, 1.0002661, -0.0123918, 0.0139694
9: -0.0161124, -0.0037573, -0.0161289, -0.0034318, -0.0126805, 0.0112485

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075770, upper bound: 0.0073737
time: 0.89 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0075770, upper bound: 0.0075769
time: 0.88 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 3.53 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 8, lower bound: -0.0071185, upper bound: 0.0072039
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 8, lower bound: -0.0071185, upper bound: 0.0073224
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 8, lower bound: -0.0075770, upper bound: 0.0073737
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.53
Output dim: 8, lower bound: -0.0075770, upper bound: 0.0075769

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: 0.0044260, 0.0098305, 0.0039540, 0.0099773, -0.0054781, 0.0056877
1: 0.0019617, 0.0027425, 0.0018935, 0.0027637, -0.0007914, 0.0008217
2: 0.0089248, 0.0119129, 0.0088437, 0.0121738, -0.0031446, 0.0030287
3: -0.0054500, -0.0023596, -0.0055339, -0.0020897, -0.0032523, 0.0031325
4: -0.0014825, 0.0018630, -0.0017747, 0.0019538, -0.0033911, 0.0035208
5: 0.0023504, 0.0055163, 0.0022644, 0.0057928, -0.0033319, 0.0032091
6: -0.0129748, -0.0004131, -0.0133159, 0.0006838, -0.0132198, 0.0127327
7: -0.0019940, 0.0151139, -0.0034880, 0.0155784, -0.0173409, 0.0180042
8: 0.9878092, 0.9998604, 0.9867568, 1.0001876, -0.0122153, 0.0126826
9: -0.0157606, -0.0048213, -0.0160576, -0.0038660, -0.0115124, 0.0110882

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0067300
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0070377
time: 1.11 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: 0.0041675, 0.0098014, 0.0039668, 0.0099211, -0.0057536, 0.0056784
1: 0.0019244, 0.0027383, 0.0018954, 0.0027556, -0.0008312, 0.0008204
2: 0.0089409, 0.0120558, 0.0088748, 0.0121667, -0.0031394, 0.0031810
3: -0.0054333, -0.0022118, -0.0055018, -0.0020970, -0.0032469, 0.0032900
4: -0.0016425, 0.0018449, -0.0017668, 0.0019190, -0.0035616, 0.0035150
5: 0.0023674, 0.0056677, 0.0022973, 0.0057853, -0.0033264, 0.0033704
6: -0.0129071, 0.0001876, -0.0131853, 0.0006542, -0.0131981, 0.0133729
7: -0.0028122, 0.0150217, -0.0034476, 0.0154006, -0.0182128, 0.0179747
8: 0.9872329, 0.9997954, 0.9867853, 1.0000623, -0.0128294, 0.0126617
9: -0.0157016, -0.0042982, -0.0159439, -0.0038919, -0.0114935, 0.0116457

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=17, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0068558
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0071536
time: 1.05 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: 0.0039003, 0.0100043, 0.0043802, 0.0100284, -0.0060407, 0.0055437
1: 0.0018858, 0.0027676, 0.0019551, 0.0027711, -0.0008727, 0.0008009
2: 0.0088287, 0.0122035, 0.0088154, 0.0119382, -0.0030649, 0.0033397
3: -0.0055494, -0.0020590, -0.0055631, -0.0023334, -0.0031699, 0.0034541
4: -0.0018079, 0.0019706, -0.0015109, 0.0019854, -0.0037393, 0.0034316
5: 0.0022485, 0.0058243, 0.0022345, 0.0055432, -0.0032475, 0.0035386
6: -0.0133788, 0.0008087, -0.0134347, -0.0003067, -0.0128850, 0.0140402
7: -0.0036581, 0.0156641, -0.0021390, 0.0157402, -0.0191216, 0.0175482
8: 0.9866371, 1.0002480, 0.9877071, 1.0003016, -0.0134697, 0.0123613
9: -0.0161124, -0.0037573, -0.0161610, -0.0047286, -0.0112208, 0.0122269

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0072039, upper bound: 0.0071172
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073224, upper bound: 0.0071172
time: 0.89 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: 0.0039003, 0.0100043, 0.0039003, 0.0100043, -0.0055494, 0.0055495
1: 0.0018858, 0.0027676, 0.0018858, 0.0027676, -0.0008017, 0.0008017
2: 0.0088287, 0.0122035, 0.0088287, 0.0122035, -0.0030681, 0.0030681
3: -0.0055494, -0.0020590, -0.0055494, -0.0020590, -0.0031732, 0.0031732
4: -0.0018079, 0.0019706, -0.0018079, 0.0019706, -0.0034352, 0.0034352
5: 0.0022485, 0.0058243, 0.0022485, 0.0058243, -0.0032509, 0.0032509
6: -0.0133788, 0.0008087, -0.0133788, 0.0008087, -0.0128984, 0.0128984
7: -0.0036581, 0.0156641, -0.0036581, 0.0156641, -0.0175666, 0.0175666
8: 0.9866371, 1.0002480, 0.9866371, 1.0002480, -0.0123743, 0.0123743
9: -0.0161124, -0.0037573, -0.0161124, -0.0037573, -0.0112325, 0.0112325

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073224, upper bound: 0.0069933
time: 1.19 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0073224, upper bound: 0.0071191
time: 1.10 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 4.02 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0067300
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0070377
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0068558
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0071536
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0072039, upper bound: 0.0071172
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0073224, upper bound: 0.0071172
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0073224, upper bound: 0.0069933
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 4.02
Output dim: 8, lower bound: -0.0073224, upper bound: 0.0071191

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0044689, 0.0098281, 0.0041970, 0.0099644, -0.0054167, 0.0054260
1: 0.0019679, 0.0027422, 0.0019286, 0.0027619, -0.0007826, 0.0007839
2: 0.0089261, 0.0118891, 0.0088508, 0.0120395, -0.0029999, 0.0029947
3: -0.0054486, -0.0023842, -0.0055266, -0.0022287, -0.0031026, 0.0030973
4: -0.0014559, 0.0018615, -0.0016243, 0.0019459, -0.0033530, 0.0033588
5: 0.0023518, 0.0054912, 0.0022719, 0.0056505, -0.0031785, 0.0031731
6: -0.0129693, -0.0005131, -0.0132861, 0.0001190, -0.0126115, 0.0125898
7: -0.0018580, 0.0151063, -0.0027188, 0.0155378, -0.0171463, 0.0171758
8: 0.9879051, 0.9998550, 0.9872987, 1.0001591, -0.0120782, 0.0120990
9: -0.0157557, -0.0049083, -0.0160317, -0.0043579, -0.0109827, 0.0109638

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0066215
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0067300
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0041001, 0.0103188, -0.0057614, 0.0054907
1: 0.0019720, 0.0027414, 0.0019147, 0.0028131, -0.0008324, 0.0007932
2: 0.0089291, 0.0118735, 0.0086549, 0.0120930, -0.0030356, 0.0031853
3: -0.0054456, -0.0024004, -0.0057292, -0.0021733, -0.0031396, 0.0032944
4: -0.0014384, 0.0018582, -0.0016842, 0.0021652, -0.0035664, 0.0033988
5: 0.0023548, 0.0054746, 0.0020643, 0.0057072, -0.0032164, 0.0033750
6: -0.0129570, -0.0005788, -0.0141098, 0.0003441, -0.0127618, 0.0133911
7: -0.0017685, 0.0150896, -0.0030254, 0.0166596, -0.0182376, 0.0173805
8: 0.9879681, 0.9998432, 0.9870827, 1.0009493, -0.0128469, 0.0122432
9: -0.0157450, -0.0049655, -0.0167489, -0.0041618, -0.0111135, 0.0116616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0068002
time: 1.12 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0070377
time: 1.15 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0042065, 0.0097988, 0.0042097, 0.0099083, -0.0056991, 0.0054164
1: 0.0019300, 0.0027379, 0.0019305, 0.0027538, -0.0008234, 0.0007825
2: 0.0089424, 0.0120342, 0.0088818, 0.0120324, -0.0029946, 0.0031509
3: -0.0054319, -0.0022341, -0.0054944, -0.0022360, -0.0030972, 0.0032588
4: -0.0016184, 0.0018433, -0.0016164, 0.0019111, -0.0035278, 0.0033529
5: 0.0023689, 0.0056449, 0.0023048, 0.0056430, -0.0031729, 0.0033385
6: -0.0129011, 0.0000969, -0.0131556, 0.0000894, -0.0125893, 0.0132463
7: -0.0026887, 0.0150135, -0.0026785, 0.0153600, -0.0180403, 0.0171455
8: 0.9873199, 0.9997897, 0.9873270, 1.0000337, -0.0127080, 0.0120777
9: -0.0156964, -0.0043771, -0.0159180, -0.0043836, -0.0109633, 0.0115355

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0067490
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0068558
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042394, 0.0097935, 0.0041140, 0.0102598, -0.0060204, 0.0054798
1: 0.0019348, 0.0027372, 0.0019167, 0.0028045, -0.0008698, 0.0007917
2: 0.0089453, 0.0120160, 0.0086875, 0.0120853, -0.0030297, 0.0033285
3: -0.0054288, -0.0022529, -0.0056954, -0.0021812, -0.0031334, 0.0034425
4: -0.0015980, 0.0018401, -0.0016757, 0.0021287, -0.0037267, 0.0033921
5: 0.0023720, 0.0056256, 0.0020989, 0.0056991, -0.0032101, 0.0035267
6: -0.0128889, 0.0000205, -0.0139725, 0.0003120, -0.0127366, 0.0139930
7: -0.0025846, 0.0149968, -0.0029816, 0.0164726, -0.0190572, 0.0173462
8: 0.9873933, 0.9997779, 0.9871136, 1.0008174, -0.0134241, 0.0122190
9: -0.0156857, -0.0044437, -0.0166294, -0.0041898, -0.0110916, 0.0121857

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0069157
time: 1.10 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0071536
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0039063, 0.0099762, 0.0044260, 0.0098305, -0.0058339, 0.0054771
1: 0.0018866, 0.0027636, 0.0019617, 0.0027425, -0.0008428, 0.0007913
2: 0.0088443, 0.0122002, 0.0089248, 0.0119129, -0.0030281, 0.0032254
3: -0.0055333, -0.0020625, -0.0054500, -0.0023596, -0.0031318, 0.0033359
4: -0.0018042, 0.0019531, -0.0014825, 0.0018630, -0.0036113, 0.0033904
5: 0.0022650, 0.0058207, 0.0023504, 0.0055163, -0.0032084, 0.0034175
6: -0.0133133, 0.0007947, -0.0129748, -0.0004131, -0.0127302, 0.0135596
7: -0.0036390, 0.0155749, -0.0019940, 0.0151139, -0.0184671, 0.0173374
8: 0.9866506, 1.0001851, 0.9878092, 0.9998604, -0.0130086, 0.0122128
9: -0.0160554, -0.0037695, -0.0157606, -0.0048213, -0.0110860, 0.0118083

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067300, upper bound: 0.0069157
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0070377, upper bound: 0.0069157
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0039194, 0.0099199, 0.0041675, 0.0098014, -0.0058238, 0.0057524
1: 0.0018885, 0.0027554, 0.0019244, 0.0027383, -0.0008414, 0.0008311
2: 0.0088754, 0.0121929, 0.0089409, 0.0120558, -0.0031804, 0.0032198
3: -0.0055011, -0.0020700, -0.0054333, -0.0022118, -0.0032893, 0.0033301
4: -0.0017961, 0.0019183, -0.0016425, 0.0018449, -0.0036050, 0.0035609
5: 0.0022980, 0.0058130, 0.0023674, 0.0056677, -0.0033698, 0.0034116
6: -0.0131827, 0.0007641, -0.0129071, 0.0001876, -0.0133703, 0.0135361
7: -0.0035974, 0.0153969, -0.0028122, 0.0150217, -0.0184350, 0.0182091
8: 0.9866798, 1.0000597, 0.9872329, 0.9997954, -0.0129860, 0.0128268
9: -0.0159416, -0.0037961, -0.0157016, -0.0042982, -0.0116434, 0.0117879

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=16, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068558, upper bound: 0.0069157
time: 1.12 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071536, upper bound: 0.0069157
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0039448, 0.0098017, 0.0039063, 0.0099762, -0.0054820, 0.0053463
1: 0.0018922, 0.0027384, 0.0018866, 0.0027636, -0.0007920, 0.0007724
2: 0.0089407, 0.0121789, 0.0088443, 0.0122002, -0.0029558, 0.0030309
3: -0.0054335, -0.0020845, -0.0055333, -0.0020625, -0.0030570, 0.0031347
4: -0.0017804, 0.0018452, -0.0018042, 0.0019531, -0.0033934, 0.0033094
5: 0.0023672, 0.0057982, 0.0022650, 0.0058207, -0.0031318, 0.0032113
6: -0.0129079, 0.0007053, -0.0133133, 0.0007947, -0.0124262, 0.0127417
7: -0.0035173, 0.0150228, -0.0036390, 0.0155749, -0.0173531, 0.0169233
8: 0.9867362, 0.9997962, 0.9866506, 1.0001851, -0.0122239, 0.0119212
9: -0.0157023, -0.0038473, -0.0160554, -0.0037695, -0.0108212, 0.0110960

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0066307
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0068111
time: 0.85 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0036944, 0.0097770, 0.0039194, 0.0099199, -0.0057496, 0.0053372
1: 0.0018560, 0.0027348, 0.0018885, 0.0027554, -0.0008307, 0.0007711
2: 0.0089544, 0.0123173, 0.0088754, 0.0121929, -0.0029508, 0.0031788
3: -0.0054194, -0.0019413, -0.0055011, -0.0020700, -0.0030519, 0.0032877
4: -0.0019354, 0.0018298, -0.0017961, 0.0019183, -0.0035591, 0.0033038
5: 0.0023817, 0.0059449, 0.0022980, 0.0058130, -0.0031265, 0.0033681
6: -0.0128503, 0.0012872, -0.0131827, 0.0007641, -0.0124051, 0.0133638
7: -0.0043098, 0.0149443, -0.0035974, 0.0153969, -0.0182003, 0.0168947
8: 0.9861780, 0.9997410, 0.9866798, 1.0000597, -0.0128207, 0.0119010
9: -0.0156522, -0.0033405, -0.0159416, -0.0037961, -0.0108029, 0.0116378

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=16, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0067576
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0069301
time: 1.12 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.77 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0066215
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0067300
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0068002
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0070377
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0067490
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0068558
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0069157
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0069157, upper bound: 0.0071536
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0067300, upper bound: 0.0069157
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0070377, upper bound: 0.0069157
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0068558, upper bound: 0.0069157
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0071536, upper bound: 0.0069157
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0066307
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0068111
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0067576
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.77
Output dim: 8, lower bound: -0.0071540, upper bound: 0.0069301

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0044689, 0.0098281, 0.0046120, 0.0099872, -0.0052226, 0.0048856
1: 0.0019679, 0.0027422, 0.0019886, 0.0027652, -0.0007545, 0.0007058
2: 0.0089261, 0.0118891, 0.0088382, 0.0118100, -0.0027011, 0.0028874
3: -0.0054486, -0.0023842, -0.0055396, -0.0024660, -0.0027936, 0.0029863
4: -0.0014559, 0.0018615, -0.0013673, 0.0019600, -0.0032329, 0.0030243
5: 0.0023518, 0.0054912, 0.0022586, 0.0054073, -0.0028620, 0.0030594
6: -0.0129693, -0.0005131, -0.0133391, -0.0008457, -0.0113555, 0.0121387
7: -0.0018580, 0.0151063, -0.0014050, 0.0156099, -0.0165319, 0.0154651
8: 0.9879051, 0.9998550, 0.9882241, 1.0002098, -0.0116454, 0.0108940
9: -0.0157557, -0.0049083, -0.0160778, -0.0051980, -0.0098888, 0.0105709

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0062503
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0063471
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0044689, 0.0098281, 0.0041569, 0.0099632, -0.0054154, 0.0055763
1: 0.0019679, 0.0027422, 0.0019228, 0.0027617, -0.0007824, 0.0008056
2: 0.0089261, 0.0118891, 0.0088515, 0.0120616, -0.0030830, 0.0029940
3: -0.0054486, -0.0023842, -0.0055258, -0.0022057, -0.0031886, 0.0030966
4: -0.0014559, 0.0018615, -0.0016491, 0.0019451, -0.0033522, 0.0034518
5: 0.0023518, 0.0054912, 0.0022726, 0.0056740, -0.0032666, 0.0031723
6: -0.0129693, -0.0005131, -0.0132832, 0.0002123, -0.0129609, 0.0125869
7: -0.0018580, 0.0151063, -0.0028458, 0.0155338, -0.0171423, 0.0176517
8: 0.9879051, 0.9998550, 0.9872092, 1.0001562, -0.0120754, 0.0124342
9: -0.0157557, -0.0049083, -0.0160291, -0.0042766, -0.0112870, 0.0109613

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0063879
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0064695
time: 1.15 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0045444, 0.0103172, -0.0056078, 0.0049513
1: 0.0019720, 0.0027414, 0.0019788, 0.0028128, -0.0008102, 0.0007153
2: 0.0089291, 0.0118735, 0.0086557, 0.0118474, -0.0027374, 0.0031004
3: -0.0054456, -0.0024004, -0.0057283, -0.0024273, -0.0028312, 0.0032066
4: -0.0014384, 0.0018582, -0.0014092, 0.0021643, -0.0034713, 0.0030649
5: 0.0023548, 0.0054746, 0.0020652, 0.0054469, -0.0029004, 0.0032850
6: -0.0129570, -0.0005788, -0.0141061, -0.0006884, -0.0115081, 0.0130340
7: -0.0017685, 0.0150896, -0.0016191, 0.0166545, -0.0177512, 0.0156731
8: 0.9879681, 0.9998432, 0.9880733, 1.0009456, -0.0125043, 0.0110404
9: -0.0157450, -0.0049655, -0.0167457, -0.0050610, -0.0100218, 0.0113506

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0063930
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0065417
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0040553, 0.0103177, -0.0057603, 0.0056385
1: 0.0019720, 0.0027414, 0.0019082, 0.0028129, -0.0008322, 0.0008146
2: 0.0089291, 0.0118735, 0.0086555, 0.0121178, -0.0031174, 0.0031847
3: -0.0054456, -0.0024004, -0.0057286, -0.0021477, -0.0032242, 0.0032938
4: -0.0014384, 0.0018582, -0.0017120, 0.0021646, -0.0035657, 0.0034904
5: 0.0023548, 0.0054746, 0.0020649, 0.0057334, -0.0033031, 0.0033744
6: -0.0129570, -0.0005788, -0.0141073, 0.0004483, -0.0131055, 0.0133886
7: -0.0017685, 0.0150896, -0.0031673, 0.0166562, -0.0182341, 0.0178486
8: 0.9879681, 0.9998432, 0.9869827, 1.0009468, -0.0128445, 0.0125729
9: -0.0157450, -0.0049655, -0.0167468, -0.0040711, -0.0114129, 0.0116594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0066777
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0067998
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0042065, 0.0097988, 0.0046243, 0.0099282, -0.0055102, 0.0048775
1: 0.0019300, 0.0027379, 0.0019904, 0.0027566, -0.0007961, 0.0007047
2: 0.0089424, 0.0120342, 0.0088708, 0.0118032, -0.0026966, 0.0030465
3: -0.0054319, -0.0022341, -0.0055058, -0.0024730, -0.0027890, 0.0031508
4: -0.0016184, 0.0018433, -0.0013598, 0.0019234, -0.0034109, 0.0030192
5: 0.0023689, 0.0056449, 0.0022932, 0.0054002, -0.0028572, 0.0032279
6: -0.0129011, 0.0000969, -0.0132018, -0.0008741, -0.0113366, 0.0128073
7: -0.0026887, 0.0150135, -0.0013663, 0.0154229, -0.0174425, 0.0154394
8: 0.9873199, 0.9997897, 0.9882514, 1.0000782, -0.0122868, 0.0108758
9: -0.0156964, -0.0043771, -0.0159582, -0.0052227, -0.0098724, 0.0111532

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0063417
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0064625
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0042065, 0.0097988, 0.0041702, 0.0099069, -0.0056979, 0.0055659
1: 0.0019300, 0.0027379, 0.0019248, 0.0027536, -0.0008232, 0.0008041
2: 0.0089424, 0.0120342, 0.0088826, 0.0120543, -0.0030772, 0.0031502
3: -0.0054319, -0.0022341, -0.0054937, -0.0022133, -0.0031826, 0.0032581
4: -0.0016184, 0.0018433, -0.0016409, 0.0019103, -0.0035271, 0.0034454
5: 0.0023689, 0.0056449, 0.0023056, 0.0056662, -0.0032605, 0.0033378
6: -0.0129011, 0.0000969, -0.0131525, 0.0001814, -0.0129366, 0.0132434
7: -0.0026887, 0.0150135, -0.0028038, 0.0153558, -0.0180364, 0.0176185
8: 0.9873199, 0.9997897, 0.9872389, 1.0000308, -0.0127052, 0.0124109
9: -0.0156964, -0.0043771, -0.0159153, -0.0043035, -0.0112658, 0.0115330

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0064902
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0065862
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0042394, 0.0097935, 0.0045570, 0.0102561, -0.0058804, 0.0049421
1: 0.0019348, 0.0027372, 0.0019807, 0.0028040, -0.0008496, 0.0007140
2: 0.0089453, 0.0120160, 0.0086895, 0.0118404, -0.0027324, 0.0032511
3: -0.0054288, -0.0022529, -0.0056934, -0.0024346, -0.0028259, 0.0033625
4: -0.0015980, 0.0018401, -0.0014014, 0.0021264, -0.0036401, 0.0030593
5: 0.0023720, 0.0056256, 0.0021010, 0.0054395, -0.0028951, 0.0034447
6: -0.0128889, 0.0000205, -0.0139641, -0.0007178, -0.0114868, 0.0136678
7: -0.0025846, 0.0149968, -0.0015791, 0.0164612, -0.0186143, 0.0156441
8: 0.9873933, 0.9997779, 0.9881015, 1.0008094, -0.0131123, 0.0110200
9: -0.0156857, -0.0044437, -0.0166221, -0.0050866, -0.0100032, 0.0119025

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0064812
time: 0.85 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0066430
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0042394, 0.0097935, 0.0040693, 0.0102587, -0.0060193, 0.0056272
1: 0.0019348, 0.0027372, 0.0019102, 0.0028044, -0.0008696, 0.0008130
2: 0.0089453, 0.0120160, 0.0086881, 0.0121100, -0.0031111, 0.0033279
3: -0.0054288, -0.0022529, -0.0056948, -0.0021557, -0.0032177, 0.0034419
4: -0.0015980, 0.0018401, -0.0017033, 0.0021280, -0.0037260, 0.0034833
5: 0.0023720, 0.0056256, 0.0020995, 0.0057253, -0.0032964, 0.0035261
6: -0.0128889, 0.0000205, -0.0139700, 0.0004158, -0.0130791, 0.0139905
7: -0.0025846, 0.0149968, -0.0031230, 0.0164692, -0.0190538, 0.0178126
8: 0.9873933, 0.9997779, 0.9870140, 1.0008152, -0.0134218, 0.0125476
9: -0.0156857, -0.0044437, -0.0166272, -0.0040994, -0.0113899, 0.0121835

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0067528
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0069076
time: 0.83 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0041569, 0.0099632, 0.0044689, 0.0098281, -0.0055763, 0.0054154
1: 0.0019228, 0.0027617, 0.0019679, 0.0027422, -0.0008056, 0.0007824
2: 0.0088515, 0.0120616, 0.0089261, 0.0118891, -0.0029940, 0.0030830
3: -0.0055258, -0.0022057, -0.0054486, -0.0023842, -0.0030966, 0.0031886
4: -0.0016491, 0.0019451, -0.0014559, 0.0018615, -0.0034518, 0.0033522
5: 0.0022726, 0.0056740, 0.0023518, 0.0054912, -0.0031723, 0.0032666
6: -0.0132832, 0.0002123, -0.0129693, -0.0005131, -0.0125869, 0.0129609
7: -0.0028458, 0.0155338, -0.0018580, 0.0151063, -0.0176517, 0.0171423
8: 0.9872092, 1.0001562, 0.9879051, 0.9998550, -0.0124342, 0.0120754
9: -0.0160291, -0.0042766, -0.0157557, -0.0049083, -0.0109613, 0.0112870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063879, upper bound: 0.0065309
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064696, upper bound: 0.0066430
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0040553, 0.0103177, 0.0044972, 0.0098228, -0.0056386, 0.0057603
1: 0.0019082, 0.0028129, 0.0019720, 0.0027414, -0.0008146, 0.0008322
2: 0.0086555, 0.0121178, 0.0089291, 0.0118735, -0.0031847, 0.0031174
3: -0.0057286, -0.0021477, -0.0054456, -0.0024004, -0.0032938, 0.0032242
4: -0.0017120, 0.0021646, -0.0014384, 0.0018582, -0.0034904, 0.0035657
5: 0.0020649, 0.0057334, 0.0023548, 0.0054746, -0.0033744, 0.0033031
6: -0.0141073, 0.0004483, -0.0129570, -0.0005788, -0.0133886, 0.0131055
7: -0.0031673, 0.0166562, -0.0017685, 0.0150896, -0.0178486, 0.0182341
8: 0.9869827, 1.0009468, 0.9879681, 0.9998432, -0.0125729, 0.0128445
9: -0.0167468, -0.0040711, -0.0157450, -0.0049655, -0.0116594, 0.0114129

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066777, upper bound: 0.0065309
time: 1.06 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067998, upper bound: 0.0066430
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0041702, 0.0099069, 0.0042065, 0.0097988, -0.0055659, 0.0056979
1: 0.0019248, 0.0027536, 0.0019300, 0.0027379, -0.0008041, 0.0008232
2: 0.0088826, 0.0120543, 0.0089424, 0.0120342, -0.0031502, 0.0030772
3: -0.0054937, -0.0022133, -0.0054319, -0.0022341, -0.0032581, 0.0031826
4: -0.0016409, 0.0019103, -0.0016184, 0.0018433, -0.0034454, 0.0035271
5: 0.0023056, 0.0056662, 0.0023689, 0.0056449, -0.0033378, 0.0032605
6: -0.0131525, 0.0001814, -0.0129011, 0.0000969, -0.0132434, 0.0129366
7: -0.0028038, 0.0153558, -0.0026887, 0.0150135, -0.0176185, 0.0180364
8: 0.9872389, 1.0000308, 0.9873199, 0.9997897, -0.0124109, 0.0127052
9: -0.0159153, -0.0043035, -0.0156964, -0.0043771, -0.0115330, 0.0112658

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064902, upper bound: 0.0065309
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065862, upper bound: 0.0066430
time: 0.94 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0040693, 0.0102587, 0.0042394, 0.0097935, -0.0056272, 0.0060193
1: 0.0019102, 0.0028044, 0.0019348, 0.0027372, -0.0008130, 0.0008696
2: 0.0086881, 0.0121100, 0.0089453, 0.0120160, -0.0033279, 0.0031111
3: -0.0056948, -0.0021557, -0.0054288, -0.0022529, -0.0034419, 0.0032177
4: -0.0017033, 0.0021280, -0.0015980, 0.0018401, -0.0034833, 0.0037260
5: 0.0020995, 0.0057253, 0.0023720, 0.0056256, -0.0035261, 0.0032964
6: -0.0139700, 0.0004158, -0.0128889, 0.0000205, -0.0139905, 0.0130791
7: -0.0031230, 0.0164692, -0.0025846, 0.0149968, -0.0178126, 0.0190538
8: 0.9870140, 1.0008152, 0.9873933, 0.9997779, -0.0125476, 0.0134218
9: -0.0166272, -0.0040994, -0.0156857, -0.0044437, -0.0121835, 0.0113899

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067528, upper bound: 0.0065309
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069076, upper bound: 0.0066430
time: 1.00 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0039907, 0.0097994, 0.0041569, 0.0099632, -0.0054194, 0.0050790
1: 0.0018988, 0.0027380, 0.0019228, 0.0027617, -0.0007829, 0.0007338
2: 0.0089420, 0.0121535, 0.0088515, 0.0120616, -0.0028080, 0.0029962
3: -0.0054322, -0.0021108, -0.0055258, -0.0022057, -0.0029042, 0.0030988
4: -0.0017519, 0.0018437, -0.0016491, 0.0019451, -0.0033547, 0.0031440
5: 0.0023686, 0.0057713, 0.0022726, 0.0056740, -0.0029752, 0.0031746
6: -0.0129024, 0.0005984, -0.0132832, 0.0002123, -0.0118049, 0.0125961
7: -0.0033717, 0.0150153, -0.0028458, 0.0155338, -0.0171548, 0.0160772
8: 0.9868388, 0.9997910, 0.9872092, 1.0001562, -0.0120842, 0.0113251
9: -0.0156975, -0.0039404, -0.0160291, -0.0042766, -0.0102802, 0.0109692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0063244
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0063620
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0040117, 0.0097942, 0.0040553, 0.0103177, -0.0057899, 0.0051597
1: 0.0019019, 0.0027373, 0.0019082, 0.0028129, -0.0008365, 0.0007454
2: 0.0089449, 0.0121419, 0.0086555, 0.0121178, -0.0028526, 0.0032011
3: -0.0054292, -0.0021227, -0.0057286, -0.0021477, -0.0029503, 0.0033107
4: -0.0017390, 0.0018405, -0.0017120, 0.0021646, -0.0035840, 0.0031939
5: 0.0023716, 0.0057590, 0.0020649, 0.0057334, -0.0030225, 0.0033917
6: -0.0128903, 0.0005497, -0.0141073, 0.0004483, -0.0119925, 0.0134573
7: -0.0033054, 0.0149988, -0.0031673, 0.0166562, -0.0183276, 0.0163327
8: 0.9868855, 0.9997794, 0.9869827, 1.0009468, -0.0129104, 0.0115051
9: -0.0156870, -0.0039828, -0.0167468, -0.0040711, -0.0104436, 0.0117192

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0064780
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0065618
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0037401, 0.0097745, 0.0041702, 0.0099069, -0.0056900, 0.0050697
1: 0.0018626, 0.0027344, 0.0019248, 0.0027536, -0.0008220, 0.0007324
2: 0.0089558, 0.0122921, 0.0088826, 0.0120543, -0.0028029, 0.0031459
3: -0.0054179, -0.0019674, -0.0054937, -0.0022133, -0.0028989, 0.0032536
4: -0.0019071, 0.0018283, -0.0016409, 0.0019103, -0.0035222, 0.0031382
5: 0.0023832, 0.0059181, 0.0023056, 0.0056662, -0.0029698, 0.0033332
6: -0.0128446, 0.0011810, -0.0131525, 0.0001814, -0.0117833, 0.0132252
7: -0.0041652, 0.0149365, -0.0028038, 0.0153558, -0.0180116, 0.0160478
8: 0.9862798, 0.9997354, 0.9872389, 1.0000308, -0.0126878, 0.0113044
9: -0.0156471, -0.0034330, -0.0159153, -0.0043035, -0.0102614, 0.0115171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0064264
time: 1.14 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2

### Relational analysis result of IS_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0064790
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0037611, 0.0097691, 0.0040693, 0.0102587, -0.0060485, 0.0051486
1: 0.0018657, 0.0027337, 0.0019102, 0.0028044, -0.0008738, 0.0007438
2: 0.0089588, 0.0122804, 0.0086881, 0.0121100, -0.0028465, 0.0033441
3: -0.0054149, -0.0019794, -0.0056948, -0.0021557, -0.0029440, 0.0034586
4: -0.0018941, 0.0018250, -0.0017033, 0.0021280, -0.0037441, 0.0031871
5: 0.0023863, 0.0059058, 0.0020995, 0.0057253, -0.0030160, 0.0035432
6: -0.0128321, 0.0011322, -0.0139700, 0.0004158, -0.0119668, 0.0140584
7: -0.0040986, 0.0149195, -0.0031230, 0.0164692, -0.0191463, 0.0162977
8: 0.9863267, 0.9997236, 0.9870140, 1.0008152, -0.0134870, 0.0114805
9: -0.0156363, -0.0034756, -0.0166272, -0.0040994, -0.0104212, 0.0122427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0065738
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0066675
time: 0.91 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 3.81 seconds
IS_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0062503
IS_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0063471
IS_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0063879
IS_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0064695
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0063930
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0065417
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0066777
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0067998
IS_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0063417
IS_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0064625
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0064902
IS_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0065862
IS_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0064812
IS_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0066430
IS_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065309, upper bound: 0.0067528
IS_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066430, upper bound: 0.0069076
IS_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0063879, upper bound: 0.0065309
IS_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0064696, upper bound: 0.0066430
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0066777, upper bound: 0.0065309
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0067998, upper bound: 0.0066430
IS_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0064902, upper bound: 0.0065309
IS_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0065862, upper bound: 0.0066430
IS_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0067528, upper bound: 0.0065309
IS_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0069076, upper bound: 0.0066430
IS_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0063244
IS_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0063620
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0064780
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0065618
IS_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0064264
IS_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0064790
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0068503, upper bound: 0.0065738
IS_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.81
Output dim: 8, lower bound: -0.0069093, upper bound: 0.0066675

## BFS IS instance: IS_A1_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0045995, 0.0098181, 0.0049858, 0.0101381, -0.0050310, 0.0043813
1: 0.0019868, 0.0027407, 0.0020426, 0.0027870, -0.0007268, 0.0006330
2: 0.0089317, 0.0118169, 0.0087548, 0.0116033, -0.0024223, 0.0027815
3: -0.0054429, -0.0024588, -0.0056259, -0.0026798, -0.0025053, 0.0028768
4: -0.0013751, 0.0018553, -0.0011360, 0.0020534, -0.0031143, 0.0027121
5: 0.0023576, 0.0054147, 0.0021702, 0.0051884, -0.0025665, 0.0029472
6: -0.0129460, -0.0008165, -0.0136897, -0.0017145, -0.0101833, 0.0116935
7: -0.0014447, 0.0150746, -0.0002218, 0.0160875, -0.0159256, 0.0138688
8: 0.9881962, 0.9998327, 0.9890577, 1.0005463, -0.0112183, 0.0097694
9: -0.0157355, -0.0051726, -0.0163832, -0.0059545, -0.0088681, 0.0101832

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B1_B1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0062638
time: 0.84 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0062638
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0044689, 0.0098281, 0.0047352, 0.0099782, -0.0052133, 0.0044433
1: 0.0019679, 0.0027422, 0.0020064, 0.0027639, -0.0007532, 0.0006419
2: 0.0089261, 0.0118891, 0.0088432, 0.0117419, -0.0024566, 0.0028823
3: -0.0054486, -0.0023842, -0.0055344, -0.0025365, -0.0025407, 0.0029810
4: -0.0014559, 0.0018615, -0.0012911, 0.0019544, -0.0032271, 0.0027505
5: 0.0023518, 0.0054912, 0.0022639, 0.0053352, -0.0026029, 0.0030539
6: -0.0129693, -0.0005131, -0.0133180, -0.0011320, -0.0103275, 0.0121171
7: -0.0018580, 0.0151063, -0.0010151, 0.0155813, -0.0165024, 0.0140651
8: 0.9879051, 0.9998550, 0.9884988, 1.0001897, -0.0116247, 0.0099078
9: -0.0157557, -0.0049083, -0.0160594, -0.0054473, -0.0089936, 0.0105521

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B1_B1_B2_B1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0063745
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0063745
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0045995, 0.0098181, 0.0045605, 0.0101013, -0.0052784, 0.0050676
1: 0.0019868, 0.0027407, 0.0019812, 0.0027816, -0.0007626, 0.0007321
2: 0.0089317, 0.0118169, 0.0087751, 0.0118385, -0.0028017, 0.0029183
3: -0.0054429, -0.0024588, -0.0056048, -0.0024365, -0.0028977, 0.0030182
4: -0.0013751, 0.0018553, -0.0013993, 0.0020306, -0.0032674, 0.0031369
5: 0.0023576, 0.0054147, 0.0021917, 0.0054375, -0.0029686, 0.0030921
6: -0.0129460, -0.0008165, -0.0136041, -0.0007258, -0.0117784, 0.0122684
7: -0.0014447, 0.0150746, -0.0015683, 0.0159709, -0.0167084, 0.0160412
8: 0.9881962, 0.9998327, 0.9881092, 1.0004641, -0.0117698, 0.0112998
9: -0.0157355, -0.0051726, -0.0163086, -0.0050936, -0.0102572, 0.0106838

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063879
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063879
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0044689, 0.0098281, 0.0042780, 0.0099552, -0.0054073, 0.0051415
1: 0.0019679, 0.0027422, 0.0019403, 0.0027605, -0.0007812, 0.0007428
2: 0.0089261, 0.0118891, 0.0088559, 0.0119947, -0.0028426, 0.0029896
3: -0.0054486, -0.0023842, -0.0055213, -0.0022750, -0.0029399, 0.0030920
4: -0.0014559, 0.0018615, -0.0015741, 0.0019401, -0.0033472, 0.0031827
5: 0.0023518, 0.0054912, 0.0022773, 0.0056030, -0.0030119, 0.0031676
6: -0.0129693, -0.0005131, -0.0132645, -0.0000693, -0.0119502, 0.0125681
7: -0.0018580, 0.0151063, -0.0024623, 0.0155084, -0.0171167, 0.0162751
8: 0.9879051, 0.9998550, 0.9874793, 1.0001384, -0.0120573, 0.0114645
9: -0.0157557, -0.0049083, -0.0160129, -0.0045219, -0.0104067, 0.0109449

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061899, upper bound: 0.0061821
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0062145
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0046303, 0.0098127, 0.0049383, 0.0104353, -0.0054228, 0.0044434
1: 0.0019912, 0.0027400, 0.0020357, 0.0028299, -0.0007834, 0.0006419
2: 0.0089347, 0.0117999, 0.0085905, 0.0116296, -0.0024566, 0.0029981
3: -0.0054398, -0.0024764, -0.0057958, -0.0026526, -0.0025408, 0.0031008
4: -0.0013561, 0.0018520, -0.0011654, 0.0022373, -0.0033568, 0.0027505
5: 0.0023608, 0.0053966, 0.0019961, 0.0052162, -0.0026029, 0.0031767
6: -0.0129335, -0.0008880, -0.0143805, -0.0016039, -0.0103276, 0.0126041
7: -0.0013473, 0.0150576, -0.0003723, 0.0170283, -0.0171656, 0.0140653
8: 0.9882648, 0.9998208, 0.9889516, 1.0012089, -0.0120918, 0.0099079
9: -0.0157246, -0.0052348, -0.0169847, -0.0058583, -0.0089937, 0.0109762

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_B1_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063930
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063930
time: 1.16 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0046598, 0.0103081, -0.0055982, 0.0045136
1: 0.0019720, 0.0027414, 0.0019955, 0.0028115, -0.0008088, 0.0006521
2: 0.0089291, 0.0118735, 0.0086608, 0.0117836, -0.0024954, 0.0030951
3: -0.0054456, -0.0024004, -0.0057231, -0.0024933, -0.0025809, 0.0032011
4: -0.0014384, 0.0018582, -0.0013378, 0.0021586, -0.0034654, 0.0027940
5: 0.0023548, 0.0054746, 0.0020706, 0.0053793, -0.0026440, 0.0032794
6: -0.0129570, -0.0005788, -0.0140848, -0.0009567, -0.0104908, 0.0130119
7: -0.0017685, 0.0150896, -0.0012538, 0.0166255, -0.0177210, 0.0142875
8: 0.9879681, 0.9998432, 0.9883308, 1.0009253, -0.0124831, 0.0100644
9: -0.0157450, -0.0049655, -0.0167272, -0.0052947, -0.0091358, 0.0113313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_B1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0065429
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0065429
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0046303, 0.0098127, 0.0044670, 0.0104468, -0.0056157, 0.0051272
1: 0.0019912, 0.0027400, 0.0019676, 0.0028316, -0.0008113, 0.0007407
2: 0.0089347, 0.0117999, 0.0085841, 0.0118902, -0.0028347, 0.0031048
3: -0.0054398, -0.0024764, -0.0058024, -0.0023831, -0.0029318, 0.0032111
4: -0.0013561, 0.0018520, -0.0014572, 0.0022445, -0.0034762, 0.0031738
5: 0.0023608, 0.0053966, 0.0019893, 0.0054923, -0.0030035, 0.0032896
6: -0.0129335, -0.0008880, -0.0144073, -0.0005084, -0.0119169, 0.0130524
7: -0.0013473, 0.0150576, -0.0018643, 0.0170648, -0.0177762, 0.0162298
8: 0.9882648, 0.9998208, 0.9879007, 1.0012348, -0.0125219, 0.0114326
9: -0.0157246, -0.0052348, -0.0170081, -0.0049043, -0.0103778, 0.0113666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0066777
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0066777
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0041779, 0.0103095, -0.0057521, 0.0052117
1: 0.0019720, 0.0027414, 0.0019259, 0.0028117, -0.0008310, 0.0007529
2: 0.0089291, 0.0118735, 0.0086600, 0.0120500, -0.0028814, 0.0031802
3: -0.0054456, -0.0024004, -0.0057239, -0.0022178, -0.0029801, 0.0032891
4: -0.0014384, 0.0018582, -0.0016361, 0.0021595, -0.0035606, 0.0032261
5: 0.0023548, 0.0054746, 0.0020698, 0.0056616, -0.0030530, 0.0033696
6: -0.0129570, -0.0005788, -0.0140881, 0.0001633, -0.0121134, 0.0133694
7: -0.0017685, 0.0150896, -0.0027791, 0.0166301, -0.0182080, 0.0164974
8: 0.9879681, 0.9998432, 0.9872562, 1.0009285, -0.0128261, 0.0116211
9: -0.0157450, -0.0049655, -0.0167301, -0.0043193, -0.0105489, 0.0116427

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B2_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0067998
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0067998
time: 1.12 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0043278, 0.0097898, 0.0049991, 0.0100855, -0.0053335, 0.0043740
1: 0.0019475, 0.0027366, 0.0020445, 0.0027794, -0.0007705, 0.0006319
2: 0.0089473, 0.0119671, 0.0087839, 0.0115960, -0.0024182, 0.0029487
3: -0.0054267, -0.0023035, -0.0055958, -0.0026873, -0.0025011, 0.0030497
4: -0.0015433, 0.0018378, -0.0011278, 0.0020208, -0.0033015, 0.0027076
5: 0.0023742, 0.0055738, 0.0022010, 0.0051806, -0.0025623, 0.0031243
6: -0.0128802, -0.0001850, -0.0135674, -0.0017453, -0.0101663, 0.0123964
7: -0.0023048, 0.0149850, -0.0001798, 0.0159209, -0.0168829, 0.0138456
8: 0.9875903, 0.9997696, 0.9890872, 1.0004289, -0.0118926, 0.0097531
9: -0.0156782, -0.0046226, -0.0162766, -0.0059814, -0.0088532, 0.0107954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060753, upper bound: 0.0060054
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0060908
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0042065, 0.0097988, 0.0047473, 0.0099192, -0.0055012, 0.0044426
1: 0.0019300, 0.0027379, 0.0020082, 0.0027553, -0.0007948, 0.0006418
2: 0.0089424, 0.0120342, 0.0088758, 0.0117352, -0.0024562, 0.0030415
3: -0.0054319, -0.0022341, -0.0055007, -0.0025434, -0.0025403, 0.0031456
4: -0.0016184, 0.0018433, -0.0012836, 0.0019179, -0.0034053, 0.0027500
5: 0.0023689, 0.0056449, 0.0022984, 0.0053281, -0.0026025, 0.0032226
6: -0.0129011, 0.0000969, -0.0131811, -0.0011601, -0.0103258, 0.0127862
7: -0.0026887, 0.0150135, -0.0009768, 0.0153947, -0.0174138, 0.0140629
8: 0.9873199, 0.9997897, 0.9885259, 1.0000583, -0.0122666, 0.0099062
9: -0.0156964, -0.0043771, -0.0159402, -0.0054718, -0.0089922, 0.0111348

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061899, upper bound: 0.0061692
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0062272
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0043278, 0.0097898, 0.0045726, 0.0100490, -0.0055717, 0.0050600
1: 0.0019475, 0.0027366, 0.0019829, 0.0027741, -0.0008049, 0.0007310
2: 0.0089473, 0.0119671, 0.0088041, 0.0118318, -0.0027975, 0.0030804
3: -0.0054267, -0.0023035, -0.0055749, -0.0024435, -0.0028933, 0.0031859
4: -0.0015433, 0.0018378, -0.0013917, 0.0019982, -0.0034490, 0.0031322
5: 0.0023742, 0.0055738, 0.0022224, 0.0054304, -0.0029641, 0.0032639
6: -0.0128802, -0.0001850, -0.0134825, -0.0007541, -0.0117608, 0.0129501
7: -0.0023048, 0.0149850, -0.0015297, 0.0158054, -0.0176369, 0.0160172
8: 0.9875903, 0.9997696, 0.9881364, 1.0003475, -0.0124238, 0.0112829
9: -0.0156782, -0.0046226, -0.0162027, -0.0051182, -0.0102418, 0.0112775

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060753, upper bound: 0.0061706
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0062300
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0042065, 0.0097988, 0.0042912, 0.0098990, -0.0056898, 0.0051387
1: 0.0019300, 0.0027379, 0.0019423, 0.0027524, -0.0008220, 0.0007424
2: 0.0089424, 0.0120342, 0.0088870, 0.0119874, -0.0028410, 0.0031457
3: -0.0054319, -0.0022341, -0.0054892, -0.0022826, -0.0029383, 0.0032535
4: -0.0016184, 0.0018433, -0.0015659, 0.0019054, -0.0035221, 0.0031809
5: 0.0023689, 0.0056449, 0.0023102, 0.0055953, -0.0030102, 0.0033331
6: -0.0129011, 0.0000969, -0.0131340, -0.0001000, -0.0119437, 0.0132246
7: -0.0026887, 0.0150135, -0.0024205, 0.0153307, -0.0180108, 0.0162663
8: 0.9873199, 0.9997897, 0.9875089, 1.0000131, -0.0126872, 0.0114583
9: -0.0156964, -0.0043771, -0.0158992, -0.0045486, -0.0104011, 0.0115166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061899, upper bound: 0.0063049
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0063448
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0043637, 0.0097845, 0.0049515, 0.0103799, -0.0057040, 0.0044352
1: 0.0019527, 0.0027359, 0.0020377, 0.0028219, -0.0008241, 0.0006408
2: 0.0089503, 0.0119473, 0.0086211, 0.0116223, -0.0024521, 0.0031536
3: -0.0054236, -0.0023240, -0.0057641, -0.0026601, -0.0025361, 0.0032616
4: -0.0015211, 0.0018345, -0.0011572, 0.0022031, -0.0035308, 0.0027455
5: 0.0023773, 0.0055528, 0.0020285, 0.0052085, -0.0025981, 0.0033414
6: -0.0128678, -0.0002684, -0.0142518, -0.0016347, -0.0103086, 0.0132576
7: -0.0021912, 0.0149681, -0.0003304, 0.0168530, -0.0180557, 0.0140394
8: 0.9876703, 0.9997576, 0.9889811, 1.0010855, -0.0127188, 0.0098897
9: -0.0156673, -0.0046952, -0.0168726, -0.0058851, -0.0089772, 0.0115453

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060454, upper bound: 0.0061311
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0062403
time: 0.88 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0042394, 0.0097935, 0.0046723, 0.0102470, -0.0058711, 0.0045086
1: 0.0019348, 0.0027372, 0.0019973, 0.0028027, -0.0008482, 0.0006514
2: 0.0089453, 0.0120160, 0.0086946, 0.0117767, -0.0024927, 0.0032460
3: -0.0054288, -0.0022529, -0.0056881, -0.0025004, -0.0025781, 0.0033572
4: -0.0015980, 0.0018401, -0.0013301, 0.0021208, -0.0036343, 0.0027909
5: 0.0023720, 0.0056256, 0.0021064, 0.0053720, -0.0026411, 0.0034393
6: -0.0128889, 0.0000205, -0.0139429, -0.0009856, -0.0104793, 0.0136461
7: -0.0025846, 0.0149968, -0.0012144, 0.0164323, -0.0185849, 0.0142719
8: 0.9873933, 0.9997779, 0.9883584, 1.0007892, -0.0130916, 0.0100534
9: -0.0156857, -0.0044437, -0.0166036, -0.0053198, -0.0091258, 0.0118837

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061849, upper bound: 0.0063573
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0064313
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0043637, 0.0097845, 0.0044798, 0.0103881, -0.0058960, 0.0051185
1: 0.0019527, 0.0027359, 0.0019695, 0.0028231, -0.0008518, 0.0007395
2: 0.0089503, 0.0119473, 0.0086166, 0.0118831, -0.0028299, 0.0032598
3: -0.0054236, -0.0023240, -0.0057688, -0.0023904, -0.0029268, 0.0033714
4: -0.0015211, 0.0018345, -0.0014492, 0.0022081, -0.0036497, 0.0031684
5: 0.0023773, 0.0055528, 0.0020237, 0.0054848, -0.0029984, 0.0034539
6: -0.0128678, -0.0002684, -0.0142707, -0.0005384, -0.0118967, 0.0137040
7: -0.0021912, 0.0149681, -0.0018235, 0.0168788, -0.0186636, 0.0162023
8: 0.9876703, 0.9997576, 0.9879294, 1.0011036, -0.0131470, 0.0114132
9: -0.0156673, -0.0046952, -0.0168891, -0.0049304, -0.0103602, 0.0119340

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B2_B2_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062421, upper bound: 0.0062853
time: 0.86 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0065216
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0042394, 0.0097935, 0.0041918, 0.0102505, -0.0060111, 0.0052046
1: 0.0019348, 0.0027372, 0.0019279, 0.0028032, -0.0008684, 0.0007519
2: 0.0089453, 0.0120160, 0.0086926, 0.0120423, -0.0028775, 0.0033234
3: -0.0054288, -0.0022529, -0.0056902, -0.0022257, -0.0029760, 0.0034372
4: -0.0015980, 0.0018401, -0.0016275, 0.0021230, -0.0037210, 0.0032217
5: 0.0023720, 0.0056256, 0.0021043, 0.0056535, -0.0030489, 0.0035213
6: -0.0128889, 0.0000205, -0.0139511, 0.0001311, -0.0120970, 0.0139715
7: -0.0025846, 0.0149968, -0.0027352, 0.0164434, -0.0190280, 0.0164750
8: 0.9873933, 0.9997779, 0.9872871, 1.0007970, -0.0134037, 0.0116053
9: -0.0156857, -0.0044437, -0.0166107, -0.0043474, -0.0105346, 0.0121670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061849, upper bound: 0.0066270
time: 1.07 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0066906
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0045605, 0.0101013, 0.0045995, 0.0098181, -0.0050676, 0.0052784
1: 0.0019812, 0.0027816, 0.0019868, 0.0027407, -0.0007321, 0.0007626
2: 0.0087751, 0.0118385, 0.0089317, 0.0118169, -0.0029183, 0.0028017
3: -0.0056048, -0.0024365, -0.0054429, -0.0024588, -0.0030182, 0.0028977
4: -0.0013993, 0.0020306, -0.0013751, 0.0018553, -0.0031369, 0.0032674
5: 0.0021917, 0.0054375, 0.0023576, 0.0054147, -0.0030921, 0.0029686
6: -0.0136041, -0.0007258, -0.0129460, -0.0008165, -0.0122684, 0.0117784
7: -0.0015683, 0.0159709, -0.0014447, 0.0150746, -0.0160412, 0.0167084
8: 0.9881092, 1.0004641, 0.9881962, 0.9998327, -0.0112998, 0.0117698
9: -0.0163086, -0.0050936, -0.0157355, -0.0051726, -0.0106838, 0.0102572

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063879, upper bound: 0.0064633
time: 0.96 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063879, upper bound: 0.0065309
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0042780, 0.0099552, 0.0044689, 0.0098281, -0.0051415, 0.0054073
1: 0.0019403, 0.0027605, 0.0019679, 0.0027422, -0.0007428, 0.0007812
2: 0.0088559, 0.0119947, 0.0089261, 0.0118891, -0.0029896, 0.0028426
3: -0.0055213, -0.0022750, -0.0054486, -0.0023842, -0.0030920, 0.0029399
4: -0.0015741, 0.0019401, -0.0014559, 0.0018615, -0.0031827, 0.0033472
5: 0.0022773, 0.0056030, 0.0023518, 0.0054912, -0.0031676, 0.0030119
6: -0.0132645, -0.0000693, -0.0129693, -0.0005131, -0.0125681, 0.0119502
7: -0.0024623, 0.0155084, -0.0018580, 0.0151063, -0.0162751, 0.0171167
8: 0.9874793, 1.0001384, 0.9879051, 0.9998550, -0.0114645, 0.0120573
9: -0.0160129, -0.0045219, -0.0157557, -0.0049083, -0.0109449, 0.0104067

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061899
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0064313
time: 1.04 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: 0.0044670, 0.0104468, 0.0046303, 0.0098127, -0.0051272, 0.0056157
1: 0.0019676, 0.0028316, 0.0019912, 0.0027400, -0.0007407, 0.0008113
2: 0.0085841, 0.0118902, 0.0089347, 0.0117999, -0.0031048, 0.0028347
3: -0.0058024, -0.0023831, -0.0054398, -0.0024764, -0.0032111, 0.0029318
4: -0.0014572, 0.0022445, -0.0013561, 0.0018520, -0.0031738, 0.0034762
5: 0.0019893, 0.0054923, 0.0023608, 0.0053966, -0.0032896, 0.0030035
6: -0.0144073, -0.0005084, -0.0129335, -0.0008880, -0.0130524, 0.0119169
7: -0.0018643, 0.0170648, -0.0013473, 0.0150576, -0.0162298, 0.0177762
8: 0.9879007, 1.0012348, 0.9882648, 0.9998208, -0.0114326, 0.0125219
9: -0.0170081, -0.0049043, -0.0157246, -0.0052348, -0.0113666, 0.0103778

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066777, upper bound: 0.0064633
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066777, upper bound: 0.0065309
time: 1.14 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: 0.0041779, 0.0103095, 0.0044972, 0.0098228, -0.0052117, 0.0057521
1: 0.0019259, 0.0028117, 0.0019720, 0.0027414, -0.0007529, 0.0008310
2: 0.0086600, 0.0120500, 0.0089291, 0.0118735, -0.0031802, 0.0028814
3: -0.0057239, -0.0022178, -0.0054456, -0.0024004, -0.0032891, 0.0029801
4: -0.0016361, 0.0021595, -0.0014384, 0.0018582, -0.0032261, 0.0035606
5: 0.0020698, 0.0056616, 0.0023548, 0.0054746, -0.0033696, 0.0030530
6: -0.0140881, 0.0001633, -0.0129570, -0.0005788, -0.0133694, 0.0121134
7: -0.0027791, 0.0166301, -0.0017685, 0.0150896, -0.0164974, 0.0182080
8: 0.9872562, 1.0009285, 0.9879681, 0.9998432, -0.0116211, 0.0128261
9: -0.0167301, -0.0043193, -0.0157450, -0.0049655, -0.0116427, 0.0105489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A2_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067998, upper bound: 0.0065539
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067998, upper bound: 0.0066430
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0045726, 0.0100490, 0.0043278, 0.0097898, -0.0050600, 0.0055717
1: 0.0019829, 0.0027741, 0.0019475, 0.0027366, -0.0007310, 0.0008049
2: 0.0088041, 0.0118318, 0.0089473, 0.0119671, -0.0030804, 0.0027975
3: -0.0055749, -0.0024435, -0.0054267, -0.0023035, -0.0031859, 0.0028933
4: -0.0013917, 0.0019982, -0.0015433, 0.0018378, -0.0031322, 0.0034490
5: 0.0022224, 0.0054304, 0.0023742, 0.0055738, -0.0032639, 0.0029641
6: -0.0134825, -0.0007541, -0.0128802, -0.0001850, -0.0129501, 0.0117608
7: -0.0015297, 0.0158054, -0.0023048, 0.0149850, -0.0160172, 0.0176369
8: 0.9881364, 1.0003475, 0.9875903, 0.9997696, -0.0112829, 0.0124238
9: -0.0162027, -0.0051182, -0.0156782, -0.0046226, -0.0112775, 0.0102418

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061706, upper bound: 0.0060753
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062300, upper bound: 0.0063210
time: 0.88 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0042912, 0.0098990, 0.0042065, 0.0097988, -0.0051387, 0.0056898
1: 0.0019423, 0.0027524, 0.0019300, 0.0027379, -0.0007424, 0.0008220
2: 0.0088870, 0.0119874, 0.0089424, 0.0120342, -0.0031457, 0.0028410
3: -0.0054892, -0.0022826, -0.0054319, -0.0022341, -0.0032535, 0.0029383
4: -0.0015659, 0.0019054, -0.0016184, 0.0018433, -0.0031809, 0.0035221
5: 0.0023102, 0.0055953, 0.0023689, 0.0056449, -0.0033331, 0.0030102
6: -0.0131340, -0.0001000, -0.0129011, 0.0000969, -0.0132246, 0.0119437
7: -0.0024205, 0.0153307, -0.0026887, 0.0150135, -0.0162663, 0.0180108
8: 0.9875089, 1.0000131, 0.9873199, 0.9997897, -0.0114583, 0.0126872
9: -0.0158992, -0.0045486, -0.0156964, -0.0043771, -0.0115166, 0.0104011

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063049, upper bound: 0.0061899
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063448, upper bound: 0.0064313
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0044798, 0.0103881, 0.0043637, 0.0097845, -0.0051185, 0.0058960
1: 0.0019695, 0.0028231, 0.0019527, 0.0027359, -0.0007395, 0.0008518
2: 0.0086166, 0.0118831, 0.0089503, 0.0119473, -0.0032598, 0.0028299
3: -0.0057688, -0.0023904, -0.0054236, -0.0023240, -0.0033714, 0.0029268
4: -0.0014492, 0.0022081, -0.0015211, 0.0018345, -0.0031684, 0.0036497
5: 0.0020237, 0.0054848, 0.0023773, 0.0055528, -0.0034539, 0.0029984
6: -0.0142707, -0.0005384, -0.0128678, -0.0002684, -0.0137040, 0.0118967
7: -0.0018235, 0.0168788, -0.0021912, 0.0149681, -0.0162023, 0.0186636
8: 0.9879294, 1.0011036, 0.9876703, 0.9997576, -0.0114132, 0.0131470
9: -0.0168891, -0.0049304, -0.0156673, -0.0046952, -0.0119340, 0.0103602

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A2_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062853, upper bound: 0.0062421
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065216, upper bound: 0.0063210
time: 0.85 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0041918, 0.0102505, 0.0042394, 0.0097935, -0.0052046, 0.0060111
1: 0.0019279, 0.0028032, 0.0019348, 0.0027372, -0.0007519, 0.0008684
2: 0.0086926, 0.0120423, 0.0089453, 0.0120160, -0.0033234, 0.0028775
3: -0.0056902, -0.0022257, -0.0054288, -0.0022529, -0.0034372, 0.0029760
4: -0.0016275, 0.0021230, -0.0015980, 0.0018401, -0.0032217, 0.0037210
5: 0.0021043, 0.0056535, 0.0023720, 0.0056256, -0.0035213, 0.0030489
6: -0.0139511, 0.0001311, -0.0128889, 0.0000205, -0.0139715, 0.0120970
7: -0.0027352, 0.0164434, -0.0025846, 0.0149968, -0.0164750, 0.0190280
8: 0.9872871, 1.0007970, 0.9873933, 0.9997779, -0.0116053, 0.0134037
9: -0.0166107, -0.0043474, -0.0156857, -0.0044437, -0.0121670, 0.0105346

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066270, upper bound: 0.0061849
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066906, upper bound: 0.0064313
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0041293, 0.0097903, 0.0045605, 0.0101013, -0.0052441, 0.0045555
1: 0.0019189, 0.0027367, 0.0019812, 0.0027816, -0.0007576, 0.0006581
2: 0.0089471, 0.0120769, 0.0087751, 0.0118385, -0.0025186, 0.0028993
3: -0.0054270, -0.0021900, -0.0056048, -0.0024365, -0.0026049, 0.0029986
4: -0.0016662, 0.0018381, -0.0013993, 0.0020306, -0.0032462, 0.0028199
5: 0.0023739, 0.0056901, 0.0021917, 0.0054375, -0.0026686, 0.0030720
6: -0.0128814, 0.0002763, -0.0136041, -0.0007258, -0.0105883, 0.0121887
7: -0.0029331, 0.0149866, -0.0015683, 0.0159709, -0.0165999, 0.0144203
8: 0.9871478, 0.9997708, 0.9881092, 1.0004641, -0.0116933, 0.0101580
9: -0.0156792, -0.0042209, -0.0163086, -0.0050936, -0.0092208, 0.0106144

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0063245
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0063244
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0039907, 0.0097994, 0.0042780, 0.0099552, -0.0054108, 0.0046505
1: 0.0018988, 0.0027380, 0.0019403, 0.0027605, -0.0007817, 0.0006719
2: 0.0089420, 0.0121535, 0.0088559, 0.0119947, -0.0025711, 0.0029915
3: -0.0054322, -0.0021108, -0.0055213, -0.0022750, -0.0026592, 0.0030940
4: -0.0017519, 0.0018437, -0.0015741, 0.0019401, -0.0033494, 0.0028787
5: 0.0023686, 0.0057713, 0.0022773, 0.0056030, -0.0027242, 0.0031696
6: -0.0129024, 0.0005984, -0.0132645, -0.0000693, -0.0108090, 0.0125762
7: -0.0033717, 0.0150153, -0.0024623, 0.0155084, -0.0171278, 0.0147209
8: 0.9868388, 0.9997910, 0.9874793, 1.0001384, -0.0120651, 0.0103697
9: -0.0156975, -0.0039404, -0.0160129, -0.0045219, -0.0094129, 0.0109519

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065019, upper bound: 0.0060922
time: 0.84 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066935, upper bound: 0.0061086
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0041510, 0.0097851, 0.0044670, 0.0104468, -0.0056128, 0.0046322
1: 0.0019220, 0.0027360, 0.0019676, 0.0028316, -0.0008109, 0.0006692
2: 0.0089499, 0.0120649, 0.0085841, 0.0118902, -0.0025610, 0.0031031
3: -0.0054240, -0.0022024, -0.0058024, -0.0023831, -0.0026487, 0.0032094
4: -0.0016527, 0.0018349, -0.0014572, 0.0022445, -0.0034744, 0.0028674
5: 0.0023770, 0.0056774, 0.0019893, 0.0054923, -0.0027135, 0.0032879
6: -0.0128692, 0.0002259, -0.0144073, -0.0005084, -0.0107665, 0.0130456
7: -0.0028644, 0.0149701, -0.0018643, 0.0170648, -0.0177670, 0.0146630
8: 0.9871961, 0.9997592, 0.9879007, 1.0012348, -0.0125154, 0.0103290
9: -0.0156686, -0.0042648, -0.0170081, -0.0049043, -0.0093759, 0.0113607

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0064780
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0064780
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0040117, 0.0097942, 0.0041779, 0.0103095, -0.0057814, 0.0047335
1: 0.0019019, 0.0027373, 0.0019259, 0.0028117, -0.0008352, 0.0006839
2: 0.0089449, 0.0121419, 0.0086600, 0.0120500, -0.0026170, 0.0031964
3: -0.0054292, -0.0021227, -0.0057239, -0.0022178, -0.0027067, 0.0033059
4: -0.0017390, 0.0018405, -0.0016361, 0.0021595, -0.0035788, 0.0029301
5: 0.0023716, 0.0057590, 0.0020698, 0.0056616, -0.0027729, 0.0033867
6: -0.0128903, 0.0005497, -0.0140881, 0.0001633, -0.0110020, 0.0134376
7: -0.0033054, 0.0149988, -0.0027791, 0.0166301, -0.0183009, 0.0149838
8: 0.9868855, 0.9997794, 0.9872562, 1.0009285, -0.0128915, 0.0105549
9: -0.0156870, -0.0039828, -0.0167301, -0.0043193, -0.0095810, 0.0117021

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068083, upper bound: 0.0065618
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0068083, upper bound: 0.0065618
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0038776, 0.0097661, 0.0045726, 0.0100490, -0.0055197, 0.0045489
1: 0.0018825, 0.0027332, 0.0019829, 0.0027741, -0.0007974, 0.0006572
2: 0.0089605, 0.0122161, 0.0088041, 0.0118318, -0.0025149, 0.0030517
3: -0.0054131, -0.0020460, -0.0055749, -0.0024435, -0.0026011, 0.0031562
4: -0.0018220, 0.0018231, -0.0013917, 0.0019982, -0.0034168, 0.0028158
5: 0.0023881, 0.0058376, 0.0022224, 0.0054304, -0.0026647, 0.0032335
6: -0.0128250, 0.0008615, -0.0134825, -0.0007541, -0.0105728, 0.0128294
7: -0.0037300, 0.0149099, -0.0015297, 0.0158054, -0.0174725, 0.0143992
8: 0.9865863, 0.9997167, 0.9881364, 1.0003475, -0.0123080, 0.0101431
9: -0.0156301, -0.0037113, -0.0162027, -0.0051182, -0.0092072, 0.0111724

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064185, upper bound: 0.0061330
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066395, upper bound: 0.0061697
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0037401, 0.0097745, 0.0042912, 0.0098990, -0.0056819, 0.0046401
1: 0.0018626, 0.0027344, 0.0019423, 0.0027524, -0.0008209, 0.0006704
2: 0.0089558, 0.0122921, 0.0088870, 0.0119874, -0.0025654, 0.0031414
3: -0.0054179, -0.0019674, -0.0054892, -0.0022826, -0.0026532, 0.0032490
4: -0.0019071, 0.0018283, -0.0015659, 0.0019054, -0.0035172, 0.0028723
5: 0.0023832, 0.0059181, 0.0023102, 0.0055953, -0.0027181, 0.0033285
6: -0.0128446, 0.0011810, -0.0131340, -0.0001000, -0.0107848, 0.0132063
7: -0.0041652, 0.0149365, -0.0024205, 0.0153307, -0.0179859, 0.0146880
8: 0.9862798, 0.9997354, 0.9875089, 1.0000131, -0.0126696, 0.0103465
9: -0.0156471, -0.0034330, -0.0158992, -0.0045486, -0.0093919, 0.0115006

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065019, upper bound: 0.0062074
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066935, upper bound: 0.0062289
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0038996, 0.0097606, 0.0044798, 0.0103881, -0.0058770, 0.0046235
1: 0.0018857, 0.0027324, 0.0019695, 0.0028231, -0.0008491, 0.0006680
2: 0.0089635, 0.0122039, 0.0086166, 0.0118831, -0.0025562, 0.0032493
3: -0.0054100, -0.0020586, -0.0057688, -0.0023904, -0.0026437, 0.0033605
4: -0.0018084, 0.0018197, -0.0014492, 0.0022081, -0.0036380, 0.0028620
5: 0.0023913, 0.0058247, 0.0020237, 0.0054848, -0.0027084, 0.0034428
6: -0.0128124, 0.0008103, -0.0142707, -0.0005384, -0.0107462, 0.0136599
7: -0.0036603, 0.0148927, -0.0018235, 0.0168788, -0.0186036, 0.0146354
8: 0.9866354, 0.9997046, 0.9879294, 1.0011036, -0.0131047, 0.0103095
9: -0.0156191, -0.0037558, -0.0168891, -0.0049304, -0.0093583, 0.0118956

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064152, upper bound: 0.0062842
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066395, upper bound: 0.0063307
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0037611, 0.0097691, 0.0041918, 0.0102505, -0.0060402, 0.0047225
1: 0.0018657, 0.0027337, 0.0019279, 0.0028032, -0.0008726, 0.0006823
2: 0.0089588, 0.0122804, 0.0086926, 0.0120423, -0.0026110, 0.0033395
3: -0.0054149, -0.0019794, -0.0056902, -0.0022257, -0.0027004, 0.0034538
4: -0.0018941, 0.0018250, -0.0016275, 0.0021230, -0.0037390, 0.0029233
5: 0.0023863, 0.0059058, 0.0021043, 0.0056535, -0.0027665, 0.0035383
6: -0.0128321, 0.0011322, -0.0139511, 0.0001311, -0.0109765, 0.0140391
7: -0.0040986, 0.0149195, -0.0027352, 0.0164434, -0.0191200, 0.0149490
8: 0.9863267, 0.9997236, 0.9872871, 1.0007970, -0.0134686, 0.0105304
9: -0.0156363, -0.0034756, -0.0166107, -0.0043474, -0.0095588, 0.0122259

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065019, upper bound: 0.0064238
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066935, upper bound: 0.0064527
time: 0.90 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 3.80 seconds
IS_A1_A1_B1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0062638
IS_A1_A1_B1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0062638
IS_A1_A1_B1_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0063745
IS_A1_A1_B1_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0063745
IS_A1_A1_B1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063879
IS_A1_A1_B1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063879
IS_A1_A1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0061899, upper bound: 0.0061821
IS_A1_A1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0062145
IS_A1_A1_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063930
IS_A1_A1_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0063930
IS_A1_A1_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0065429
IS_A1_A1_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0065429
IS_A1_A1_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0066777
IS_A1_A1_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064633, upper bound: 0.0066777
IS_A1_A1_B2_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0067998
IS_A1_A1_B2_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065539, upper bound: 0.0067998
IS_A1_A2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0060753, upper bound: 0.0060054
IS_A1_A2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0060908
IS_A1_A2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0061899, upper bound: 0.0061692
IS_A1_A2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0062272
IS_A1_A2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0060753, upper bound: 0.0061706
IS_A1_A2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0062300
IS_A1_A2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0061899, upper bound: 0.0063049
IS_A1_A2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0063448
IS_A1_A2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0060454, upper bound: 0.0061311
IS_A1_A2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0062403
IS_A1_A2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0061849, upper bound: 0.0063573
IS_A1_A2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0064313
IS_A1_A2_B2_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0062421, upper bound: 0.0062853
IS_A1_A2_B2_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063210, upper bound: 0.0065216
IS_A1_A2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0061849, upper bound: 0.0066270
IS_A1_A2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064313, upper bound: 0.0066906
IS_A2_B1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063879, upper bound: 0.0064633
IS_A2_B1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063879, upper bound: 0.0065309
IS_A2_B1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061899
IS_A2_B1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0064313
IS_A2_B1_B1_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066777, upper bound: 0.0064633
IS_A2_B1_B1_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066777, upper bound: 0.0065309
IS_A2_B1_B1_A2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0067998, upper bound: 0.0065539
IS_A2_B1_B1_A2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0067998, upper bound: 0.0066430
IS_A2_B1_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0061706, upper bound: 0.0060753
IS_A2_B1_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0062300, upper bound: 0.0063210
IS_A2_B1_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063049, upper bound: 0.0061899
IS_A2_B1_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0063448, upper bound: 0.0064313
IS_A2_B1_B2_A2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0062853, upper bound: 0.0062421
IS_A2_B1_B2_A2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065216, upper bound: 0.0063210
IS_A2_B1_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066270, upper bound: 0.0061849
IS_A2_B1_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066906, upper bound: 0.0064313
IS_A2_B2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0063245
IS_A2_B2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0063244
IS_A2_B2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065019, upper bound: 0.0060922
IS_A2_B2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066935, upper bound: 0.0061086
IS_A2_B2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0064780
IS_A2_B2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0067697, upper bound: 0.0064780
IS_A2_B2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0068083, upper bound: 0.0065618
IS_A2_B2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0068083, upper bound: 0.0065618
IS_A2_B2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064185, upper bound: 0.0061330
IS_A2_B2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066395, upper bound: 0.0061697
IS_A2_B2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065019, upper bound: 0.0062074
IS_A2_B2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066935, upper bound: 0.0062289
IS_A2_B2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0064152, upper bound: 0.0062842
IS_A2_B2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066395, upper bound: 0.0063307
IS_A2_B2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0065019, upper bound: 0.0064238
IS_A2_B2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.80
Output dim: 8, lower bound: -0.0066935, upper bound: 0.0064527

## BFS IS instance: IS_A1_A1_B1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0045995, 0.0098181, 0.0050264, 0.0099710, -0.0048550, 0.0041207
1: 0.0019868, 0.0027407, 0.0020485, 0.0027628, -0.0007014, 0.0005953
2: 0.0089317, 0.0118169, 0.0088471, 0.0115809, -0.0022782, 0.0026842
3: -0.0054429, -0.0024588, -0.0055303, -0.0027030, -0.0023563, 0.0027762
4: -0.0013751, 0.0018553, -0.0011108, 0.0019499, -0.0030053, 0.0025508
5: 0.0023576, 0.0054147, 0.0022680, 0.0051646, -0.0024139, 0.0028441
6: -0.0129460, -0.0008165, -0.0133014, -0.0018088, -0.0095776, 0.0112844
7: -0.0014447, 0.0150746, -0.0000933, 0.0155586, -0.0153684, 0.0130439
8: 0.9881962, 0.9998327, 0.9891482, 1.0001737, -0.0108258, 0.0091884
9: -0.0157355, -0.0051726, -0.0160450, -0.0060367, -0.0083406, 0.0098270

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0059157
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0059963
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0045995, 0.0098181, 0.0047361, 0.0099572, -0.0048901, 0.0045078
1: 0.0019868, 0.0027407, 0.0020065, 0.0027608, -0.0007065, 0.0006512
2: 0.0089317, 0.0118169, 0.0088548, 0.0117414, -0.0024922, 0.0027036
3: -0.0054429, -0.0024588, -0.0055224, -0.0025370, -0.0025776, 0.0027962
4: -0.0013751, 0.0018553, -0.0012906, 0.0019414, -0.0030271, 0.0027904
5: 0.0023576, 0.0054147, 0.0022762, 0.0053346, -0.0026406, 0.0028646
6: -0.0129460, -0.0008165, -0.0132692, -0.0011340, -0.0104773, 0.0113660
7: -0.0014447, 0.0150746, -0.0010123, 0.0155147, -0.0154795, 0.0142692
8: 0.9881962, 0.9998327, 0.9885008, 1.0001428, -0.0109041, 0.0100515
9: -0.0157355, -0.0051726, -0.0160169, -0.0054491, -0.0091241, 0.0098980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0059157
time: 1.08 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0059963
time: 1.14 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0044689, 0.0098281, 0.0047737, 0.0098080, -0.0050416, 0.0044136
1: 0.0019679, 0.0027422, 0.0020120, 0.0027393, -0.0007284, 0.0006376
2: 0.0089261, 0.0118891, 0.0089373, 0.0117206, -0.0024401, 0.0027874
3: -0.0054486, -0.0023842, -0.0054371, -0.0025585, -0.0025237, 0.0028829
4: -0.0014559, 0.0018615, -0.0012673, 0.0018490, -0.0031209, 0.0027321
5: 0.0023518, 0.0054912, 0.0023635, 0.0053126, -0.0025855, 0.0029534
6: -0.0129693, -0.0005131, -0.0129225, -0.0012214, -0.0102584, 0.0117182
7: -0.0018580, 0.0151063, -0.0008933, 0.0150426, -0.0159591, 0.0139710
8: 0.9879051, 0.9998550, 0.9885846, 0.9998102, -0.0112420, 0.0098415
9: -0.0157557, -0.0049083, -0.0157150, -0.0055252, -0.0089334, 0.0102047

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0060712
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0061202
time: 1.20 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0044689, 0.0098281, 0.0045036, 0.0097787, -0.0050540, 0.0047686
1: 0.0019679, 0.0027422, 0.0019729, 0.0027350, -0.0007302, 0.0006889
2: 0.0089261, 0.0118891, 0.0089535, 0.0118700, -0.0026364, 0.0027942
3: -0.0054486, -0.0023842, -0.0054204, -0.0024040, -0.0027267, 0.0028899
4: -0.0014559, 0.0018615, -0.0014345, 0.0018309, -0.0031285, 0.0029518
5: 0.0023518, 0.0054912, 0.0023807, 0.0054709, -0.0027934, 0.0029606
6: -0.0129693, -0.0005131, -0.0128545, -0.0005935, -0.0110836, 0.0117469
7: -0.0018580, 0.0151063, -0.0017484, 0.0149499, -0.0159982, 0.0150949
8: 0.9879051, 0.9998550, 0.9879823, 0.9997450, -0.0112695, 0.0106331
9: -0.0157557, -0.0049083, -0.0156557, -0.0049784, -0.0096521, 0.0102297

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B1_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0060712
time: 1.10 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0061202
time: 1.31 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0045995, 0.0098181, 0.0046001, 0.0099191, -0.0051042, 0.0050351
1: 0.0019868, 0.0027407, 0.0019869, 0.0027553, -0.0007374, 0.0007274
2: 0.0089317, 0.0118169, 0.0088758, 0.0118166, -0.0027838, 0.0028220
3: -0.0054429, -0.0024588, -0.0055007, -0.0024592, -0.0028791, 0.0029186
4: -0.0013751, 0.0018553, -0.0013747, 0.0019178, -0.0031596, 0.0031168
5: 0.0023576, 0.0054147, 0.0022984, 0.0054143, -0.0029496, 0.0029900
6: -0.0129460, -0.0008165, -0.0131808, -0.0008180, -0.0117030, 0.0118636
7: -0.0014447, 0.0150746, -0.0014426, 0.0153944, -0.0161572, 0.0159385
8: 0.9881962, 0.9998327, 0.9881977, 1.0000581, -0.0113815, 0.0112274
9: -0.0157355, -0.0051726, -0.0159399, -0.0051739, -0.0101915, 0.0103314

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0060605
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061146
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0045995, 0.0098181, 0.0043237, 0.0099115, -0.0051030, 0.0053538
1: 0.0019868, 0.0027407, 0.0019470, 0.0027542, -0.0007372, 0.0007735
2: 0.0089317, 0.0118169, 0.0088800, 0.0119694, -0.0029600, 0.0028213
3: -0.0054429, -0.0024588, -0.0054963, -0.0023012, -0.0030614, 0.0029180
4: -0.0013751, 0.0018553, -0.0015458, 0.0019131, -0.0031589, 0.0033141
5: 0.0023576, 0.0054147, 0.0023029, 0.0055762, -0.0031363, 0.0029893
6: -0.0129460, -0.0008165, -0.0131632, -0.0001755, -0.0124438, 0.0118609
7: -0.0014447, 0.0150746, -0.0023176, 0.0153704, -0.0161535, 0.0169474
8: 0.9881962, 0.9998327, 0.9875812, 1.0000410, -0.0113788, 0.0119381
9: -0.0157355, -0.0051726, -0.0159246, -0.0046144, -0.0108366, 0.0103290

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0060605
time: 1.05 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061146
time: 1.34 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0048611, 0.0100234, 0.0044172, 0.0099466, -0.0048575, 0.0049622
1: 0.0020246, 0.0027704, 0.0019605, 0.0027593, -0.0007018, 0.0007169
2: 0.0088182, 0.0116723, 0.0088606, 0.0119177, -0.0027435, 0.0026856
3: -0.0055603, -0.0026084, -0.0055164, -0.0023546, -0.0028374, 0.0027776
4: -0.0012132, 0.0019824, -0.0014879, 0.0019348, -0.0030069, 0.0030717
5: 0.0022373, 0.0052614, 0.0022823, 0.0055214, -0.0029069, 0.0028455
6: -0.0134232, -0.0014246, -0.0132447, -0.0003929, -0.0115336, 0.0112901
7: -0.0006166, 0.0157245, -0.0020217, 0.0154814, -0.0153762, 0.0157078
8: 0.9887796, 1.0002905, 0.9877898, 1.0001193, -0.0108313, 0.0110649
9: -0.0161510, -0.0057021, -0.0159956, -0.0048037, -0.0100440, 0.0098319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0061821
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0061821
time: 1.16 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0045718, 0.0098221, 0.0042780, 0.0099552, -0.0049305, 0.0051358
1: 0.0019828, 0.0027413, 0.0019403, 0.0027605, -0.0007123, 0.0007420
2: 0.0089295, 0.0118322, 0.0088559, 0.0119947, -0.0028394, 0.0027259
3: -0.0054452, -0.0024430, -0.0055213, -0.0022750, -0.0029367, 0.0028193
4: -0.0013922, 0.0018577, -0.0015741, 0.0019401, -0.0030521, 0.0031791
5: 0.0023553, 0.0054309, 0.0022773, 0.0056030, -0.0030085, 0.0028883
6: -0.0129552, -0.0007522, -0.0132645, -0.0000693, -0.0119370, 0.0114598
7: -0.0015323, 0.0150871, -0.0024623, 0.0155084, -0.0156073, 0.0162572
8: 0.9881345, 0.9998416, 0.9874793, 1.0001384, -0.0109941, 0.0114519
9: -0.0157435, -0.0051166, -0.0160129, -0.0045219, -0.0103953, 0.0099797

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0062145
time: 1.02 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0062145
time: 1.19 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0046303, 0.0098127, 0.0049742, 0.0102649, -0.0052566, 0.0041870
1: 0.0019912, 0.0027400, 0.0020409, 0.0028053, -0.0007594, 0.0006049
2: 0.0089347, 0.0117999, 0.0086847, 0.0116098, -0.0023149, 0.0029062
3: -0.0054398, -0.0024764, -0.0056984, -0.0026731, -0.0023941, 0.0030058
4: -0.0013561, 0.0018520, -0.0011432, 0.0021319, -0.0032539, 0.0025918
5: 0.0023608, 0.0053966, 0.0020959, 0.0051952, -0.0024527, 0.0030793
6: -0.0129335, -0.0008880, -0.0139845, -0.0016874, -0.0097316, 0.0122178
7: -0.0013473, 0.0150576, -0.0002587, 0.0164890, -0.0166395, 0.0132536
8: 0.9882648, 0.9998208, 0.9890316, 1.0008290, -0.0117212, 0.0093361
9: -0.0157246, -0.0052348, -0.0166398, -0.0059309, -0.0084747, 0.0106398

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0060467
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061477
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0046303, 0.0098127, 0.0047046, 0.0102396, -0.0052394, 0.0045700
1: 0.0019912, 0.0027400, 0.0020020, 0.0028016, -0.0007569, 0.0006602
2: 0.0089347, 0.0117999, 0.0086987, 0.0117588, -0.0025267, 0.0028967
3: -0.0054398, -0.0024764, -0.0056839, -0.0025189, -0.0026132, 0.0029959
4: -0.0013561, 0.0018520, -0.0013101, 0.0021162, -0.0032433, 0.0028289
5: 0.0023608, 0.0053966, 0.0021107, 0.0053531, -0.0026771, 0.0030692
6: -0.0129335, -0.0008880, -0.0139257, -0.0010608, -0.0106220, 0.0121778
7: -0.0013473, 0.0150576, -0.0011121, 0.0164088, -0.0165850, 0.0144663
8: 0.9882648, 0.9998208, 0.9884305, 1.0007726, -0.0116829, 0.0101904
9: -0.0157246, -0.0052348, -0.0165886, -0.0053853, -0.0092501, 0.0106049

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0060467
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061477
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B1

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0046992, 0.0101375, -0.0054273, 0.0044824
1: 0.0019720, 0.0027414, 0.0020012, 0.0027869, -0.0007841, 0.0006476
2: 0.0089291, 0.0118735, 0.0087551, 0.0117618, -0.0024782, 0.0030006
3: -0.0054456, -0.0024004, -0.0056255, -0.0025158, -0.0025631, 0.0031034
4: -0.0014384, 0.0018582, -0.0013134, 0.0020530, -0.0033596, 0.0027747
5: 0.0023548, 0.0054746, 0.0021705, 0.0053563, -0.0026258, 0.0031793
6: -0.0129570, -0.0005788, -0.0136884, -0.0010482, -0.0104184, 0.0126145
7: -0.0017685, 0.0150896, -0.0011292, 0.0160857, -0.0171799, 0.0141889
8: 0.9879681, 0.9998432, 0.9884184, 1.0005449, -0.0121019, 0.0099950
9: -0.0157450, -0.0049655, -0.0163820, -0.0053743, -0.0090728, 0.0109853

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0062584
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0063211
time: 1.10 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B2

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0044480, 0.0100962, -0.0054139, 0.0048367
1: 0.0019720, 0.0027414, 0.0019649, 0.0027809, -0.0007821, 0.0006988
2: 0.0089291, 0.0118735, 0.0087779, 0.0119007, -0.0026741, 0.0029932
3: -0.0054456, -0.0024004, -0.0056019, -0.0023722, -0.0027657, 0.0030957
4: -0.0014384, 0.0018582, -0.0014689, 0.0020274, -0.0033513, 0.0029940
5: 0.0023548, 0.0054746, 0.0021947, 0.0055034, -0.0028333, 0.0031714
6: -0.0129570, -0.0005788, -0.0135923, -0.0004644, -0.0112419, 0.0125833
7: -0.0017685, 0.0150896, -0.0019243, 0.0159549, -0.0171374, 0.0153104
8: 0.9879681, 0.9998432, 0.9878585, 1.0004529, -0.0120719, 0.0107850
9: -0.0157450, -0.0049655, -0.0162983, -0.0048659, -0.0097899, 0.0109581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0062584
time: 1.04 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0063211
time: 1.10 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0046303, 0.0098127, 0.0045079, 0.0102772, -0.0054515, 0.0050935
1: 0.0019912, 0.0027400, 0.0019736, 0.0028071, -0.0007876, 0.0007359
2: 0.0089347, 0.0117999, 0.0086778, 0.0118675, -0.0028160, 0.0030140
3: -0.0054398, -0.0024764, -0.0057054, -0.0024065, -0.0029125, 0.0031172
4: -0.0013561, 0.0018520, -0.0014318, 0.0021395, -0.0033746, 0.0031529
5: 0.0023608, 0.0053966, 0.0020887, 0.0054683, -0.0029837, 0.0031935
6: -0.0129335, -0.0008880, -0.0140131, -0.0006037, -0.0118386, 0.0126708
7: -0.0013473, 0.0150576, -0.0017345, 0.0165280, -0.0172565, 0.0161231
8: 0.9882648, 0.9998208, 0.9879920, 1.0008565, -0.0121558, 0.0113575
9: -0.0157246, -0.0052348, -0.0166648, -0.0049873, -0.0103096, 0.0110343

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0063281
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0064169
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0046303, 0.0098127, 0.0042515, 0.0102460, -0.0054309, 0.0054139
1: 0.0019912, 0.0027400, 0.0019365, 0.0028026, -0.0007846, 0.0007821
2: 0.0089347, 0.0117999, 0.0086951, 0.0120093, -0.0029932, 0.0030026
3: -0.0054398, -0.0024764, -0.0056876, -0.0022599, -0.0030957, 0.0031054
4: -0.0013561, 0.0018520, -0.0015905, 0.0021202, -0.0033618, 0.0033513
5: 0.0023608, 0.0053966, 0.0021070, 0.0056185, -0.0031714, 0.0031814
6: -0.0129335, -0.0008880, -0.0139406, -0.0000077, -0.0125833, 0.0126229
7: -0.0013473, 0.0150576, -0.0025463, 0.0164291, -0.0171913, 0.0171373
8: 0.9882648, 0.9998208, 0.9874203, 1.0007869, -0.0121099, 0.0120719
9: -0.0157246, -0.0052348, -0.0166016, -0.0044682, -0.0109581, 0.0109926

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0063281
time: 1.05 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0064169
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0042184, 0.0101414, -0.0055855, 0.0051769
1: 0.0019720, 0.0027414, 0.0019317, 0.0027874, -0.0008069, 0.0007479
2: 0.0089291, 0.0118735, 0.0087530, 0.0120276, -0.0028622, 0.0030881
3: -0.0054456, -0.0024004, -0.0056277, -0.0022409, -0.0029602, 0.0031938
4: -0.0014384, 0.0018582, -0.0016110, 0.0020554, -0.0034575, 0.0032046
5: 0.0023548, 0.0054746, 0.0021682, 0.0056379, -0.0030326, 0.0032720
6: -0.0129570, -0.0005788, -0.0136974, 0.0000693, -0.0120325, 0.0129823
7: -0.0017685, 0.0150896, -0.0026512, 0.0160979, -0.0176807, 0.0163872
8: 0.9879681, 0.9998432, 0.9873464, 1.0005536, -0.0124547, 0.0115435
9: -0.0157450, -0.0049655, -0.0163898, -0.0044011, -0.0104784, 0.0113055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0065161
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0065661
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0044972, 0.0098228, 0.0039777, 0.0100976, -0.0055587, 0.0054806
1: 0.0019720, 0.0027414, 0.0018970, 0.0027811, -0.0008031, 0.0007918
2: 0.0089291, 0.0118735, 0.0087772, 0.0121607, -0.0030301, 0.0030733
3: -0.0054456, -0.0024004, -0.0056027, -0.0021033, -0.0031339, 0.0031785
4: -0.0014384, 0.0018582, -0.0017600, 0.0020283, -0.0034409, 0.0033926
5: 0.0023548, 0.0054746, 0.0021939, 0.0057789, -0.0032105, 0.0032563
6: -0.0129570, -0.0005788, -0.0135955, 0.0006288, -0.0127385, 0.0129199
7: -0.0017685, 0.0150896, -0.0034130, 0.0159592, -0.0175958, 0.0173488
8: 0.9879681, 0.9998432, 0.9868096, 1.0004559, -0.0123949, 0.0122208
9: -0.0157450, -0.0049655, -0.0163011, -0.0039140, -0.0110933, 0.0112512

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0065161
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0065661
time: 1.11 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0047105, 0.0099804, 0.0051158, 0.0100758, -0.0048178, 0.0041938
1: 0.0020028, 0.0027642, 0.0020614, 0.0027780, -0.0006960, 0.0006059
2: 0.0088419, 0.0117556, 0.0087892, 0.0115315, -0.0023187, 0.0026636
3: -0.0055357, -0.0025223, -0.0055902, -0.0027541, -0.0023981, 0.0027549
4: -0.0013064, 0.0019558, -0.0010555, 0.0020148, -0.0029823, 0.0025961
5: 0.0022625, 0.0053496, 0.0022067, 0.0051122, -0.0024567, 0.0028222
6: -0.0133232, -0.0010745, -0.0135449, -0.0020166, -0.0097476, 0.0111979
7: -0.0010933, 0.0155884, 0.0001897, 0.0158903, -0.0152505, 0.0132754
8: 0.9884437, 1.0001947, 0.9893475, 1.0004073, -0.0107428, 0.0093515
9: -0.0160640, -0.0053972, -0.0162570, -0.0062176, -0.0084887, 0.0097516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0057801
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0057930
time: 0.89 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0044325, 0.0097835, 0.0049991, 0.0100855, -0.0048766, 0.0043680
1: 0.0019627, 0.0027357, 0.0020445, 0.0027794, -0.0007045, 0.0006311
2: 0.0089508, 0.0119093, 0.0087839, 0.0115960, -0.0024150, 0.0026962
3: -0.0054231, -0.0023633, -0.0055958, -0.0026873, -0.0024977, 0.0027885
4: -0.0014785, 0.0018339, -0.0011278, 0.0020208, -0.0030187, 0.0027039
5: 0.0023779, 0.0055125, 0.0022010, 0.0051806, -0.0025588, 0.0028567
6: -0.0128657, -0.0004283, -0.0135674, -0.0017453, -0.0101525, 0.0113347
7: -0.0019734, 0.0149652, -0.0001798, 0.0159209, -0.0154368, 0.0138269
8: 0.9878238, 0.9997557, 0.9890872, 1.0004289, -0.0108740, 0.0097399
9: -0.0156655, -0.0048345, -0.0162766, -0.0059814, -0.0088413, 0.0098707

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0059212
time: 0.84 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0059212
time: 0.88 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0045959, 0.0099903, 0.0048753, 0.0099099, -0.0049735, 0.0042688
1: 0.0019863, 0.0027656, 0.0020266, 0.0027540, -0.0007185, 0.0006167
2: 0.0088365, 0.0118189, 0.0088809, 0.0116644, -0.0023601, 0.0027497
3: -0.0055414, -0.0024568, -0.0054954, -0.0026166, -0.0024409, 0.0028439
4: -0.0013773, 0.0019619, -0.0012044, 0.0019121, -0.0030787, 0.0026425
5: 0.0022567, 0.0054168, 0.0023039, 0.0052531, -0.0025006, 0.0029134
6: -0.0133462, -0.0008082, -0.0131593, -0.0014576, -0.0099218, 0.0115597
7: -0.0014560, 0.0156197, -0.0005715, 0.0153651, -0.0157433, 0.0135127
8: 0.9881883, 1.0002167, 0.9888113, 1.0000374, -0.0110899, 0.0095186
9: -0.0160840, -0.0051654, -0.0159212, -0.0057309, -0.0086404, 0.0100667

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059691
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059964
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043065, 0.0097927, 0.0047473, 0.0099192, -0.0050131, 0.0044368
1: 0.0019445, 0.0027371, 0.0020082, 0.0027553, -0.0007242, 0.0006410
2: 0.0089458, 0.0119789, 0.0088758, 0.0117352, -0.0024530, 0.0027716
3: -0.0054283, -0.0022913, -0.0055007, -0.0025434, -0.0025370, 0.0028665
4: -0.0015565, 0.0018395, -0.0012836, 0.0019179, -0.0031032, 0.0027464
5: 0.0023725, 0.0055863, 0.0022984, 0.0053281, -0.0025991, 0.0029366
6: -0.0128868, -0.0001354, -0.0131811, -0.0011601, -0.0103123, 0.0116517
7: -0.0023723, 0.0149940, -0.0009768, 0.0153947, -0.0158686, 0.0140444
8: 0.9875427, 0.9997759, 0.9885259, 1.0000583, -0.0111782, 0.0098932
9: -0.0156839, -0.0045794, -0.0159402, -0.0054718, -0.0089804, 0.0101468

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0061303
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B1_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062129
time: 1.17 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0047105, 0.0099804, 0.0046926, 0.0100401, -0.0050565, 0.0048901
1: 0.0020028, 0.0027642, 0.0020003, 0.0027728, -0.0007305, 0.0007065
2: 0.0088419, 0.0117556, 0.0088089, 0.0117654, -0.0027036, 0.0027956
3: -0.0055357, -0.0025223, -0.0055698, -0.0025121, -0.0027962, 0.0028913
4: -0.0013064, 0.0019558, -0.0013175, 0.0019927, -0.0031300, 0.0030271
5: 0.0022625, 0.0053496, 0.0022276, 0.0053601, -0.0028646, 0.0029621
6: -0.0133232, -0.0010745, -0.0134620, -0.0010330, -0.0113660, 0.0117527
7: -0.0010933, 0.0155884, -0.0011498, 0.0157774, -0.0160061, 0.0154795
8: 0.9884437, 1.0001947, 0.9884039, 1.0003278, -0.0112750, 0.0109041
9: -0.0160640, -0.0053972, -0.0161848, -0.0053611, -0.0098980, 0.0102347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0059714
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0059666
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0044325, 0.0097835, 0.0045726, 0.0100490, -0.0051490, 0.0050541
1: 0.0019627, 0.0027357, 0.0019829, 0.0027741, -0.0007439, 0.0007302
2: 0.0089508, 0.0119093, 0.0088041, 0.0118318, -0.0027943, 0.0028468
3: -0.0054231, -0.0023633, -0.0055749, -0.0024435, -0.0028900, 0.0029443
4: -0.0014785, 0.0018339, -0.0013917, 0.0019982, -0.0031873, 0.0031286
5: 0.0023779, 0.0055125, 0.0022224, 0.0054304, -0.0029607, 0.0030163
6: -0.0128657, -0.0004283, -0.0134825, -0.0007541, -0.0117470, 0.0119678
7: -0.0019734, 0.0149652, -0.0015297, 0.0158054, -0.0162991, 0.0159985
8: 0.9878238, 0.9997557, 0.9881364, 1.0003475, -0.0114814, 0.0112696
9: -0.0156655, -0.0048345, -0.0162027, -0.0051182, -0.0102298, 0.0104221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060483
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060491
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0045959, 0.0099903, 0.0044305, 0.0098905, -0.0051623, 0.0049726
1: 0.0019863, 0.0027656, 0.0019624, 0.0027512, -0.0007458, 0.0007184
2: 0.0088365, 0.0118189, 0.0088916, 0.0119103, -0.0027492, 0.0028541
3: -0.0055414, -0.0024568, -0.0054843, -0.0023622, -0.0028434, 0.0029519
4: -0.0013773, 0.0019619, -0.0014797, 0.0019001, -0.0031956, 0.0030781
5: 0.0022567, 0.0054168, 0.0023152, 0.0055137, -0.0029129, 0.0030241
6: -0.0133462, -0.0008082, -0.0131143, -0.0004238, -0.0115577, 0.0119986
7: -0.0014560, 0.0156197, -0.0019796, 0.0153038, -0.0163411, 0.0157405
8: 0.9881883, 1.0002167, 0.9878193, 0.9999942, -0.0115110, 0.0110880
9: -0.0160840, -0.0051654, -0.0158820, -0.0048305, -0.0100649, 0.0104489

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0061332
time: 0.87 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0061319
time: 0.88 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043065, 0.0097927, 0.0042912, 0.0098990, -0.0052337, 0.0051329
1: 0.0019445, 0.0027371, 0.0019423, 0.0027524, -0.0007561, 0.0007416
2: 0.0089458, 0.0119789, 0.0088870, 0.0119874, -0.0028378, 0.0028936
3: -0.0054283, -0.0022913, -0.0054892, -0.0022826, -0.0029350, 0.0029927
4: -0.0015565, 0.0018395, -0.0015659, 0.0019054, -0.0032397, 0.0031773
5: 0.0023725, 0.0055863, 0.0023102, 0.0055953, -0.0030068, 0.0030659
6: -0.0128868, -0.0001354, -0.0131340, -0.0001000, -0.0119302, 0.0121645
7: -0.0023723, 0.0149940, -0.0024205, 0.0153307, -0.0165670, 0.0162479
8: 0.9875427, 0.9997759, 0.9875089, 1.0000131, -0.0116702, 0.0114454
9: -0.0156839, -0.0045794, -0.0158992, -0.0045486, -0.0103893, 0.0105934

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062646
time: 0.95 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063272
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0047496, 0.0099760, 0.0050712, 0.0103705, -0.0051868, 0.0042551
1: 0.0020085, 0.0027635, 0.0020549, 0.0028205, -0.0007493, 0.0006147
2: 0.0088444, 0.0117339, 0.0086263, 0.0115561, -0.0023525, 0.0028676
3: -0.0055332, -0.0025447, -0.0057588, -0.0027286, -0.0024331, 0.0029658
4: -0.0012822, 0.0019530, -0.0010831, 0.0021972, -0.0032107, 0.0026340
5: 0.0022651, 0.0053268, 0.0020340, 0.0051383, -0.0024926, 0.0030384
6: -0.0133130, -0.0011653, -0.0142299, -0.0019130, -0.0098899, 0.0120555
7: -0.0009696, 0.0155744, 0.0000486, 0.0168232, -0.0164185, 0.0134692
8: 0.9885308, 1.0001848, 0.9892481, 1.0010644, -0.0115656, 0.0094880
9: -0.0160550, -0.0054763, -0.0168535, -0.0061274, -0.0086126, 0.0104985

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057749, upper bound: 0.0059059
time: 0.88 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058055, upper bound: 0.0059058
time: 1.05 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0044671, 0.0097783, 0.0049515, 0.0103799, -0.0052488, 0.0044294
1: 0.0019677, 0.0027350, 0.0020377, 0.0028219, -0.0007583, 0.0006399
2: 0.0089537, 0.0118901, 0.0086211, 0.0116223, -0.0024489, 0.0029019
3: -0.0054201, -0.0023832, -0.0057641, -0.0026601, -0.0025327, 0.0030013
4: -0.0014570, 0.0018306, -0.0011572, 0.0022031, -0.0032491, 0.0027418
5: 0.0023809, 0.0054922, 0.0020285, 0.0052085, -0.0025947, 0.0030747
6: -0.0128534, -0.0005089, -0.0142518, -0.0016347, -0.0102951, 0.0121995
7: -0.0018637, 0.0149485, -0.0003304, 0.0168530, -0.0166147, 0.0140210
8: 0.9879010, 0.9997439, 0.9889811, 1.0010855, -0.0117038, 0.0098767
9: -0.0156548, -0.0049047, -0.0168726, -0.0058851, -0.0089654, 0.0106239

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0060632
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060632
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046351, 0.0099859, 0.0048055, 0.0102378, -0.0053434, 0.0043347
1: 0.0019919, 0.0027650, 0.0020166, 0.0028014, -0.0007720, 0.0006262
2: 0.0088389, 0.0117972, 0.0086996, 0.0117030, -0.0023965, 0.0029542
3: -0.0055389, -0.0024792, -0.0056829, -0.0025766, -0.0024786, 0.0030554
4: -0.0013531, 0.0019592, -0.0012476, 0.0021151, -0.0033077, 0.0026832
5: 0.0022593, 0.0053938, 0.0021117, 0.0052940, -0.0025392, 0.0031302
6: -0.0133360, -0.0008993, -0.0139216, -0.0012953, -0.0100750, 0.0124196
7: -0.0013320, 0.0156058, -0.0007927, 0.0164032, -0.0169144, 0.0137213
8: 0.9882756, 1.0002069, 0.9886555, 1.0007687, -0.0119149, 0.0096655
9: -0.0160751, -0.0052446, -0.0165850, -0.0055895, -0.0087737, 0.0108155

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0061533
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0061781
time: 1.08 seconds

## BFS IS instance: IS_A1_A2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043372, 0.0097875, 0.0046723, 0.0102470, -0.0053750, 0.0045029
1: 0.0019489, 0.0027363, 0.0019973, 0.0028027, -0.0007765, 0.0006505
2: 0.0089486, 0.0119619, 0.0086946, 0.0117767, -0.0024895, 0.0029717
3: -0.0054254, -0.0023089, -0.0056881, -0.0025004, -0.0025748, 0.0030735
4: -0.0015375, 0.0018363, -0.0013301, 0.0021208, -0.0033272, 0.0027874
5: 0.0023756, 0.0055683, 0.0021064, 0.0053720, -0.0026378, 0.0031487
6: -0.0128748, -0.0002069, -0.0139429, -0.0009856, -0.0104660, 0.0124930
7: -0.0022750, 0.0149777, -0.0012144, 0.0164323, -0.0170144, 0.0142537
8: 0.9876114, 0.9997644, 0.9883584, 1.0007892, -0.0119853, 0.0100406
9: -0.0156735, -0.0046417, -0.0166036, -0.0053198, -0.0091142, 0.0108795

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.72 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063210
time: 0.90 seconds

## Relational analysis of IS_A1_A2_B2_B1_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063898
time: 1.00 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0044934, 0.0097753, 0.0048594, 0.0105938, -0.0058395, 0.0046497
1: 0.0019715, 0.0027345, 0.0020243, 0.0028528, -0.0008436, 0.0006717
2: 0.0089554, 0.0118755, 0.0085028, 0.0116732, -0.0025707, 0.0032285
3: -0.0054184, -0.0023982, -0.0058864, -0.0026074, -0.0026588, 0.0033391
4: -0.0014408, 0.0018288, -0.0012142, 0.0023354, -0.0036147, 0.0028783
5: 0.0023827, 0.0054768, 0.0019032, 0.0052624, -0.0027238, 0.0034208
6: -0.0128464, -0.0005700, -0.0147488, -0.0014205, -0.0108072, 0.0135726
7: -0.0017804, 0.0149390, -0.0006221, 0.0175299, -0.0184847, 0.0147185
8: 0.9879597, 0.9997372, 0.9887756, 1.0015624, -0.0130210, 0.0103680
9: -0.0156487, -0.0049579, -0.0173054, -0.0056986, -0.0094114, 0.0118196

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060330, upper bound: 0.0060397
time: 1.07 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B1_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060492, upper bound: 0.0060295
time: 0.94 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0043637, 0.0097845, 0.0045922, 0.0103816, -0.0058895, 0.0046816
1: 0.0019527, 0.0027359, 0.0019857, 0.0028221, -0.0008509, 0.0006764
2: 0.0089503, 0.0119473, 0.0086202, 0.0118210, -0.0025883, 0.0032561
3: -0.0054236, -0.0023240, -0.0057651, -0.0024546, -0.0026770, 0.0033676
4: -0.0015211, 0.0018345, -0.0013797, 0.0022041, -0.0036457, 0.0028980
5: 0.0023773, 0.0055528, 0.0020275, 0.0054190, -0.0027425, 0.0034500
6: -0.0128678, -0.0002684, -0.0142557, -0.0007994, -0.0108814, 0.0136887
7: -0.0021912, 0.0149681, -0.0014679, 0.0168583, -0.0186428, 0.0148195
8: 0.9876703, 0.9997576, 0.9881798, 1.0010892, -0.0131324, 0.0104392
9: -0.0156673, -0.0046952, -0.0168760, -0.0051577, -0.0094760, 0.0119207

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0063404
time: 1.11 seconds

## Relational analysis of IS_A1_A2_B2_B2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0063404
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0046351, 0.0099859, 0.0043334, 0.0102418, -0.0054963, 0.0050385
1: 0.0019919, 0.0027650, 0.0019484, 0.0028019, -0.0007941, 0.0007279
2: 0.0088389, 0.0117972, 0.0086974, 0.0119640, -0.0027857, 0.0030388
3: -0.0055389, -0.0024792, -0.0056852, -0.0023067, -0.0028811, 0.0031428
4: -0.0013531, 0.0019592, -0.0015398, 0.0021176, -0.0034023, 0.0031189
5: 0.0022593, 0.0053938, 0.0021094, 0.0055705, -0.0029516, 0.0032197
6: -0.0133360, -0.0008993, -0.0139308, -0.0001981, -0.0117110, 0.0127749
7: -0.0013320, 0.0156058, -0.0022869, 0.0164158, -0.0173983, 0.0159493
8: 0.9882756, 1.0002069, 0.9876029, 1.0007775, -0.0122557, 0.0112350
9: -0.0160751, -0.0052446, -0.0165930, -0.0046340, -0.0101984, 0.0111249

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0064136
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0064463
time: 1.04 seconds

## BFS IS instance: IS_A1_A2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0043372, 0.0097875, 0.0041918, 0.0102505, -0.0055774, 0.0051989
1: 0.0019489, 0.0027363, 0.0019279, 0.0028032, -0.0008058, 0.0007511
2: 0.0089486, 0.0119619, 0.0086926, 0.0120423, -0.0028743, 0.0030836
3: -0.0054254, -0.0023089, -0.0056902, -0.0022257, -0.0029728, 0.0031892
4: -0.0015375, 0.0018363, -0.0016275, 0.0021230, -0.0034525, 0.0032182
5: 0.0023756, 0.0055683, 0.0021043, 0.0056535, -0.0030455, 0.0032672
6: -0.0128748, -0.0002069, -0.0139511, 0.0001311, -0.0120837, 0.0129634
7: -0.0022750, 0.0149777, -0.0027352, 0.0164434, -0.0176550, 0.0164569
8: 0.9876114, 0.9997644, 0.9872871, 1.0007970, -0.0124366, 0.0115926
9: -0.0156735, -0.0046417, -0.0166107, -0.0043474, -0.0105230, 0.0112891

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0066044
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0066405
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0046001, 0.0099191, 0.0045995, 0.0098181, -0.0050351, 0.0051042
1: 0.0019869, 0.0027553, 0.0019868, 0.0027407, -0.0007274, 0.0007374
2: 0.0088758, 0.0118166, 0.0089317, 0.0118169, -0.0028220, 0.0027838
3: -0.0055007, -0.0024592, -0.0054429, -0.0024588, -0.0029186, 0.0028791
4: -0.0013747, 0.0019178, -0.0013751, 0.0018553, -0.0031168, 0.0031596
5: 0.0022984, 0.0054143, 0.0023576, 0.0054147, -0.0029900, 0.0029496
6: -0.0131808, -0.0008180, -0.0129460, -0.0008165, -0.0118636, 0.0117030
7: -0.0014426, 0.0153944, -0.0014447, 0.0150746, -0.0159385, 0.0161572
8: 0.9881977, 1.0000581, 0.9881962, 0.9998327, -0.0112274, 0.0113815
9: -0.0159399, -0.0051739, -0.0157355, -0.0051726, -0.0103314, 0.0101915

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.81 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060605, upper bound: 0.0060238
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061146, upper bound: 0.0062453
time: 0.96 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A1_A2

### Backsubstitution after applying IS history:
0: 0.0043237, 0.0099115, 0.0045995, 0.0098181, -0.0053538, 0.0051030
1: 0.0019470, 0.0027542, 0.0019868, 0.0027407, -0.0007735, 0.0007372
2: 0.0088800, 0.0119694, 0.0089317, 0.0118169, -0.0028213, 0.0029600
3: -0.0054963, -0.0023012, -0.0054429, -0.0024588, -0.0029180, 0.0030614
4: -0.0015458, 0.0019131, -0.0013751, 0.0018553, -0.0033141, 0.0031589
5: 0.0023029, 0.0055762, 0.0023576, 0.0054147, -0.0029893, 0.0031363
6: -0.0131632, -0.0001755, -0.0129460, -0.0008165, -0.0118609, 0.0124438
7: -0.0023176, 0.0153704, -0.0014447, 0.0150746, -0.0169474, 0.0161535
8: 0.9875812, 1.0000410, 0.9881962, 0.9998327, -0.0119381, 0.0113788
9: -0.0159246, -0.0046144, -0.0157355, -0.0051726, -0.0103290, 0.0108366

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060605, upper bound: 0.0060753
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_A1_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061146, upper bound: 0.0063210
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044172, 0.0099466, 0.0048611, 0.0100234, -0.0049622, 0.0048575
1: 0.0019605, 0.0027593, 0.0020246, 0.0027704, -0.0007169, 0.0007018
2: 0.0088606, 0.0119177, 0.0088182, 0.0116723, -0.0026856, 0.0027435
3: -0.0055164, -0.0023546, -0.0055603, -0.0026084, -0.0027776, 0.0028374
4: -0.0014879, 0.0019348, -0.0012132, 0.0019824, -0.0030717, 0.0030069
5: 0.0022823, 0.0055214, 0.0022373, 0.0052614, -0.0028455, 0.0029069
6: -0.0132447, -0.0003929, -0.0134232, -0.0014246, -0.0112901, 0.0115336
7: -0.0020217, 0.0154814, -0.0006166, 0.0157245, -0.0157078, 0.0153762
8: 0.9877898, 1.0001193, 0.9887796, 1.0002905, -0.0110649, 0.0108313
9: -0.0159956, -0.0048037, -0.0161510, -0.0057021, -0.0098319, 0.0100440

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061330
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061899
time: 1.10 seconds

## BFS IS instance: IS_A2_B1_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042780, 0.0099552, 0.0045718, 0.0098221, -0.0051358, 0.0049305
1: 0.0019403, 0.0027605, 0.0019828, 0.0027413, -0.0007420, 0.0007123
2: 0.0088559, 0.0119947, 0.0089295, 0.0118322, -0.0027259, 0.0028394
3: -0.0055213, -0.0022750, -0.0054452, -0.0024430, -0.0028193, 0.0029367
4: -0.0015741, 0.0019401, -0.0013922, 0.0018577, -0.0031791, 0.0030521
5: 0.0022773, 0.0056030, 0.0023553, 0.0054309, -0.0028883, 0.0030085
6: -0.0132645, -0.0000693, -0.0129552, -0.0007522, -0.0114598, 0.0119370
7: -0.0024623, 0.0155084, -0.0015323, 0.0150871, -0.0162572, 0.0156073
8: 0.9874793, 1.0001384, 0.9881345, 0.9998416, -0.0114519, 0.0109941
9: -0.0160129, -0.0045219, -0.0157435, -0.0051166, -0.0099797, 0.0103953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.23 seconds

### Candidate
type: A, layer: 1, pos: 77

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0063314
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B1_A1_A2_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0064313
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0045079, 0.0102772, 0.0046303, 0.0098127, -0.0050935, 0.0054515
1: 0.0019736, 0.0028071, 0.0019912, 0.0027400, -0.0007359, 0.0007876
2: 0.0086778, 0.0118675, 0.0089347, 0.0117999, -0.0030140, 0.0028160
3: -0.0057054, -0.0024065, -0.0054398, -0.0024764, -0.0031172, 0.0029125
4: -0.0014318, 0.0021395, -0.0013561, 0.0018520, -0.0031529, 0.0033746
5: 0.0020887, 0.0054683, 0.0023608, 0.0053966, -0.0031935, 0.0029837
6: -0.0140131, -0.0006037, -0.0129335, -0.0008880, -0.0126708, 0.0118386
7: -0.0017345, 0.0165280, -0.0013473, 0.0150576, -0.0161231, 0.0172565
8: 0.9879920, 1.0008565, 0.9882648, 0.9998208, -0.0113575, 0.0121558
9: -0.0166648, -0.0049873, -0.0157246, -0.0052348, -0.0110343, 0.0103096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A2_A1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063281, upper bound: 0.0060058
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064169, upper bound: 0.0062453
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0042515, 0.0102460, 0.0046303, 0.0098127, -0.0054139, 0.0054309
1: 0.0019365, 0.0028026, 0.0019912, 0.0027400, -0.0007821, 0.0007846
2: 0.0086951, 0.0120093, 0.0089347, 0.0117999, -0.0030026, 0.0029932
3: -0.0056876, -0.0022599, -0.0054398, -0.0024764, -0.0031054, 0.0030957
4: -0.0015905, 0.0021202, -0.0013561, 0.0018520, -0.0033513, 0.0033618
5: 0.0021070, 0.0056185, 0.0023608, 0.0053966, -0.0031814, 0.0031714
6: -0.0139406, -0.0000077, -0.0129335, -0.0008880, -0.0126229, 0.0125833
7: -0.0025463, 0.0164291, -0.0013473, 0.0150576, -0.0171373, 0.0171913
8: 0.9874203, 1.0007869, 0.9882648, 0.9998208, -0.0120719, 0.0121099
9: -0.0166016, -0.0044682, -0.0157246, -0.0052348, -0.0109926, 0.0109581

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A2_A1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063281, upper bound: 0.0060454
time: 0.88 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064169, upper bound: 0.0063210
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A1

### Backsubstitution after applying IS history:
0: 0.0042184, 0.0101414, 0.0044972, 0.0098228, -0.0051769, 0.0055855
1: 0.0019317, 0.0027874, 0.0019720, 0.0027414, -0.0007479, 0.0008069
2: 0.0087530, 0.0120276, 0.0089291, 0.0118735, -0.0030881, 0.0028622
3: -0.0056277, -0.0022409, -0.0054456, -0.0024004, -0.0031938, 0.0029602
4: -0.0016110, 0.0020554, -0.0014384, 0.0018582, -0.0032046, 0.0034575
5: 0.0021682, 0.0056379, 0.0023548, 0.0054746, -0.0032720, 0.0030326
6: -0.0136974, 0.0000693, -0.0129570, -0.0005788, -0.0129823, 0.0120325
7: -0.0026512, 0.0160979, -0.0017685, 0.0150896, -0.0163872, 0.0176807
8: 0.9873464, 1.0005536, 0.9879681, 0.9998432, -0.0115435, 0.0124547
9: -0.0163898, -0.0044011, -0.0157450, -0.0049655, -0.0113055, 0.0104784

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065161, upper bound: 0.0061322
time: 1.08 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065661, upper bound: 0.0063314
time: 1.16 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2_A2

### Backsubstitution after applying IS history:
0: 0.0039777, 0.0100976, 0.0044972, 0.0098228, -0.0054806, 0.0055587
1: 0.0018970, 0.0027811, 0.0019720, 0.0027414, -0.0007918, 0.0008031
2: 0.0087772, 0.0121607, 0.0089291, 0.0118735, -0.0030733, 0.0030301
3: -0.0056027, -0.0021033, -0.0054456, -0.0024004, -0.0031785, 0.0031339
4: -0.0017600, 0.0020283, -0.0014384, 0.0018582, -0.0033926, 0.0034409
5: 0.0021939, 0.0057789, 0.0023548, 0.0054746, -0.0032563, 0.0032105
6: -0.0135955, 0.0006288, -0.0129570, -0.0005788, -0.0129199, 0.0127385
7: -0.0034130, 0.0159592, -0.0017685, 0.0150896, -0.0173488, 0.0175958
8: 0.9868096, 1.0004559, 0.9879681, 0.9998432, -0.0122208, 0.0123949
9: -0.0163011, -0.0039140, -0.0157450, -0.0049655, -0.0112512, 0.0110933

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065161, upper bound: 0.0061849
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065661, upper bound: 0.0064313
time: 1.08 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B1

### Backsubstitution after applying IS history:
0: 0.0046926, 0.0100401, 0.0047105, 0.0099804, -0.0048901, 0.0050565
1: 0.0020003, 0.0027728, 0.0020028, 0.0027642, -0.0007065, 0.0007305
2: 0.0088089, 0.0117654, 0.0088419, 0.0117556, -0.0027956, 0.0027036
3: -0.0055698, -0.0025121, -0.0055357, -0.0025223, -0.0028913, 0.0027962
4: -0.0013175, 0.0019927, -0.0013064, 0.0019558, -0.0030271, 0.0031300
5: 0.0022276, 0.0053601, 0.0022625, 0.0053496, -0.0029621, 0.0028646
6: -0.0134620, -0.0010330, -0.0133232, -0.0010745, -0.0117527, 0.0113660
7: -0.0011498, 0.0157774, -0.0010933, 0.0155884, -0.0154795, 0.0160061
8: 0.9884039, 1.0003278, 0.9884437, 1.0001947, -0.0109041, 0.0112750
9: -0.0161848, -0.0053611, -0.0160640, -0.0053972, -0.0102347, 0.0098980

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.73 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059714, upper bound: 0.0058734
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059667, upper bound: 0.0058734
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A1_B2

### Backsubstitution after applying IS history:
0: 0.0045726, 0.0100490, 0.0044325, 0.0097835, -0.0050541, 0.0051490
1: 0.0019829, 0.0027741, 0.0019627, 0.0027357, -0.0007302, 0.0007439
2: 0.0088041, 0.0118318, 0.0089508, 0.0119093, -0.0028468, 0.0027943
3: -0.0055749, -0.0024435, -0.0054231, -0.0023633, -0.0029443, 0.0028900
4: -0.0013917, 0.0019982, -0.0014785, 0.0018339, -0.0031286, 0.0031873
5: 0.0022224, 0.0054304, 0.0023779, 0.0055125, -0.0030163, 0.0029607
6: -0.0134825, -0.0007541, -0.0128657, -0.0004283, -0.0119678, 0.0117470
7: -0.0015297, 0.0158054, -0.0019734, 0.0149652, -0.0159985, 0.0162991
8: 0.9881364, 1.0003475, 0.9878238, 0.9997557, -0.0112696, 0.0114814
9: -0.0162027, -0.0051182, -0.0156655, -0.0048345, -0.0104221, 0.0102298

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060483, upper bound: 0.0061508
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B2_A1_A1_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060491, upper bound: 0.0061508
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B1

### Backsubstitution after applying IS history:
0: 0.0044305, 0.0098905, 0.0045959, 0.0099903, -0.0049726, 0.0051623
1: 0.0019624, 0.0027512, 0.0019863, 0.0027656, -0.0007184, 0.0007458
2: 0.0088916, 0.0119103, 0.0088365, 0.0118189, -0.0028541, 0.0027492
3: -0.0054843, -0.0023622, -0.0055414, -0.0024568, -0.0029519, 0.0028434
4: -0.0014797, 0.0019001, -0.0013773, 0.0019619, -0.0030781, 0.0031956
5: 0.0023152, 0.0055137, 0.0022567, 0.0054168, -0.0030241, 0.0029129
6: -0.0131143, -0.0004238, -0.0133462, -0.0008082, -0.0119986, 0.0115577
7: -0.0019796, 0.0153038, -0.0014560, 0.0156197, -0.0157405, 0.0163411
8: 0.9878193, 0.9999942, 0.9881883, 1.0002167, -0.0110880, 0.0115110
9: -0.0158820, -0.0048305, -0.0160840, -0.0051654, -0.0104489, 0.0100649

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.76 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061332, upper bound: 0.0060047
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061319, upper bound: 0.0060047
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B2_A1_A2_B2

### Backsubstitution after applying IS history:
0: 0.0042912, 0.0098990, 0.0043065, 0.0097927, -0.0051329, 0.0052337
1: 0.0019423, 0.0027524, 0.0019445, 0.0027371, -0.0007416, 0.0007561
2: 0.0088870, 0.0119874, 0.0089458, 0.0119789, -0.0028936, 0.0028378
3: -0.0054892, -0.0022826, -0.0054283, -0.0022913, -0.0029927, 0.0029350
4: -0.0015659, 0.0019054, -0.0015565, 0.0018395, -0.0031773, 0.0032397
5: 0.0023102, 0.0055953, 0.0023725, 0.0055863, -0.0030659, 0.0030068
6: -0.0131340, -0.0001000, -0.0128868, -0.0001354, -0.0121645, 0.0119302
7: -0.0024205, 0.0153307, -0.0023723, 0.0149940, -0.0162479, 0.0165670
8: 0.9875089, 1.0000131, 0.9875427, 0.9997759, -0.0114454, 0.0116702
9: -0.0158992, -0.0045486, -0.0156839, -0.0045794, -0.0105934, 0.0103893

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062646, upper bound: 0.0062404
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B2_A1_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062646, upper bound: 0.0063898
time: 1.11 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_A1

### Backsubstitution after applying IS history:
0: 0.0048594, 0.0105938, 0.0044934, 0.0097753, -0.0046497, 0.0058395
1: 0.0020243, 0.0028528, 0.0019715, 0.0027345, -0.0006717, 0.0008436
2: 0.0085028, 0.0116732, 0.0089554, 0.0118755, -0.0032285, 0.0025707
3: -0.0058864, -0.0026074, -0.0054184, -0.0023982, -0.0033391, 0.0026588
4: -0.0012142, 0.0023354, -0.0014408, 0.0018288, -0.0028783, 0.0036147
5: 0.0019032, 0.0052624, 0.0023827, 0.0054768, -0.0034208, 0.0027238
6: -0.0147488, -0.0014205, -0.0128464, -0.0005700, -0.0135726, 0.0108072
7: -0.0006221, 0.0175299, -0.0017804, 0.0149390, -0.0147185, 0.0184847
8: 0.9887756, 1.0015624, 0.9879597, 0.9997372, -0.0103680, 0.0130210
9: -0.0173054, -0.0056986, -0.0156487, -0.0049579, -0.0118196, 0.0094114

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A2_A1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060397, upper bound: 0.0060330
time: 0.83 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A1_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060295, upper bound: 0.0060492
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A1_A2

### Backsubstitution after applying IS history:
0: 0.0045922, 0.0103816, 0.0043637, 0.0097845, -0.0046816, 0.0058895
1: 0.0019857, 0.0028221, 0.0019527, 0.0027359, -0.0006764, 0.0008509
2: 0.0086202, 0.0118210, 0.0089503, 0.0119473, -0.0032561, 0.0025883
3: -0.0057651, -0.0024546, -0.0054236, -0.0023240, -0.0033676, 0.0026770
4: -0.0013797, 0.0022041, -0.0015211, 0.0018345, -0.0028980, 0.0036457
5: 0.0020275, 0.0054190, 0.0023773, 0.0055528, -0.0034500, 0.0027425
6: -0.0142557, -0.0007994, -0.0128678, -0.0002684, -0.0136887, 0.0108814
7: -0.0014679, 0.0168583, -0.0021912, 0.0149681, -0.0148195, 0.0186428
8: 0.9881798, 1.0010892, 0.9876703, 0.9997576, -0.0104392, 0.0131324
9: -0.0168760, -0.0051577, -0.0156673, -0.0046952, -0.0119207, 0.0094760

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A2_A1_A2_A1

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063404, upper bound: 0.0061237
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B2_A2_A1_A2_A2

### Relational analysis result of IS_A2_B1_B2_A2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063404, upper bound: 0.0061508
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0043334, 0.0102418, 0.0046351, 0.0099859, -0.0050385, 0.0054963
1: 0.0019484, 0.0028019, 0.0019919, 0.0027650, -0.0007279, 0.0007941
2: 0.0086974, 0.0119640, 0.0088389, 0.0117972, -0.0030388, 0.0027857
3: -0.0056852, -0.0023067, -0.0055389, -0.0024792, -0.0031428, 0.0028811
4: -0.0015398, 0.0021176, -0.0013531, 0.0019592, -0.0031189, 0.0034023
5: 0.0021094, 0.0055705, 0.0022593, 0.0053938, -0.0032197, 0.0029516
6: -0.0139308, -0.0001981, -0.0133360, -0.0008993, -0.0127749, 0.0117110
7: -0.0022869, 0.0164158, -0.0013320, 0.0156058, -0.0159493, 0.0173983
8: 0.9876029, 1.0007775, 0.9882756, 1.0002069, -0.0112350, 0.0122557
9: -0.0165930, -0.0046340, -0.0160751, -0.0052446, -0.0111249, 0.0101984

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064136, upper bound: 0.0059794
time: 1.00 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064463, upper bound: 0.0059793
time: 1.07 seconds

## BFS IS instance: IS_A2_B1_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0041918, 0.0102505, 0.0043372, 0.0097875, -0.0051989, 0.0055774
1: 0.0019279, 0.0028032, 0.0019489, 0.0027363, -0.0007511, 0.0008058
2: 0.0086926, 0.0120423, 0.0089486, 0.0119619, -0.0030836, 0.0028743
3: -0.0056902, -0.0022257, -0.0054254, -0.0023089, -0.0031892, 0.0029728
4: -0.0016275, 0.0021230, -0.0015375, 0.0018363, -0.0032182, 0.0034525
5: 0.0021043, 0.0056535, 0.0023756, 0.0055683, -0.0032672, 0.0030455
6: -0.0139511, 0.0001311, -0.0128748, -0.0002069, -0.0129634, 0.0120837
7: -0.0027352, 0.0164434, -0.0022750, 0.0149777, -0.0164569, 0.0176550
8: 0.9872871, 1.0007970, 0.9876114, 0.9997644, -0.0115926, 0.0124366
9: -0.0166107, -0.0043474, -0.0156735, -0.0046417, -0.0112891, 0.0105230

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.83 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 197
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 77
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 197

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B1

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066044, upper bound: 0.0062404
time: 1.09 seconds

## Relational analysis of IS_A2_B1_B2_A2_A2_B2_B2

### Relational analysis result of IS_A2_B1_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0066044, upper bound: 0.0063898
time: 1.12 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B1

### Backsubstitution after applying IS history:
0: 0.0041293, 0.0097903, 0.0046001, 0.0099191, -0.0050702, 0.0045247
1: 0.0019189, 0.0027367, 0.0019869, 0.0027553, -0.0007325, 0.0006537
2: 0.0089471, 0.0120769, 0.0088758, 0.0118166, -0.0025016, 0.0028032
3: -0.0054270, -0.0021900, -0.0055007, -0.0024592, -0.0025872, 0.0028992
4: -0.0016662, 0.0018381, -0.0013747, 0.0019178, -0.0031385, 0.0028008
5: 0.0023739, 0.0056901, 0.0022984, 0.0054143, -0.0026505, 0.0029701
6: -0.0128814, 0.0002763, -0.0131808, -0.0008180, -0.0105166, 0.0117845
7: -0.0029331, 0.0149866, -0.0014426, 0.0153944, -0.0160494, 0.0143226
8: 0.9871478, 0.9997708, 0.9881977, 1.0000581, -0.0113055, 0.0100892
9: -0.0156792, -0.0042209, -0.0159399, -0.0051739, -0.0091583, 0.0102624

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0060257
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0060536
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B1_B2

### Backsubstitution after applying IS history:
0: 0.0041293, 0.0097903, 0.0043237, 0.0099115, -0.0050882, 0.0048719
1: 0.0019189, 0.0027367, 0.0019470, 0.0027542, -0.0007351, 0.0007039
2: 0.0089471, 0.0120769, 0.0088800, 0.0119694, -0.0026936, 0.0028131
3: -0.0054270, -0.0021900, -0.0054963, -0.0023012, -0.0027858, 0.0029095
4: -0.0016662, 0.0018381, -0.0015458, 0.0019131, -0.0031497, 0.0030158
5: 0.0023739, 0.0056901, 0.0023029, 0.0055762, -0.0028540, 0.0029807
6: -0.0128814, 0.0002763, -0.0131632, -0.0001755, -0.0113237, 0.0118264
7: -0.0029331, 0.0149866, -0.0023176, 0.0153704, -0.0161065, 0.0154219
8: 0.9871478, 0.9997708, 0.9875812, 1.0000410, -0.0113458, 0.0108635
9: -0.0156792, -0.0042209, -0.0159246, -0.0046144, -0.0098612, 0.0102989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.75 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0060257
time: 1.02 seconds

## Relational analysis of IS_A2_B2_A1_B1_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0060536
time: 1.21 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0044109, 0.0099831, 0.0044172, 0.0099466, -0.0048479, 0.0044798
1: 0.0019595, 0.0027646, 0.0019605, 0.0027593, -0.0007004, 0.0006472
2: 0.0088405, 0.0119212, 0.0088606, 0.0119177, -0.0024768, 0.0026803
3: -0.0055372, -0.0023510, -0.0055164, -0.0023546, -0.0025616, 0.0027721
4: -0.0014919, 0.0019574, -0.0014879, 0.0019348, -0.0030009, 0.0027731
5: 0.0022610, 0.0055252, 0.0022823, 0.0055214, -0.0026243, 0.0028399
6: -0.0133295, -0.0003781, -0.0132447, -0.0003929, -0.0104123, 0.0112679
7: -0.0020417, 0.0155969, -0.0020217, 0.0154814, -0.0153459, 0.0141806
8: 0.9877756, 1.0002006, 0.9877898, 1.0001193, -0.0108100, 0.0099891
9: -0.0160694, -0.0047908, -0.0159956, -0.0048037, -0.0090675, 0.0098126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0060922
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0060922
time: 1.06 seconds

## BFS IS instance: IS_A2_B2_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0040966, 0.0097935, 0.0042780, 0.0099552, -0.0049298, 0.0046447
1: 0.0019141, 0.0027372, 0.0019403, 0.0027605, -0.0007122, 0.0006710
2: 0.0089453, 0.0120950, 0.0088559, 0.0119947, -0.0025679, 0.0027255
3: -0.0054288, -0.0021713, -0.0055213, -0.0022750, -0.0026559, 0.0028189
4: -0.0016864, 0.0018400, -0.0015741, 0.0019401, -0.0030516, 0.0028751
5: 0.0023721, 0.0057093, 0.0022773, 0.0056030, -0.0027208, 0.0028879
6: -0.0128887, 0.0003524, -0.0132645, -0.0000693, -0.0107955, 0.0114582
7: -0.0030366, 0.0149966, -0.0024623, 0.0155084, -0.0156050, 0.0147026
8: 0.9870748, 0.9997778, 0.9874793, 1.0001384, -0.0109925, 0.0103568
9: -0.0156856, -0.0041547, -0.0160129, -0.0045219, -0.0094012, 0.0099783

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.70 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A2_B2_A1_B1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0061086
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0061086
time: 1.09 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: 0.0041510, 0.0097851, 0.0045079, 0.0102772, -0.0054484, 0.0045998
1: 0.0019220, 0.0027360, 0.0019736, 0.0028071, -0.0007871, 0.0006645
2: 0.0089499, 0.0120649, 0.0086778, 0.0118675, -0.0025431, 0.0030123
3: -0.0054240, -0.0022024, -0.0057054, -0.0024065, -0.0026302, 0.0031154
4: -0.0016527, 0.0018349, -0.0014318, 0.0021395, -0.0033726, 0.0028474
5: 0.0023770, 0.0056774, 0.0020887, 0.0054683, -0.0026946, 0.0031916
6: -0.0128692, 0.0002259, -0.0140131, -0.0006037, -0.0106912, 0.0126635
7: -0.0028644, 0.0149701, -0.0017345, 0.0165280, -0.0172466, 0.0145605
8: 0.9871961, 0.9997592, 0.9879920, 1.0008565, -0.0121489, 0.0102567
9: -0.0156686, -0.0042648, -0.0166648, -0.0049873, -0.0093104, 0.0110280

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061875
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0062252
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: 0.0041510, 0.0097851, 0.0042515, 0.0102460, -0.0054317, 0.0049493
1: 0.0019220, 0.0027360, 0.0019365, 0.0028026, -0.0007847, 0.0007150
2: 0.0089499, 0.0120649, 0.0086951, 0.0120093, -0.0027364, 0.0030030
3: -0.0054240, -0.0022024, -0.0056876, -0.0022599, -0.0028301, 0.0031059
4: -0.0016527, 0.0018349, -0.0015905, 0.0021202, -0.0033623, 0.0030637
5: 0.0023770, 0.0056774, 0.0021070, 0.0056185, -0.0028993, 0.0031819
6: -0.0128692, 0.0002259, -0.0139406, -0.0000077, -0.0115036, 0.0126247
7: -0.0028644, 0.0149701, -0.0025463, 0.0164291, -0.0171937, 0.0156670
8: 0.9871961, 0.9997592, 0.9874203, 1.0007869, -0.0121116, 0.0110361
9: -0.0156686, -0.0042648, -0.0166016, -0.0044682, -0.0100179, 0.0109941

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061875
time: 1.17 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0062252
time: 1.14 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: 0.0040117, 0.0097942, 0.0042184, 0.0101414, -0.0056150, 0.0046996
1: 0.0019019, 0.0027373, 0.0019317, 0.0027874, -0.0008112, 0.0006790
2: 0.0089449, 0.0121419, 0.0087530, 0.0120276, -0.0025983, 0.0031044
3: -0.0054292, -0.0021227, -0.0056277, -0.0022409, -0.0026873, 0.0032107
4: -0.0017390, 0.0018405, -0.0016110, 0.0020554, -0.0034758, 0.0029091
5: 0.0023716, 0.0057590, 0.0021682, 0.0056379, -0.0027530, 0.0032892
6: -0.0128903, 0.0005497, -0.0136974, 0.0000693, -0.0109232, 0.0130507
7: -0.0033054, 0.0149988, -0.0026512, 0.0160979, -0.0177740, 0.0148764
8: 0.9868855, 0.9997794, 0.9873464, 1.0005536, -0.0125204, 0.0104792
9: -0.0156870, -0.0039828, -0.0163898, -0.0044011, -0.0095124, 0.0113652

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0063198
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0063408
time: 1.16 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: 0.0040117, 0.0097942, 0.0039777, 0.0100976, -0.0055962, 0.0050320
1: 0.0019019, 0.0027373, 0.0018970, 0.0027811, -0.0008085, 0.0007270
2: 0.0089449, 0.0121419, 0.0087772, 0.0121607, -0.0027821, 0.0030940
3: -0.0054292, -0.0021227, -0.0056027, -0.0021033, -0.0028774, 0.0031999
4: -0.0017390, 0.0018405, -0.0017600, 0.0020283, -0.0034641, 0.0031149
5: 0.0023716, 0.0057590, 0.0021939, 0.0057789, -0.0029478, 0.0032782
6: -0.0128903, 0.0005497, -0.0135955, 0.0006288, -0.0116959, 0.0130071
7: -0.0033054, 0.0149988, -0.0034130, 0.0159592, -0.0177145, 0.0159288
8: 0.9868855, 0.9997794, 0.9868096, 1.0004559, -0.0124785, 0.0112205
9: -0.0156870, -0.0039828, -0.0163011, -0.0039140, -0.0101853, 0.0113271

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0063198
time: 1.16 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0063408
time: 1.17 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0042925, 0.0099298, 0.0046926, 0.0100401, -0.0049946, 0.0043890
1: 0.0019424, 0.0027569, 0.0020003, 0.0027728, -0.0007216, 0.0006341
2: 0.0088699, 0.0119867, 0.0088089, 0.0117654, -0.0024266, 0.0027614
3: -0.0055067, -0.0022833, -0.0055698, -0.0025121, -0.0025097, 0.0028560
4: -0.0015652, 0.0019244, -0.0013175, 0.0019927, -0.0030918, 0.0027169
5: 0.0022922, 0.0055945, 0.0022276, 0.0053601, -0.0025711, 0.0029258
6: -0.0132056, -0.0001029, -0.0134620, -0.0010330, -0.0102012, 0.0116089
7: -0.0024165, 0.0154281, -0.0011498, 0.0157774, -0.0158103, 0.0138932
8: 0.9875116, 1.0000817, 0.9884039, 1.0003278, -0.0111371, 0.0097867
9: -0.0159615, -0.0045511, -0.0161848, -0.0053611, -0.0088837, 0.0101096

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062239, upper bound: 0.0059255
time: 1.08 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062239, upper bound: 0.0059483
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0039802, 0.0097601, 0.0045726, 0.0100490, -0.0050838, 0.0045428
1: 0.0018973, 0.0027323, 0.0019829, 0.0027741, -0.0007345, 0.0006563
2: 0.0089638, 0.0121593, 0.0088041, 0.0118318, -0.0025116, 0.0028107
3: -0.0054097, -0.0021047, -0.0055749, -0.0024435, -0.0025976, 0.0029070
4: -0.0017585, 0.0018194, -0.0013917, 0.0019982, -0.0031470, 0.0028121
5: 0.0023916, 0.0057775, 0.0022224, 0.0054304, -0.0026612, 0.0029781
6: -0.0128111, 0.0006230, -0.0134825, -0.0007541, -0.0105588, 0.0118163
7: -0.0034052, 0.0148909, -0.0015297, 0.0158054, -0.0160927, 0.0143802
8: 0.9868152, 0.9997033, 0.9881364, 1.0003475, -0.0113360, 0.0101297
9: -0.0156180, -0.0039190, -0.0162027, -0.0051182, -0.0091951, 0.0102901

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 251

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065921, upper bound: 0.0060818
time: 0.91 seconds

## Relational analysis of IS_A2_B2_A2_B1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065921, upper bound: 0.0061697
time: 1.07 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0041702, 0.0099387, 0.0044305, 0.0098905, -0.0051424, 0.0044870
1: 0.0019248, 0.0027582, 0.0019624, 0.0027512, -0.0007429, 0.0006482
2: 0.0088650, 0.0120543, 0.0088916, 0.0119103, -0.0024807, 0.0028431
3: -0.0055119, -0.0022133, -0.0054843, -0.0023622, -0.0025657, 0.0029404
4: -0.0016409, 0.0019299, -0.0014797, 0.0019001, -0.0031832, 0.0027775
5: 0.0022870, 0.0056662, 0.0023152, 0.0055137, -0.0026285, 0.0030124
6: -0.0132263, 0.0001814, -0.0131143, -0.0004238, -0.0104289, 0.0119523
7: -0.0028038, 0.0154564, -0.0019796, 0.0153038, -0.0162780, 0.0142033
8: 0.9872388, 1.0001017, 0.9878193, 0.9999942, -0.0114665, 0.0100051
9: -0.0159796, -0.0043035, -0.0158820, -0.0048305, -0.0090820, 0.0104086

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061594
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061960
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038419, 0.0097686, 0.0042912, 0.0098990, -0.0052190, 0.0046341
1: 0.0018773, 0.0027336, 0.0019423, 0.0027524, -0.0007540, 0.0006695
2: 0.0089591, 0.0122358, 0.0088870, 0.0119874, -0.0025621, 0.0028855
3: -0.0054146, -0.0020256, -0.0054892, -0.0022826, -0.0026498, 0.0029843
4: -0.0018441, 0.0018246, -0.0015659, 0.0019054, -0.0032307, 0.0028686
5: 0.0023866, 0.0058585, 0.0023102, 0.0055953, -0.0027147, 0.0030573
6: -0.0128308, 0.0009445, -0.0131340, -0.0001000, -0.0107710, 0.0121304
7: -0.0038430, 0.0149178, -0.0024205, 0.0153307, -0.0165206, 0.0146691
8: 0.9865068, 0.9997223, 0.9875089, 1.0000131, -0.0116374, 0.0103332
9: -0.0156352, -0.0036390, -0.0158992, -0.0045486, -0.0093798, 0.0105637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.69 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0061812
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0062178
time: 1.30 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0043211, 0.0099255, 0.0046025, 0.0103791, -0.0053525, 0.0044621
1: 0.0019466, 0.0027562, 0.0019872, 0.0028218, -0.0007733, 0.0006446
2: 0.0088723, 0.0119708, 0.0086216, 0.0118153, -0.0024670, 0.0029593
3: -0.0055043, -0.0022997, -0.0057636, -0.0024605, -0.0025514, 0.0030606
4: -0.0015474, 0.0019218, -0.0013733, 0.0022025, -0.0033133, 0.0027621
5: 0.0022947, 0.0055777, 0.0020290, 0.0054129, -0.0026139, 0.0031355
6: -0.0131956, -0.0001695, -0.0142498, -0.0008234, -0.0103711, 0.0124407
7: -0.0023259, 0.0154145, -0.0014353, 0.0168502, -0.0169432, 0.0141245
8: 0.9875754, 1.0000722, 0.9882028, 1.0010835, -0.0119351, 0.0099496
9: -0.0159528, -0.0046091, -0.0168708, -0.0051786, -0.0090316, 0.0108339

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061716, upper bound: 0.0060913
time: 0.87 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062061, upper bound: 0.0060913
time: 0.86 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0040015, 0.0097548, 0.0044798, 0.0103881, -0.0054503, 0.0046176
1: 0.0019004, 0.0027316, 0.0019695, 0.0028231, -0.0007874, 0.0006671
2: 0.0089667, 0.0121475, 0.0086166, 0.0118831, -0.0025529, 0.0030133
3: -0.0054067, -0.0021169, -0.0057688, -0.0023904, -0.0026404, 0.0031165
4: -0.0017453, 0.0018161, -0.0014492, 0.0022081, -0.0033738, 0.0028583
5: 0.0023947, 0.0057650, 0.0020237, 0.0054848, -0.0027050, 0.0031928
6: -0.0127987, 0.0005734, -0.0142707, -0.0005384, -0.0107325, 0.0126680
7: -0.0033376, 0.0148741, -0.0018235, 0.0168788, -0.0172528, 0.0146167
8: 0.9868628, 0.9996915, 0.9879294, 1.0011036, -0.0121532, 0.0102963
9: -0.0156072, -0.0039622, -0.0168891, -0.0049304, -0.0093463, 0.0110319

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A2_B2_A2_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064651, upper bound: 0.0061992
time: 0.85 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0064984, upper bound: 0.0061992
time: 1.13 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0041977, 0.0099344, 0.0043334, 0.0102418, -0.0055017, 0.0045676
1: 0.0019287, 0.0027575, 0.0019484, 0.0028019, -0.0007948, 0.0006599
2: 0.0088674, 0.0120391, 0.0086974, 0.0119640, -0.0025253, 0.0030418
3: -0.0055094, -0.0022291, -0.0056852, -0.0023067, -0.0026118, 0.0031459
4: -0.0016238, 0.0019273, -0.0015398, 0.0021176, -0.0034057, 0.0028274
5: 0.0022895, 0.0056500, 0.0021094, 0.0055705, -0.0026757, 0.0032229
6: -0.0132164, 0.0001174, -0.0139308, -0.0001981, -0.0106163, 0.0127875
7: -0.0027166, 0.0154429, -0.0022869, 0.0164158, -0.0174155, 0.0144584
8: 0.9873003, 1.0000921, 0.9876029, 1.0007775, -0.0122678, 0.0101848
9: -0.0159709, -0.0043593, -0.0165930, -0.0046340, -0.0092451, 0.0111359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.92 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063748, upper bound: 0.0063659
time: 1.15 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0063748, upper bound: 0.0063769
time: 1.11 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0038628, 0.0097633, 0.0041918, 0.0102505, -0.0055723, 0.0047167
1: 0.0018804, 0.0027328, 0.0019279, 0.0028032, -0.0008050, 0.0006814
2: 0.0089620, 0.0122242, 0.0086926, 0.0120423, -0.0026077, 0.0030808
3: -0.0054116, -0.0020376, -0.0056902, -0.0022257, -0.0026971, 0.0031863
4: -0.0018311, 0.0018214, -0.0016275, 0.0021230, -0.0034494, 0.0029197
5: 0.0023897, 0.0058462, 0.0021043, 0.0056535, -0.0027630, 0.0032643
6: -0.0128186, 0.0008958, -0.0139511, 0.0001311, -0.0109629, 0.0129517
7: -0.0037767, 0.0149012, -0.0027352, 0.0164434, -0.0176390, 0.0149305
8: 0.9865535, 0.9997106, 0.9872871, 1.0007970, -0.0124253, 0.0105174
9: -0.0156245, -0.0036814, -0.0166107, -0.0043474, -0.0095470, 0.0112789

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=26, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0063952
time: 1.06 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0064085
time: 1.22 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 4.12 seconds
IS_A1_A1_B1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0059157
IS_A1_A1_B1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0059963
IS_A1_A1_B1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0059157
IS_A1_A1_B1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0059963
IS_A1_A1_B1_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0060712
IS_A1_A1_B1_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0061202
IS_A1_A1_B1_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0060712
IS_A1_A1_B1_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0061202
IS_A1_A1_B1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0060605
IS_A1_A1_B1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061146
IS_A1_A1_B1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060238, upper bound: 0.0060605
IS_A1_A1_B1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061146
IS_A1_A1_B1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0061821
IS_A1_A1_B1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061330, upper bound: 0.0061821
IS_A1_A1_B1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0062145
IS_A1_A1_B1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0062145
IS_A1_A1_B2_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0060467
IS_A1_A1_B2_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061477
IS_A1_A1_B2_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0060467
IS_A1_A1_B2_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0061477
IS_A1_A1_B2_B1_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0062584
IS_A1_A1_B2_B1_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0063211
IS_A1_A1_B2_B1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0062584
IS_A1_A1_B2_B1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0063211
IS_A1_A1_B2_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0063281
IS_A1_A1_B2_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0064169
IS_A1_A1_B2_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060058, upper bound: 0.0063281
IS_A1_A1_B2_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062453, upper bound: 0.0064169
IS_A1_A1_B2_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0065161
IS_A1_A1_B2_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0065661
IS_A1_A1_B2_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061322, upper bound: 0.0065161
IS_A1_A1_B2_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063314, upper bound: 0.0065661
IS_A1_A2_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0057801
IS_A1_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0057930
IS_A1_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0059212
IS_A1_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0059212
IS_A1_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059691
IS_A1_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059964
IS_A1_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0061303
IS_A1_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062129
IS_A1_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0059714
IS_A1_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0059666
IS_A1_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060483
IS_A1_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060491
IS_A1_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0061332
IS_A1_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0061319
IS_A1_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062646
IS_A1_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063272
IS_A1_A2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0057749, upper bound: 0.0059059
IS_A1_A2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0058055, upper bound: 0.0059058
IS_A1_A2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0060632
IS_A1_A2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060632
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0061533
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0061781
IS_A1_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063210
IS_A1_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063898
IS_A1_A2_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060330, upper bound: 0.0060397
IS_A1_A2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060492, upper bound: 0.0060295
IS_A1_A2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0063404
IS_A1_A2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0063404
IS_A1_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0064136
IS_A1_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0064463
IS_A1_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0066044
IS_A1_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0066405
IS_A2_B1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060605, upper bound: 0.0060238
IS_A2_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061146, upper bound: 0.0062453
IS_A2_B1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060605, upper bound: 0.0060753
IS_A2_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061146, upper bound: 0.0063210
IS_A2_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061330
IS_A2_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061899
IS_A2_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0063314
IS_A2_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0064313
IS_A2_B1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063281, upper bound: 0.0060058
IS_A2_B1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064169, upper bound: 0.0062453
IS_A2_B1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063281, upper bound: 0.0060454
IS_A2_B1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064169, upper bound: 0.0063210
IS_A2_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065161, upper bound: 0.0061322
IS_A2_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065661, upper bound: 0.0063314
IS_A2_B1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065161, upper bound: 0.0061849
IS_A2_B1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065661, upper bound: 0.0064313
IS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0059714, upper bound: 0.0058734
IS_A2_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0059667, upper bound: 0.0058734
IS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060483, upper bound: 0.0061508
IS_A2_B1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060491, upper bound: 0.0061508
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061332, upper bound: 0.0060047
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061319, upper bound: 0.0060047
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062646, upper bound: 0.0062404
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062646, upper bound: 0.0063898
IS_A2_B1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060397, upper bound: 0.0060330
IS_A2_B1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0060295, upper bound: 0.0060492
IS_A2_B1_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063404, upper bound: 0.0061237
IS_A2_B1_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063404, upper bound: 0.0061508
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064136, upper bound: 0.0059794
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064463, upper bound: 0.0059793
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0066044, upper bound: 0.0062404
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0066044, upper bound: 0.0063898
IS_A2_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0060257
IS_A2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0060536
IS_A2_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0060257
IS_A2_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0060536
IS_A2_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0060922
IS_A2_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0060922
IS_A2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0061086
IS_A2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0061086
IS_A2_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061875
IS_A2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0062252
IS_A2_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061875
IS_A2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0062252
IS_A2_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0063198
IS_A2_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0063408
IS_A2_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0063198
IS_A2_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0063408
IS_A2_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062239, upper bound: 0.0059255
IS_A2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062239, upper bound: 0.0059483
IS_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065921, upper bound: 0.0060818
IS_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065921, upper bound: 0.0061697
IS_A2_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061594
IS_A2_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061960
IS_A2_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0061812
IS_A2_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0062178
IS_A2_B2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0061716, upper bound: 0.0060913
IS_A2_B2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0062061, upper bound: 0.0060913
IS_A2_B2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064651, upper bound: 0.0061992
IS_A2_B2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0064984, upper bound: 0.0061992
IS_A2_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063748, upper bound: 0.0063659
IS_A2_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0063748, upper bound: 0.0063769
IS_A2_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0063952
IS_A2_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0064085

## BFS IS instance: IS_A1_A1_B1_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0049782, 0.0100124, 0.0051364, 0.0099612, -0.0043154, 0.0039660
1: 0.0020415, 0.0027688, 0.0020644, 0.0027614, -0.0006234, 0.0005730
2: 0.0088243, 0.0116075, 0.0088526, 0.0115201, -0.0021927, 0.0023859
3: -0.0055540, -0.0026754, -0.0055247, -0.0027659, -0.0022678, 0.0024676
4: -0.0011407, 0.0019756, -0.0010428, 0.0019438, -0.0026713, 0.0024550
5: 0.0022438, 0.0051928, 0.0022738, 0.0051001, -0.0023233, 0.0025279
6: -0.0133976, -0.0016968, -0.0132785, -0.0020645, -0.0092182, 0.0100302
7: -0.0002458, 0.0156897, 0.0002549, 0.0155274, -0.0136602, 0.0125543
8: 0.9890407, 1.0002660, 0.9893934, 1.0001516, -0.0096225, 0.0088435
9: -0.0161288, -0.0059392, -0.0160250, -0.0062593, -0.0080276, 0.0087347

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0057764
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0058114
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0047057, 0.0098120, 0.0050264, 0.0099710, -0.0043557, 0.0041143
1: 0.0020021, 0.0027398, 0.0020485, 0.0027628, -0.0006293, 0.0005944
2: 0.0089351, 0.0117582, 0.0088471, 0.0115809, -0.0022747, 0.0024081
3: -0.0054394, -0.0025196, -0.0055303, -0.0027030, -0.0023526, 0.0024906
4: -0.0013094, 0.0018515, -0.0011108, 0.0019499, -0.0026962, 0.0025468
5: 0.0023612, 0.0053524, 0.0022680, 0.0051646, -0.0024101, 0.0025515
6: -0.0129317, -0.0010634, -0.0133014, -0.0018088, -0.0095627, 0.0101238
7: -0.0011084, 0.0150552, -0.0000933, 0.0155586, -0.0137877, 0.0130236
8: 0.9884331, 0.9998190, 0.9891482, 1.0001737, -0.0097123, 0.0091741
9: -0.0157230, -0.0053876, -0.0160450, -0.0060367, -0.0083276, 0.0088162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0059117
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0059117
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0049782, 0.0100124, 0.0048489, 0.0099477, -0.0043504, 0.0043689
1: 0.0020415, 0.0027688, 0.0020228, 0.0027595, -0.0006285, 0.0006312
2: 0.0088243, 0.0116075, 0.0088600, 0.0116790, -0.0024154, 0.0024052
3: -0.0055540, -0.0026754, -0.0055170, -0.0026014, -0.0024982, 0.0024876
4: -0.0011407, 0.0019756, -0.0012207, 0.0019355, -0.0026929, 0.0027044
5: 0.0022438, 0.0051928, 0.0022817, 0.0052686, -0.0025593, 0.0025484
6: -0.0133976, -0.0016968, -0.0132472, -0.0013961, -0.0101544, 0.0101114
7: -0.0002458, 0.0156897, -0.0006553, 0.0154848, -0.0137709, 0.0138295
8: 0.9890407, 1.0002660, 0.9887522, 1.0001216, -0.0097005, 0.0097418
9: -0.0161288, -0.0059392, -0.0159977, -0.0056773, -0.0088429, 0.0088055

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0056924
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0057163
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0047057, 0.0098120, 0.0047361, 0.0099572, -0.0044271, 0.0045013
1: 0.0020021, 0.0027398, 0.0020065, 0.0027608, -0.0006396, 0.0006503
2: 0.0089351, 0.0117582, 0.0088548, 0.0117414, -0.0024887, 0.0024476
3: -0.0054394, -0.0025196, -0.0055224, -0.0025370, -0.0025739, 0.0025315
4: -0.0013094, 0.0018515, -0.0012906, 0.0019414, -0.0027405, 0.0027864
5: 0.0023612, 0.0053524, 0.0022762, 0.0053346, -0.0026369, 0.0025934
6: -0.0129317, -0.0010634, -0.0132692, -0.0011340, -0.0104623, 0.0102898
7: -0.0011084, 0.0150552, -0.0010123, 0.0155147, -0.0140139, 0.0142488
8: 0.9884331, 0.9998190, 0.9885008, 1.0001428, -0.0098717, 0.0100372
9: -0.0157230, -0.0053876, -0.0160169, -0.0054491, -0.0091111, 0.0089608

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=21, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0058407
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B1_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0058407
time: 0.85 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0048611, 0.0100234, 0.0049023, 0.0097985, -0.0044916, 0.0042266
1: 0.0020246, 0.0027704, 0.0020305, 0.0027379, -0.0006489, 0.0006106
2: 0.0088182, 0.0116723, 0.0089425, 0.0116495, -0.0023368, 0.0024833
3: -0.0055603, -0.0026084, -0.0054317, -0.0026320, -0.0024168, 0.0025683
4: -0.0012132, 0.0019824, -0.0011877, 0.0018432, -0.0027804, 0.0026163
5: 0.0022373, 0.0052614, 0.0023691, 0.0052373, -0.0024759, 0.0026312
6: -0.0134232, -0.0014246, -0.0129004, -0.0015203, -0.0098238, 0.0104398
7: -0.0006166, 0.0157245, -0.0004862, 0.0150126, -0.0142180, 0.0133792
8: 0.9887796, 1.0002905, 0.9888715, 0.9997891, -0.0100155, 0.0094246
9: -0.0161510, -0.0057021, -0.0156958, -0.0057855, -0.0085550, 0.0090914

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060318, upper bound: 0.0059087
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060318, upper bound: 0.0059515
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0045718, 0.0098221, 0.0047737, 0.0098080, -0.0045275, 0.0044079
1: 0.0019828, 0.0027413, 0.0020120, 0.0027393, -0.0006541, 0.0006368
2: 0.0089295, 0.0118322, 0.0089373, 0.0117206, -0.0024370, 0.0025031
3: -0.0054452, -0.0024430, -0.0054371, -0.0025585, -0.0025205, 0.0025888
4: -0.0013922, 0.0018577, -0.0012673, 0.0018490, -0.0028026, 0.0027286
5: 0.0023553, 0.0054309, 0.0023635, 0.0053126, -0.0025821, 0.0026522
6: -0.0129552, -0.0007522, -0.0129225, -0.0012214, -0.0102452, 0.0105231
7: -0.0015323, 0.0150871, -0.0008933, 0.0150426, -0.0143315, 0.0139530
8: 0.9881345, 0.9998416, 0.9885846, 0.9998102, -0.0100954, 0.0098288
9: -0.0157435, -0.0051166, -0.0157150, -0.0055252, -0.0089220, 0.0091640

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0060993
time: 0.94 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0061279
time: 1.16 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0048611, 0.0100234, 0.0046306, 0.0097696, -0.0045038, 0.0045964
1: 0.0020246, 0.0027704, 0.0019913, 0.0027337, -0.0006507, 0.0006640
2: 0.0088182, 0.0116723, 0.0089585, 0.0117997, -0.0025412, 0.0024900
3: -0.0055603, -0.0026084, -0.0054151, -0.0024766, -0.0026283, 0.0025753
4: -0.0012132, 0.0019824, -0.0013559, 0.0018253, -0.0027879, 0.0028452
5: 0.0022373, 0.0052614, 0.0023860, 0.0053964, -0.0026926, 0.0026383
6: -0.0134232, -0.0014246, -0.0128332, -0.0008888, -0.0106833, 0.0104680
7: -0.0006166, 0.0157245, -0.0013462, 0.0149210, -0.0142565, 0.0145497
8: 0.9887796, 1.0002905, 0.9882656, 0.9997246, -0.0100426, 0.0102491
9: -0.0161510, -0.0057021, -0.0156372, -0.0052356, -0.0093035, 0.0091160

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0058699
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059062
time: 0.94 seconds

## BFS IS instance: IS_A1_A1_B1_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0045718, 0.0098221, 0.0045036, 0.0097787, -0.0045595, 0.0047629
1: 0.0019828, 0.0027413, 0.0019729, 0.0027350, -0.0006587, 0.0006881
2: 0.0089295, 0.0118322, 0.0089535, 0.0118700, -0.0026333, 0.0025208
3: -0.0054452, -0.0024430, -0.0054204, -0.0024040, -0.0027235, 0.0026071
4: -0.0013922, 0.0018577, -0.0014345, 0.0018309, -0.0028224, 0.0029483
5: 0.0023553, 0.0054309, 0.0023807, 0.0054709, -0.0027901, 0.0026709
6: -0.0129552, -0.0007522, -0.0128545, -0.0005935, -0.0110704, 0.0105974
7: -0.0015323, 0.0150871, -0.0017484, 0.0149499, -0.0144328, 0.0150769
8: 0.9881345, 0.9998416, 0.9879823, 0.9997450, -0.0101667, 0.0106205
9: -0.0157435, -0.0051166, -0.0156557, -0.0049784, -0.0096406, 0.0092287

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 144
type: A, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0060436
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B1_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B1_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0060924
time: 1.04 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0049782, 0.0100124, 0.0047205, 0.0099102, -0.0045648, 0.0048617
1: 0.0020415, 0.0027688, 0.0020043, 0.0027540, -0.0006595, 0.0007024
2: 0.0088243, 0.0116075, 0.0088808, 0.0117500, -0.0026879, 0.0025238
3: -0.0055540, -0.0026754, -0.0054955, -0.0025280, -0.0027800, 0.0026102
4: -0.0011407, 0.0019756, -0.0013002, 0.0019123, -0.0028257, 0.0030095
5: 0.0022438, 0.0051928, 0.0023037, 0.0053438, -0.0028480, 0.0026741
6: -0.0133976, -0.0016968, -0.0131600, -0.0010977, -0.0113000, 0.0106099
7: -0.0002458, 0.0156897, -0.0010618, 0.0153661, -0.0144497, 0.0153897
8: 0.9890407, 1.0002660, 0.9884659, 1.0000380, -0.0101787, 0.0108408
9: -0.0161288, -0.0059392, -0.0159218, -0.0054174, -0.0098406, 0.0092396

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0059365
time: 0.96 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0059451
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0047057, 0.0098120, 0.0046001, 0.0099191, -0.0046491, 0.0050294
1: 0.0020021, 0.0027398, 0.0019869, 0.0027553, -0.0006717, 0.0007266
2: 0.0089351, 0.0117582, 0.0088758, 0.0118166, -0.0027806, 0.0025703
3: -0.0054394, -0.0025196, -0.0055007, -0.0024592, -0.0028759, 0.0026584
4: -0.0013094, 0.0018515, -0.0013747, 0.0019178, -0.0028778, 0.0031133
5: 0.0023612, 0.0053524, 0.0022984, 0.0054143, -0.0029462, 0.0027234
6: -0.0129317, -0.0010634, -0.0131808, -0.0008180, -0.0116897, 0.0108057
7: -0.0011084, 0.0150552, -0.0014426, 0.0153944, -0.0147164, 0.0159204
8: 0.9884331, 0.9998190, 0.9881977, 1.0000581, -0.0103666, 0.0112147
9: -0.0157230, -0.0053876, -0.0159399, -0.0051739, -0.0101799, 0.0094101

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0060064
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0060067
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0049782, 0.0100124, 0.0044446, 0.0099028, -0.0045638, 0.0051893
1: 0.0020415, 0.0027688, 0.0019644, 0.0027530, -0.0006593, 0.0007497
2: 0.0088243, 0.0116075, 0.0088849, 0.0119025, -0.0028690, 0.0025232
3: -0.0055540, -0.0026754, -0.0054913, -0.0023703, -0.0029673, 0.0026096
4: -0.0011407, 0.0019756, -0.0014710, 0.0019077, -0.0028251, 0.0032122
5: 0.0022438, 0.0051928, 0.0023080, 0.0055054, -0.0030399, 0.0026735
6: -0.0133976, -0.0016968, -0.0131427, -0.0004566, -0.0120613, 0.0106075
7: -0.0002458, 0.0156897, -0.0019349, 0.0153426, -0.0144465, 0.0164264
8: 0.9890407, 1.0002660, 0.9878508, 1.0000215, -0.0101764, 0.0115711
9: -0.0161288, -0.0059392, -0.0159068, -0.0048591, -0.0105035, 0.0092375

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0058606
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0058698
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0047057, 0.0098120, 0.0043237, 0.0099115, -0.0046699, 0.0053481
1: 0.0020021, 0.0027398, 0.0019470, 0.0027542, -0.0006747, 0.0007727
2: 0.0089351, 0.0117582, 0.0088800, 0.0119694, -0.0029569, 0.0025819
3: -0.0054394, -0.0025196, -0.0054963, -0.0023012, -0.0030581, 0.0026703
4: -0.0013094, 0.0018515, -0.0015458, 0.0019131, -0.0028908, 0.0033106
5: 0.0023612, 0.0053524, 0.0023029, 0.0055762, -0.0031329, 0.0027356
6: -0.0129317, -0.0010634, -0.0131632, -0.0001755, -0.0124306, 0.0108542
7: -0.0011084, 0.0150552, -0.0023176, 0.0153704, -0.0147824, 0.0169293
8: 0.9884331, 0.9998190, 0.9875812, 1.0000410, -0.0104131, 0.0119254
9: -0.0157230, -0.0053876, -0.0159246, -0.0046144, -0.0108251, 0.0094523

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0059448
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0059448
time: 0.95 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: 0.0048611, 0.0100234, 0.0044569, 0.0097722, -0.0046887, 0.0049278
1: 0.0020246, 0.0027704, 0.0019662, 0.0027341, -0.0006774, 0.0007119
2: 0.0088182, 0.0116723, 0.0089571, 0.0118958, -0.0027245, 0.0025923
3: -0.0055603, -0.0026084, -0.0054166, -0.0023773, -0.0028178, 0.0026810
4: -0.0012132, 0.0019824, -0.0014634, 0.0018269, -0.0029024, 0.0030504
5: 0.0022373, 0.0052614, 0.0023845, 0.0054982, -0.0028867, 0.0027466
6: -0.0134232, -0.0014246, -0.0128393, -0.0004851, -0.0114536, 0.0108978
7: -0.0006166, 0.0157245, -0.0018961, 0.0149293, -0.0148419, 0.0155988
8: 0.9887796, 1.0002905, 0.9878783, 0.9997304, -0.0104550, 0.0109882
9: -0.0161510, -0.0057021, -0.0156425, -0.0048840, -0.0099743, 0.0094903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059898, upper bound: 0.0061269
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059898, upper bound: 0.0061558
time: 1.11 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: 0.0048611, 0.0100234, 0.0042088, 0.0097473, -0.0046695, 0.0052429
1: 0.0020246, 0.0027704, 0.0019304, 0.0027305, -0.0006746, 0.0007575
2: 0.0088182, 0.0116723, 0.0089708, 0.0120329, -0.0028987, 0.0025816
3: -0.0055603, -0.0026084, -0.0054024, -0.0022354, -0.0029980, 0.0026700
4: -0.0012132, 0.0019824, -0.0016170, 0.0018115, -0.0028905, 0.0032455
5: 0.0022373, 0.0052614, 0.0023991, 0.0056435, -0.0030713, 0.0027354
6: -0.0134232, -0.0014246, -0.0127814, 0.0000916, -0.0121860, 0.0108531
7: -0.0006166, 0.0157245, -0.0026814, 0.0148505, -0.0147810, 0.0165963
8: 0.9887796, 1.0002905, 0.9873250, 0.9996749, -0.0104121, 0.0116908
9: -0.0161510, -0.0057021, -0.0155921, -0.0043818, -0.0106121, 0.0094514

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059584, upper bound: 0.0060032
time: 1.15 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059584, upper bound: 0.0060179
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A2_B1

### Backsubstitution after applying IS history:
0: 0.0045718, 0.0098221, 0.0043172, 0.0097807, -0.0047620, 0.0051020
1: 0.0019828, 0.0027413, 0.0019460, 0.0027353, -0.0006880, 0.0007371
2: 0.0089295, 0.0118322, 0.0089524, 0.0119730, -0.0028208, 0.0026328
3: -0.0054452, -0.0024430, -0.0054215, -0.0022974, -0.0029174, 0.0027229
4: -0.0013922, 0.0018577, -0.0015498, 0.0018321, -0.0029477, 0.0031582
5: 0.0023553, 0.0054309, 0.0023795, 0.0055800, -0.0029887, 0.0027895
6: -0.0129552, -0.0007522, -0.0128591, -0.0001605, -0.0118584, 0.0110681
7: -0.0015323, 0.0150871, -0.0023382, 0.0149563, -0.0150738, 0.0161501
8: 0.9881345, 0.9998416, 0.9875668, 0.9997494, -0.0106183, 0.0113765
9: -0.0157435, -0.0051166, -0.0156598, -0.0046012, -0.0103268, 0.0096386

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.77 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061601
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061910
time: 1.12 seconds

## BFS IS instance: IS_A1_A1_B1_B2_B2_A2_B2

### Backsubstitution after applying IS history:
0: 0.0045718, 0.0098221, 0.0040721, 0.0097556, -0.0047552, 0.0054034
1: 0.0019828, 0.0027413, 0.0019106, 0.0027317, -0.0006870, 0.0007806
2: 0.0089295, 0.0118322, 0.0089662, 0.0121085, -0.0029874, 0.0026290
3: -0.0054452, -0.0024430, -0.0054072, -0.0021573, -0.0030897, 0.0027191
4: -0.0013922, 0.0018577, -0.0017016, 0.0018166, -0.0029436, 0.0033448
5: 0.0023553, 0.0054309, 0.0023942, 0.0057236, -0.0031653, 0.0027856
6: -0.0129552, -0.0007522, -0.0128008, 0.0004093, -0.0125589, 0.0110525
7: -0.0015323, 0.0150871, -0.0031141, 0.0148768, -0.0150525, 0.0171042
8: 0.9881345, 0.9998416, 0.9870202, 0.9996935, -0.0106033, 0.0120485
9: -0.0157435, -0.0051166, -0.0156090, -0.0041051, -0.0109369, 0.0096250

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.61 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061601
time: 1.07 seconds

## Relational analysis of IS_A1_A1_B1_B2_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061910
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0050124, 0.0100081, 0.0050871, 0.0102551, -0.0047124, 0.0040337
1: 0.0020465, 0.0027682, 0.0020572, 0.0028039, -0.0006808, 0.0005827
2: 0.0088267, 0.0115886, 0.0086901, 0.0115474, -0.0022301, 0.0026054
3: -0.0055515, -0.0026950, -0.0056928, -0.0027376, -0.0023065, 0.0026946
4: -0.0011195, 0.0019729, -0.0010733, 0.0021258, -0.0029171, 0.0024969
5: 0.0022463, 0.0051728, 0.0021016, 0.0051291, -0.0023629, 0.0027605
6: -0.0133875, -0.0017763, -0.0139617, -0.0019498, -0.0093754, 0.0109530
7: -0.0001376, 0.0156759, 0.0000987, 0.0164580, -0.0149170, 0.0127685
8: 0.9891169, 1.0002563, 0.9892834, 1.0008072, -0.0105078, 0.0089944
9: -0.0161200, -0.0060084, -0.0166200, -0.0061594, -0.0081645, 0.0095383

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058473, upper bound: 0.0059496
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059063, upper bound: 0.0059496
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0047342, 0.0098067, 0.0049742, 0.0102649, -0.0047744, 0.0041806
1: 0.0020063, 0.0027391, 0.0020409, 0.0028053, -0.0006898, 0.0006040
2: 0.0089380, 0.0117425, 0.0086847, 0.0116098, -0.0023114, 0.0026396
3: -0.0054364, -0.0025359, -0.0056984, -0.0026731, -0.0023905, 0.0027300
4: -0.0012917, 0.0018482, -0.0011432, 0.0021319, -0.0029554, 0.0025879
5: 0.0023643, 0.0053358, 0.0020959, 0.0051952, -0.0024490, 0.0027968
6: -0.0129195, -0.0011295, -0.0139845, -0.0016874, -0.0097170, 0.0110970
7: -0.0010184, 0.0150386, -0.0002587, 0.0164890, -0.0151131, 0.0132336
8: 0.9884965, 0.9998074, 0.9890316, 1.0008290, -0.0106460, 0.0093221
9: -0.0157124, -0.0054452, -0.0166398, -0.0059309, -0.0084620, 0.0096637

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0060702
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0060702
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0050124, 0.0100081, 0.0048213, 0.0102303, -0.0046955, 0.0044281
1: 0.0020465, 0.0027682, 0.0020188, 0.0028003, -0.0006784, 0.0006397
2: 0.0088267, 0.0115886, 0.0087038, 0.0116943, -0.0024482, 0.0025960
3: -0.0055515, -0.0026950, -0.0056786, -0.0025856, -0.0025320, 0.0026849
4: -0.0011195, 0.0019729, -0.0012378, 0.0021104, -0.0029066, 0.0027411
5: 0.0022463, 0.0051728, 0.0021162, 0.0052848, -0.0025940, 0.0027506
6: -0.0133875, -0.0017763, -0.0139040, -0.0013319, -0.0102922, 0.0109137
7: -0.0001376, 0.0156759, -0.0007427, 0.0163793, -0.0148635, 0.0140170
8: 0.9891169, 1.0002563, 0.9886906, 1.0007517, -0.0104702, 0.0098739
9: -0.0161200, -0.0060084, -0.0165697, -0.0056214, -0.0089629, 0.0095041

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0057751, upper bound: 0.0058378
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058068, upper bound: 0.0058378
time: 0.90 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0047342, 0.0098067, 0.0047046, 0.0102396, -0.0047827, 0.0045637
1: 0.0020063, 0.0027391, 0.0020020, 0.0028016, -0.0006910, 0.0006593
2: 0.0089380, 0.0117425, 0.0086987, 0.0117588, -0.0025232, 0.0026442
3: -0.0054364, -0.0025359, -0.0056839, -0.0025189, -0.0026096, 0.0027348
4: -0.0012917, 0.0018482, -0.0013101, 0.0021162, -0.0029606, 0.0028250
5: 0.0023643, 0.0053358, 0.0021107, 0.0053531, -0.0026734, 0.0028017
6: -0.0129195, -0.0011295, -0.0139257, -0.0010608, -0.0106074, 0.0111164
7: -0.0010184, 0.0150386, -0.0011121, 0.0164088, -0.0151395, 0.0144463
8: 0.9884965, 0.9998074, 0.9884305, 1.0007726, -0.0106646, 0.0101763
9: -0.0157124, -0.0054452, -0.0165886, -0.0053853, -0.0092374, 0.0096806

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=22, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0059906
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0059906
time: 1.21 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0048946, 0.0100191, 0.0048327, 0.0101282, -0.0048735, 0.0042956
1: 0.0020294, 0.0027698, 0.0020205, 0.0027855, -0.0007041, 0.0006206
2: 0.0088206, 0.0116537, 0.0087602, 0.0116880, -0.0023749, 0.0026944
3: -0.0055578, -0.0026276, -0.0056202, -0.0025922, -0.0024563, 0.0027867
4: -0.0011924, 0.0019797, -0.0012307, 0.0020472, -0.0030168, 0.0026590
5: 0.0022399, 0.0052418, 0.0021760, 0.0052780, -0.0025163, 0.0028549
6: -0.0134131, -0.0015025, -0.0136667, -0.0013586, -0.0099841, 0.0113274
7: -0.0005105, 0.0157108, -0.0007064, 0.0160562, -0.0154270, 0.0135975
8: 0.9888542, 1.0002809, 0.9887162, 1.0005242, -0.0108671, 0.0095784
9: -0.0161422, -0.0057699, -0.0163631, -0.0056447, -0.0086946, 0.0098644

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0061067
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0061534
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0045983, 0.0098169, 0.0046992, 0.0101375, -0.0049106, 0.0044769
1: 0.0019866, 0.0027406, 0.0020012, 0.0027869, -0.0007094, 0.0006468
2: 0.0089324, 0.0118176, 0.0087551, 0.0117618, -0.0024751, 0.0027149
3: -0.0054422, -0.0024582, -0.0056255, -0.0025158, -0.0025599, 0.0028079
4: -0.0013759, 0.0018545, -0.0013134, 0.0020530, -0.0030397, 0.0027713
5: 0.0023583, 0.0054154, 0.0021705, 0.0053563, -0.0026225, 0.0028766
6: -0.0129432, -0.0008137, -0.0136884, -0.0010482, -0.0104055, 0.0114136
7: -0.0014485, 0.0150707, -0.0011292, 0.0160857, -0.0155444, 0.0141714
8: 0.9881935, 0.9998301, 0.9884184, 1.0005449, -0.0109498, 0.0099826
9: -0.0157330, -0.0051701, -0.0163820, -0.0053743, -0.0090616, 0.0099395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0062957
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0063097
time: 1.23 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0048946, 0.0100191, 0.0045833, 0.0100873, -0.0048600, 0.0046650
1: 0.0020294, 0.0027698, 0.0019845, 0.0027796, -0.0007021, 0.0006740
2: 0.0088206, 0.0116537, 0.0087829, 0.0118259, -0.0025792, 0.0026869
3: -0.0055578, -0.0026276, -0.0055968, -0.0024496, -0.0026675, 0.0027790
4: -0.0011924, 0.0019797, -0.0013851, 0.0020219, -0.0030084, 0.0028877
5: 0.0022399, 0.0052418, 0.0021999, 0.0054242, -0.0027328, 0.0028470
6: -0.0134131, -0.0015025, -0.0135716, -0.0007789, -0.0108428, 0.0112959
7: -0.0005105, 0.0157108, -0.0014959, 0.0159266, -0.0153840, 0.0147669
8: 0.9888542, 1.0002809, 0.9881601, 1.0004330, -0.0108368, 0.0104021
9: -0.0161422, -0.0057699, -0.0162803, -0.0051398, -0.0094424, 0.0098369

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.20 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0060532
time: 0.95 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0060905
time: 0.88 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0045983, 0.0098169, 0.0044480, 0.0100962, -0.0049088, 0.0048312
1: 0.0019866, 0.0027406, 0.0019649, 0.0027809, -0.0007092, 0.0006980
2: 0.0089324, 0.0118176, 0.0087779, 0.0119007, -0.0026710, 0.0027140
3: -0.0054422, -0.0024582, -0.0056019, -0.0023722, -0.0027625, 0.0028069
4: -0.0013759, 0.0018545, -0.0014689, 0.0020274, -0.0030387, 0.0029906
5: 0.0023583, 0.0054154, 0.0021947, 0.0055034, -0.0028301, 0.0028756
6: -0.0129432, -0.0008137, -0.0135923, -0.0004644, -0.0112290, 0.0114095
7: -0.0014485, 0.0150707, -0.0019243, 0.0159549, -0.0155387, 0.0152929
8: 0.9881935, 0.9998301, 0.9878585, 1.0004529, -0.0109458, 0.0107726
9: -0.0157330, -0.0051701, -0.0162983, -0.0048659, -0.0097787, 0.0099359

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062375
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062679
time: 1.17 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: 0.0050124, 0.0100081, 0.0046309, 0.0102681, -0.0049081, 0.0049217
1: 0.0020465, 0.0027682, 0.0019913, 0.0028057, -0.0007091, 0.0007110
2: 0.0088267, 0.0115886, 0.0086829, 0.0117995, -0.0027211, 0.0027136
3: -0.0055515, -0.0026950, -0.0057002, -0.0024768, -0.0028143, 0.0028065
4: -0.0011195, 0.0019729, -0.0013557, 0.0021338, -0.0030382, 0.0030466
5: 0.0022463, 0.0051728, 0.0020940, 0.0053963, -0.0028831, 0.0028752
6: -0.0133875, -0.0017763, -0.0139919, -0.0008895, -0.0114393, 0.0114078
7: -0.0001376, 0.0156759, -0.0013452, 0.0164990, -0.0155365, 0.0155793
8: 0.9891169, 1.0002563, 0.9882663, 1.0008361, -0.0109442, 0.0109744
9: -0.0161200, -0.0060084, -0.0166463, -0.0052362, -0.0099619, 0.0099345

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.63 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059063, upper bound: 0.0061922
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059063, upper bound: 0.0062345
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: 0.0047342, 0.0098067, 0.0045079, 0.0102772, -0.0050145, 0.0050879
1: 0.0020063, 0.0027391, 0.0019736, 0.0028071, -0.0007244, 0.0007351
2: 0.0089380, 0.0117425, 0.0086778, 0.0118675, -0.0028130, 0.0027724
3: -0.0054364, -0.0025359, -0.0057054, -0.0024065, -0.0029093, 0.0028673
4: -0.0012917, 0.0018482, -0.0014318, 0.0021395, -0.0031041, 0.0031495
5: 0.0023643, 0.0053358, 0.0020887, 0.0054683, -0.0029805, 0.0029375
6: -0.0129195, -0.0011295, -0.0140131, -0.0006037, -0.0118256, 0.0116551
7: -0.0010184, 0.0150386, -0.0017345, 0.0165280, -0.0158732, 0.0161055
8: 0.9884965, 0.9998074, 0.9879920, 1.0008565, -0.0111814, 0.0113450
9: -0.0157124, -0.0054452, -0.0166648, -0.0049873, -0.0102983, 0.0101497

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.59 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0063269
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0063269
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: 0.0050124, 0.0100081, 0.0043803, 0.0102372, -0.0048874, 0.0052525
1: 0.0020465, 0.0027682, 0.0019551, 0.0028013, -0.0007061, 0.0007588
2: 0.0088267, 0.0115886, 0.0087000, 0.0119381, -0.0029040, 0.0027021
3: -0.0055515, -0.0026950, -0.0056825, -0.0023335, -0.0030035, 0.0027947
4: -0.0011195, 0.0019729, -0.0015108, 0.0021147, -0.0030254, 0.0032514
5: 0.0022463, 0.0051728, 0.0021121, 0.0055431, -0.0030769, 0.0028630
6: -0.0133875, -0.0017763, -0.0139200, -0.0003071, -0.0122084, 0.0113596
7: -0.0001376, 0.0156759, -0.0021385, 0.0164011, -0.0154708, 0.0166267
8: 0.9891169, 1.0002563, 0.9877074, 1.0007671, -0.0108980, 0.0117122
9: -0.0161200, -0.0060084, -0.0165837, -0.0047289, -0.0106316, 0.0098925

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.60 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058088, upper bound: 0.0060844
time: 1.04 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058068, upper bound: 0.0061187
time: 0.86 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: 0.0047342, 0.0098067, 0.0042515, 0.0102460, -0.0050135, 0.0054083
1: 0.0020063, 0.0027391, 0.0019365, 0.0028026, -0.0007243, 0.0007813
2: 0.0089380, 0.0117425, 0.0086951, 0.0120093, -0.0029901, 0.0027718
3: -0.0054364, -0.0025359, -0.0056876, -0.0022599, -0.0030925, 0.0028668
4: -0.0012917, 0.0018482, -0.0015905, 0.0021202, -0.0031034, 0.0033478
5: 0.0023643, 0.0053358, 0.0021070, 0.0056185, -0.0031682, 0.0029369
6: -0.0129195, -0.0011295, -0.0139406, -0.0000077, -0.0125703, 0.0116527
7: -0.0010184, 0.0150386, -0.0025463, 0.0164291, -0.0158700, 0.0171197
8: 0.9884965, 0.9998074, 0.9874203, 1.0007869, -0.0111792, 0.0120595
9: -0.0157124, -0.0054452, -0.0166016, -0.0044682, -0.0109468, 0.0101477

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=24, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0062510
time: 1.13 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0062510
time: 1.20 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: 0.0048946, 0.0100191, 0.0043608, 0.0101325, -0.0050320, 0.0049971
1: 0.0020294, 0.0027698, 0.0019523, 0.0027862, -0.0007270, 0.0007219
2: 0.0088206, 0.0116537, 0.0087579, 0.0119489, -0.0027628, 0.0027821
3: -0.0055578, -0.0026276, -0.0056227, -0.0023224, -0.0028574, 0.0028773
4: -0.0011924, 0.0019797, -0.0015229, 0.0020499, -0.0031149, 0.0030933
5: 0.0022399, 0.0052418, 0.0021734, 0.0055545, -0.0029273, 0.0029477
6: -0.0134131, -0.0015025, -0.0136768, -0.0002618, -0.0116147, 0.0116957
7: -0.0005105, 0.0157108, -0.0022002, 0.0160699, -0.0159286, 0.0158182
8: 0.9888542, 1.0002809, 0.9876640, 1.0005338, -0.0112204, 0.0111427
9: -0.0161422, -0.0057699, -0.0163719, -0.0046895, -0.0101146, 0.0101852

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0063415
time: 1.09 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0063897
time: 1.05 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: 0.0045983, 0.0098169, 0.0042184, 0.0101414, -0.0051151, 0.0051713
1: 0.0019866, 0.0027406, 0.0019317, 0.0027874, -0.0007390, 0.0007471
2: 0.0089324, 0.0118176, 0.0087530, 0.0120276, -0.0028591, 0.0028280
3: -0.0054422, -0.0024582, -0.0056277, -0.0022409, -0.0029570, 0.0029248
4: -0.0013759, 0.0018545, -0.0016110, 0.0020554, -0.0031663, 0.0032011
5: 0.0023583, 0.0054154, 0.0021682, 0.0056379, -0.0030294, 0.0029964
6: -0.0129432, -0.0008137, -0.0136974, 0.0000693, -0.0120196, 0.0118888
7: -0.0014485, 0.0150707, -0.0026512, 0.0160979, -0.0161915, 0.0163696
8: 0.9881935, 0.9998301, 0.9873464, 1.0005536, -0.0114056, 0.0115311
9: -0.0157330, -0.0051701, -0.0163898, -0.0044011, -0.0104672, 0.0103533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0065564
time: 0.85 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B1_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0065394
time: 1.10 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2_A1

### Backsubstitution after applying IS history:
0: 0.0048946, 0.0100191, 0.0041231, 0.0100891, -0.0050050, 0.0053133
1: 0.0020294, 0.0027698, 0.0019180, 0.0027799, -0.0007231, 0.0007676
2: 0.0088206, 0.0116537, 0.0087819, 0.0120803, -0.0029376, 0.0027671
3: -0.0055578, -0.0026276, -0.0055978, -0.0021864, -0.0030382, 0.0028619
4: -0.0011924, 0.0019797, -0.0016700, 0.0020230, -0.0030982, 0.0032890
5: 0.0022399, 0.0052418, 0.0021989, 0.0056938, -0.0031125, 0.0029319
6: -0.0134131, -0.0015025, -0.0135757, 0.0002909, -0.0123495, 0.0116330
7: -0.0005105, 0.0157108, -0.0029528, 0.0159323, -0.0158431, 0.0168189
8: 0.9888542, 1.0002809, 0.9871338, 1.0004369, -0.0111602, 0.0118476
9: -0.0161422, -0.0057699, -0.0162839, -0.0042082, -0.0107544, 0.0101305

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=25, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.94 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 176

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0062966
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0063508
time: 1.07 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2_B2_A2

### Backsubstitution after applying IS history:
0: 0.0045983, 0.0098169, 0.0039777, 0.0100976, -0.0050971, 0.0054751
1: 0.0019866, 0.0027406, 0.0018970, 0.0027811, -0.0007364, 0.0007910
2: 0.0089324, 0.0118176, 0.0087772, 0.0121607, -0.0030270, 0.0028180
3: -0.0054422, -0.0024582, -0.0056027, -0.0021033, -0.0031307, 0.0029146
4: -0.0013759, 0.0018545, -0.0017600, 0.0020283, -0.0031552, 0.0033892
5: 0.0023583, 0.0054154, 0.0021939, 0.0057789, -0.0032073, 0.0029859
6: -0.0129432, -0.0008137, -0.0135955, 0.0006288, -0.0127256, 0.0118470
7: -0.0014485, 0.0150707, -0.0034130, 0.0159592, -0.0161346, 0.0173312
8: 0.9881935, 0.9998301, 0.9868096, 1.0004559, -0.0113656, 0.0122085
9: -0.0157330, -0.0051701, -0.0163011, -0.0039140, -0.0110820, 0.0103169

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=25, inp2_unstable=26, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 126
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 197

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0065084
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_B2_A2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0065077
time: 1.07 seconds

## BFS IS instance: IS_A1_A2_B1_B1_B1_A1_A1

### Backsubstitution after applying IS history:
0: 0.0047617, 0.0098693, 0.0051306, 0.0100473, -0.0045544, 0.0040724
1: 0.0020102, 0.0027481, 0.0020635, 0.0027738, -0.0006580, 0.0005884
2: 0.0089034, 0.0117272, 0.0088050, 0.0115233, -0.0022515, 0.0025180
3: -0.0054722, -0.0025516, -0.0055740, -0.0027625, -0.0023287, 0.0026043
4: -0.0012747, 0.0018870, -0.0010464, 0.0019972, -0.0028193, 0.0025209
5: 0.0023276, 0.0053196, 0.0022234, 0.0051036, -0.0023856, 0.0026680
6: -0.0130649, -0.0011936, -0.0134787, -0.0020508, -0.0094655, 0.0105858
7: -0.0009312, 0.0152366, 0.0002363, 0.0158001, -0.0144169, 0.0128912
8: 0.9885579, 0.9999469, 0.9893804, 1.0003438, -0.0101556, 0.0090808
9: -0.0158390, -0.0055009, -0.0161994, -0.0062475, -0.0082430, 0.0092186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=24, inp2_unstable=23, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=1, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 77
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 247
type: A, layer: 1, pos: 247
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 197
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 126
type: A, layer: 1, pos: 126
type: A, layer: 1, pos: 211
type: B, layer: 1, pos: 211
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 215
type: A, layer: 1, pos: 215
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 144
type: B, layer: 1, pos: 144

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 77

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058241, upper bound: 0.0057801
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B1_B1_B1_A1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0058241, upper bound: 0.0057801
time: 1.10 seconds

## Summary of splitting at layer (split count: 7)
- Time for IS candidates: 3.85 seconds
IS_A1_A1_B1_B1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0057764
IS_A1_A1_B1_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0058114
IS_A1_A1_B1_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0059117
IS_A1_A1_B1_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0059117
IS_A1_A1_B1_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0056924
IS_A1_A1_B1_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0057163
IS_A1_A1_B1_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0058407
IS_A1_A1_B1_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0058407
IS_A1_A1_B1_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060318, upper bound: 0.0059087
IS_A1_A1_B1_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060318, upper bound: 0.0059515
IS_A1_A1_B1_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0060993
IS_A1_A1_B1_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0061279
IS_A1_A1_B1_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0058699
IS_A1_A1_B1_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059062
IS_A1_A1_B1_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0060436
IS_A1_A1_B1_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0060924
IS_A1_A1_B1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0059365
IS_A1_A1_B1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059166, upper bound: 0.0059451
IS_A1_A1_B1_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0060064
IS_A1_A1_B1_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0060067
IS_A1_A1_B1_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0058606
IS_A1_A1_B1_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0058698
IS_A1_A1_B1_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0059448
IS_A1_A1_B1_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0059448
IS_A1_A1_B1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059898, upper bound: 0.0061269
IS_A1_A1_B1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059898, upper bound: 0.0061558
IS_A1_A1_B1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059584, upper bound: 0.0060032
IS_A1_A1_B1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059584, upper bound: 0.0060179
IS_A1_A1_B1_B2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061601
IS_A1_A1_B1_B2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061910
IS_A1_A1_B1_B2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061601
IS_A1_A1_B1_B2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061607, upper bound: 0.0061910
IS_A1_A1_B2_B1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058473, upper bound: 0.0059496
IS_A1_A1_B2_B1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059063, upper bound: 0.0059496
IS_A1_A1_B2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0060702
IS_A1_A1_B2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0060702
IS_A1_A1_B2_B1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0057751, upper bound: 0.0058378
IS_A1_A1_B2_B1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058068, upper bound: 0.0058378
IS_A1_A1_B2_B1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0059906
IS_A1_A1_B2_B1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0059906
IS_A1_A1_B2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0061067
IS_A1_A1_B2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0061534
IS_A1_A1_B2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0062957
IS_A1_A1_B2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0063097
IS_A1_A1_B2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0060532
IS_A1_A1_B2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0060905
IS_A1_A1_B2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062375
IS_A1_A1_B2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062679
IS_A1_A1_B2_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059063, upper bound: 0.0061922
IS_A1_A1_B2_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059063, upper bound: 0.0062345
IS_A1_A1_B2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060931, upper bound: 0.0063269
IS_A1_A1_B2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061451, upper bound: 0.0063269
IS_A1_A1_B2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058088, upper bound: 0.0060844
IS_A1_A1_B2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058068, upper bound: 0.0061187
IS_A1_A1_B2_B2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0062510
IS_A1_A1_B2_B2_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0062510
IS_A1_A1_B2_B2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0063415
IS_A1_A1_B2_B2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0060311, upper bound: 0.0063897
IS_A1_A1_B2_B2_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0065564
IS_A1_A1_B2_B2_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062191, upper bound: 0.0065394
IS_A1_A1_B2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0062966
IS_A1_A1_B2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0063508
IS_A1_A1_B2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0065084
IS_A1_A1_B2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0065077
IS_A1_A2_B1_B1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058241, upper bound: 0.0057801
IS_A1_A2_B1_B1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 8, time: 3.85
Output dim: 8, lower bound: -0.0058241, upper bound: 0.0057801
IS_A1_A2_B1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0057930
IS_A1_A2_B1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0059212
IS_A1_A2_B1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0059212
IS_A1_A2_B1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059691
IS_A1_A2_B1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0059964
IS_A1_A2_B1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0061303
IS_A1_A2_B1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062129
IS_A1_A2_B1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0059714
IS_A1_A2_B1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0058734, upper bound: 0.0059666
IS_A1_A2_B1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060483
IS_A1_A2_B1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060491
IS_A1_A2_B1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0061332
IS_A1_A2_B1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060047, upper bound: 0.0061319
IS_A1_A2_B1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0062646
IS_A1_A2_B1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063272
IS_A1_A2_B2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0057749, upper bound: 0.0059059
IS_A1_A2_B2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0058055, upper bound: 0.0059058
IS_A1_A2_B2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0060632
IS_A1_A2_B2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0060632
IS_A1_A2_B2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0061533
IS_A1_A2_B2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0061781
IS_A1_A2_B2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063210
IS_A1_A2_B2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0063898
IS_A1_A2_B2_B2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060330, upper bound: 0.0060397
IS_A1_A2_B2_B2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060492, upper bound: 0.0060295
IS_A1_A2_B2_B2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061237, upper bound: 0.0063404
IS_A1_A2_B2_B2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061508, upper bound: 0.0063404
IS_A1_A2_B2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0059794, upper bound: 0.0064136
IS_A1_A2_B2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0059793, upper bound: 0.0064463
IS_A1_A2_B2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0066044
IS_A1_A2_B2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062404, upper bound: 0.0066405
IS_A2_B1_B1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060605, upper bound: 0.0060238
IS_A2_B1_B1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061146, upper bound: 0.0062453
IS_A2_B1_B1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060605, upper bound: 0.0060753
IS_A2_B1_B1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061146, upper bound: 0.0063210
IS_A2_B1_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061330
IS_A2_B1_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061821, upper bound: 0.0061899
IS_A2_B1_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0063314
IS_A2_B1_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062145, upper bound: 0.0064313
IS_A2_B1_B1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063281, upper bound: 0.0060058
IS_A2_B1_B1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064169, upper bound: 0.0062453
IS_A2_B1_B1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063281, upper bound: 0.0060454
IS_A2_B1_B1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064169, upper bound: 0.0063210
IS_A2_B1_B1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065161, upper bound: 0.0061322
IS_A2_B1_B1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065661, upper bound: 0.0063314
IS_A2_B1_B1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065161, upper bound: 0.0061849
IS_A2_B1_B1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065661, upper bound: 0.0064313
IS_A2_B1_B2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0059714, upper bound: 0.0058734
IS_A2_B1_B2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0059667, upper bound: 0.0058734
IS_A2_B1_B2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060483, upper bound: 0.0061508
IS_A2_B1_B2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060491, upper bound: 0.0061508
IS_A2_B1_B2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061332, upper bound: 0.0060047
IS_A2_B1_B2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061319, upper bound: 0.0060047
IS_A2_B1_B2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062646, upper bound: 0.0062404
IS_A2_B1_B2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062646, upper bound: 0.0063898
IS_A2_B1_B2_A2_A1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060397, upper bound: 0.0060330
IS_A2_B1_B2_A2_A1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0060295, upper bound: 0.0060492
IS_A2_B1_B2_A2_A1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063404, upper bound: 0.0061237
IS_A2_B1_B2_A2_A1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063404, upper bound: 0.0061508
IS_A2_B1_B2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064136, upper bound: 0.0059794
IS_A2_B1_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064463, upper bound: 0.0059793
IS_A2_B1_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0066044, upper bound: 0.0062404
IS_A2_B1_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0066044, upper bound: 0.0063898
IS_A2_B2_A1_B1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0060257
IS_A2_B2_A1_B1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0060536
IS_A2_B2_A1_B1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0060257
IS_A2_B2_A1_B1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0060536
IS_A2_B2_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0060922
IS_A2_B2_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0060922
IS_A2_B2_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0061086
IS_A2_B2_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0061086
IS_A2_B2_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061875
IS_A2_B2_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0062252
IS_A2_B2_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061875
IS_A2_B2_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065473, upper bound: 0.0062252
IS_A2_B2_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0063198
IS_A2_B2_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0063408
IS_A2_B2_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064440, upper bound: 0.0063198
IS_A2_B2_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065801, upper bound: 0.0063408
IS_A2_B2_A2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062239, upper bound: 0.0059255
IS_A2_B2_A2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062239, upper bound: 0.0059483
IS_A2_B2_A2_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065921, upper bound: 0.0060818
IS_A2_B2_A2_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065921, upper bound: 0.0061697
IS_A2_B2_A2_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061594
IS_A2_B2_A2_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063830, upper bound: 0.0061960
IS_A2_B2_A2_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0061812
IS_A2_B2_A2_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0062178
IS_A2_B2_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0061716, upper bound: 0.0060913
IS_A2_B2_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0062061, upper bound: 0.0060913
IS_A2_B2_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064651, upper bound: 0.0061992
IS_A2_B2_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0064984, upper bound: 0.0061992
IS_A2_B2_A2_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063748, upper bound: 0.0063659
IS_A2_B2_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0063748, upper bound: 0.0063769
IS_A2_B2_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0063952
IS_A2_B2_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.85
Output dim: 8, lower bound: -0.0065659, upper bound: 0.0064085

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.64 + 597.64 = 601.28 seconds
