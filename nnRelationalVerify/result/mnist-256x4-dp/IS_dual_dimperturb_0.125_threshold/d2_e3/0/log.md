## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00379488


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=6, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0003518, 0.0000312, -0.0003518, 0.0000312, -0.0002068, 0.0002068)
1: (-0.0003267, 0.0013695, -0.0003267, 0.0013695, -0.0008952, 0.0008952)
2: (0.0142890, 0.0168293, 0.0142890, 0.0168293, -0.0013377, 0.0013377)
3: (0.0001178, 0.0020280, 0.0001178, 0.0020280, -0.0010046, 0.0010046)
4: (-0.0042709, -0.0025090, -0.0042709, -0.0025090, -0.0009372, 0.0009372)
5: (0.0080558, 0.0099625, 0.0080558, 0.0099625, -0.0010026, 0.0010026)
6: (0.0091734, 0.0098929, 0.0091734, 0.0098929, -0.0004136, 0.0004136)
7: (-0.0200269, -0.0158877, -0.0200269, -0.0158877, -0.0021605, 0.0021605)
8: (0.9664114, 0.9782706, 0.9664114, 0.9782706, -0.0062481, 0.0062481)
9: (0.0040394, 0.0075248, 0.0040394, 0.0075248, -0.0018246, 0.0018246)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.39 + 1.29 = 2.68 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.0043588, upper bound: 0.0043588

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 109
type: A, layer: 1, pos: 109
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 109

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041234, upper bound: 0.0036818
time: 0.48 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0041234, upper bound: 0.0041235
time: 0.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 1.07 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 8, lower bound: -0.0041234, upper bound: 0.0036818
IS_B2, status: Status.UNKNOWN, split count: 1, time: 1.07
Output dim: 8, lower bound: -0.0041234, upper bound: 0.0041235

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -0.0003428, 0.0000309, -0.0003192, 0.0000301, -0.0001940, 0.0001658
1: -0.0002845, 0.0013690, -0.0001743, 0.0013677, -0.0008410, 0.0007068
2: 0.0142897, 0.0167660, 0.0142916, 0.0166010, -0.0010547, 0.0012561
3: 0.0001184, 0.0019804, 0.0001198, 0.0018563, -0.0007913, 0.0009429
4: -0.0042705, -0.0025529, -0.0042691, -0.0026673, -0.0007433, 0.0008814
5: 0.0080563, 0.0099150, 0.0080578, 0.0097911, -0.0007897, 0.0009411
6: 0.0091913, 0.0098927, 0.0092380, 0.0098921, -0.0003920, 0.0003420
7: -0.0199238, -0.0158888, -0.0196549, -0.0158920, -0.0020251, 0.0016954
8: 0.9667068, 0.9782676, 0.9674773, 0.9782584, -0.0058675, 0.0049278
9: 0.0040403, 0.0074380, 0.0040430, 0.0072116, -0.0014337, 0.0017108

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.29 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0036818
time: 0.46 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0036818
time: 0.49 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -0.0003476, 0.0000310, -0.0003388, 0.0001052, -0.0002942, 0.0001775
1: -0.0003072, 0.0013691, -0.0002657, 0.0014829, -0.0011000, 0.0008048
2: 0.0142896, 0.0168000, 0.0141192, 0.0167380, -0.0011854, 0.0016446
3: 0.0001182, 0.0020060, -0.0000099, 0.0019594, -0.0008833, 0.0012355
4: -0.0042706, -0.0025293, -0.0043887, -0.0025723, -0.0008905, 0.0011491
5: 0.0080562, 0.0099405, 0.0079283, 0.0098940, -0.0008809, 0.0012331
6: 0.0091816, 0.0098927, 0.0091992, 0.0099410, -0.0004984, 0.0005663
7: -0.0199792, -0.0158886, -0.0198781, -0.0156110, -0.0026620, 0.0018393
8: 0.9665480, 0.9782681, 0.9668378, 0.9790636, -0.0076808, 0.0055536
9: 0.0040401, 0.0074847, 0.0038064, 0.0073995, -0.0015707, 0.0022466

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=6, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 109
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 109

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0041234
time: 0.50 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0041235
time: 0.44 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 2.41 seconds
IS_B1_A1, status: Status.VERIFIED, split count: 2, time: 2.41
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0036818
IS_B1_A2, status: Status.VERIFIED, split count: 2, time: 2.41
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0036818
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0041234
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 2.41
Output dim: 8, lower bound: -0.0036818, upper bound: 0.0041235

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -0.0003192, 0.0000301, -0.0003388, 0.0001052, -0.0002541, 0.0002044
1: -0.0001743, 0.0013677, -0.0002657, 0.0014829, -0.0009158, 0.0009136
2: 0.0142916, 0.0166010, 0.0141192, 0.0167380, -0.0013571, 0.0013677
3: 0.0001198, 0.0018563, -0.0000099, 0.0019594, -0.0010160, 0.0010266
4: -0.0042691, -0.0026673, -0.0043887, -0.0025723, -0.0009781, 0.0009604
5: 0.0080578, 0.0097911, 0.0079283, 0.0098940, -0.0010137, 0.0010246
6: 0.0092380, 0.0098921, 0.0091992, 0.0099410, -0.0004306, 0.0005317
7: -0.0196549, -0.0158920, -0.0198781, -0.0156110, -0.0022053, 0.0021519
8: 0.9674773, 0.9782584, 0.9668378, 0.9790636, -0.0063889, 0.0063469
9: 0.0040430, 0.0072116, 0.0038064, 0.0073995, -0.0018266, 0.0018631

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 134
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 191

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0035329, upper bound: 0.0039746
time: 0.45 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.0035337, upper bound: 0.0039746
time: 0.54 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -0.0003388, 0.0001052, -0.0003388, 0.0001052, -0.0002281, 0.0002281
1: -0.0002657, 0.0014829, -0.0002657, 0.0014829, -0.0008168, 0.0008168
2: 0.0141192, 0.0167380, 0.0141192, 0.0167380, -0.0012033, 0.0012033
3: -0.0000099, 0.0019594, -0.0000099, 0.0019594, -0.0008968, 0.0008968
4: -0.0043887, -0.0025723, -0.0043887, -0.0025723, -0.0009029, 0.0009029
5: 0.0079283, 0.0098940, 0.0079283, 0.0098940, -0.0008944, 0.0008944
6: 0.0091992, 0.0099410, 0.0091992, 0.0099410, -0.0005714, 0.0005714
7: -0.0198781, -0.0156110, -0.0198781, -0.0156110, -0.0018685, 0.0018685
8: 0.9668378, 0.9790636, 0.9668378, 0.9790636, -0.0056371, 0.0056371
9: 0.0038064, 0.0073995, 0.0038064, 0.0073995, -0.0015953, 0.0015953

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=5, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=3, inp2_unstable=3, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 191
type: B, layer: 1, pos: 191
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 191

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035337, upper bound: 0.0035329
time: 0.47 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0035337, upper bound: 0.0035337
time: 0.48 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 2.35 seconds
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0035329, upper bound: 0.0039746
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0035337, upper bound: 0.0039746
IS_B2_A2_A1, status: Status.VERIFIED, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0035337, upper bound: 0.0035329
IS_B2_A2_A2, status: Status.VERIFIED, split count: 3, time: 2.35
Output dim: 8, lower bound: -0.0035337, upper bound: 0.0035337

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.0003192, 0.0000275, -0.0003387, 0.0000867, -0.0002360, 0.0002014
1: -0.0001742, 0.0013638, -0.0002655, 0.0014544, -0.0008813, 0.0008970
2: 0.0142976, 0.0166009, 0.0141618, 0.0167376, -0.0013357, 0.0013164
3: 0.0001243, 0.0018563, 0.0000222, 0.0019591, -0.0010008, 0.0009883
4: -0.0042650, -0.0026674, -0.0043592, -0.0025726, -0.0009531, 0.0009236
5: 0.0080622, 0.0097911, 0.0079603, 0.0098937, -0.0009986, 0.0009864
6: 0.0092380, 0.0098905, 0.0091993, 0.0099289, -0.0004104, 0.0004850
7: -0.0196548, -0.0159016, -0.0198776, -0.0156805, -0.0021242, 0.0021312
8: 0.9674774, 0.9782309, 0.9668393, 0.9788644, -0.0061492, 0.0062436
9: 0.0040511, 0.0072115, 0.0038649, 0.0073991, -0.0018057, 0.0017942

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 134
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 134

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033308, upper bound: 0.0037379
time: 0.48 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033158, upper bound: 0.0037404
time: 0.47 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.0003192, 0.0000179, -0.0003457, 0.0000693, -0.0002252, 0.0002048
1: -0.0001742, 0.0013491, -0.0002980, 0.0014279, -0.0008748, 0.0009372
2: 0.0143196, 0.0166009, 0.0142016, 0.0167862, -0.0013968, 0.0013081
3: 0.0001408, 0.0018562, 0.0000521, 0.0019956, -0.0010476, 0.0009827
4: -0.0042497, -0.0026674, -0.0043316, -0.0025389, -0.0009902, 0.0009135
5: 0.0080787, 0.0097910, 0.0079902, 0.0099301, -0.0010454, 0.0009809
6: 0.0092381, 0.0098842, 0.0091856, 0.0099176, -0.0003901, 0.0004704
7: -0.0196547, -0.0159375, -0.0199567, -0.0157452, -0.0021192, 0.0022384
8: 0.9674779, 0.9781281, 0.9666126, 0.9786789, -0.0061089, 0.0065275
9: 0.0040813, 0.0072114, 0.0039194, 0.0074657, -0.0018953, 0.0017880

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=5, inp2_unstable=4, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=4, inp2_unstable=4, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=0, inp2_unstable=0, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=1, inp2_unstable=1, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=2, inp2_unstable=2, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 134
type: A, layer: 1, pos: 134
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 191
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 134

## Relational analysis of IS_B2_A1_B2_B1

### Relational analysis result of IS_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033152, upper bound: 0.0037721
time: 0.47 seconds

## Relational analysis of IS_B2_A1_B2_B2

### Relational analysis result of IS_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.0033156, upper bound: 0.0037404
time: 0.50 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 2.41 seconds
IS_B2_A1_B1_A1, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 8, lower bound: -0.0033308, upper bound: 0.0037379
IS_B2_A1_B1_A2, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 8, lower bound: -0.0033158, upper bound: 0.0037404
IS_B2_A1_B2_B1, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 8, lower bound: -0.0033152, upper bound: 0.0037721
IS_B2_A1_B2_B2, status: Status.VERIFIED, split count: 4, time: 2.41
Output dim: 8, lower bound: -0.0033156, upper bound: 0.0037404

## IS Result
status: Status.VERIFIED
execution time: (base) + (is) = 2.68 + 15.40 = 18.07 seconds
